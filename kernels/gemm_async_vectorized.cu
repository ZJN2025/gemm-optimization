#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include "gemm.hpp"

// 以 float4 (16B) 为单位，把第 tile_k 个 tile 的 A、B 子块异步拷进 stage 号缓冲区。
// 越界时不能走 16B 拷贝（会读越界），退化为逐元素处理：合法元素取值、非法元素填 0。
// 前提：K、N 为 4 的倍数，保证全局地址 16 字节对齐。
template <int BM, int BN, int BK>
__device__ __forceinline__ void load_tile_async(
    int M,
    int N,
    int K,
    const float* A,
    const float* B,
    float (*As)[BM][BK],
    float (*Bs)[BK][BN],
    int stage,
    int tile_k,
    int block_row,
    int block_col,
    int a_row,
    int a_c4,
    int b_row,
    int b_c4) {

    // ----------------------------
    // A tile：沿 K 方向取 4 个连续元素
    // ----------------------------

    const int a_gr = block_row + a_row;
    const int a_gc = tile_k + a_c4;

    if (a_gr < M && a_gc + 3 < K) {

        __pipeline_memcpy_async(
            &As[stage][a_row][a_c4],
            &A[a_gr * K + a_gc],
            sizeof(float4));

    } else {

        for (int i = 0; i < 4; ++i) {

            As[stage][a_row][a_c4 + i] =
                (a_gr < M && a_gc + i < K)
                    ? A[a_gr * K + a_gc + i]
                    : 0.0f;
        }
    }

    // ----------------------------
    // B tile：沿 N 方向取 4 个连续元素
    // ----------------------------

    const int b_gr = tile_k + b_row;
    const int b_gc = block_col + b_c4;

    if (b_gr < K && b_gc + 3 < N) {

        __pipeline_memcpy_async(
            &Bs[stage][b_row][b_c4],
            &B[b_gr * N + b_gc],
            sizeof(float4));

    } else {

        for (int i = 0; i < 4; ++i) {

            Bs[stage][b_row][b_c4 + i] =
                (b_gr < K && b_gc + i < N)
                    ? B[b_gr * N + b_gc + i]
                    : 0.0f;
        }
    }
}

template <
    int BM,
    int BN,
    int BK,
    int TM,
    int TN>
__global__ void gemm_async_vectorized(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C) {

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid =
        ty * blockDim.x + tx;

    const int block_row =
        blockIdx.y * BM;

    const int block_col =
        blockIdx.x * BN;

    // A tile：每个线程负责一个 float4
    // （要求 BM * BK / 4 == 线程数）
    constexpr int A_VEC_PER_ROW =
        BK / 4;

    const int a_row =
        tid / A_VEC_PER_ROW;

    const int a_c4 =
        (tid % A_VEC_PER_ROW) * 4;

    // B tile：每个线程负责一个 float4
    // （要求 BK * BN / 4 == 线程数）
    constexpr int B_VEC_PER_ROW =
        BN / 4;

    const int b_row =
        tid / B_VEC_PER_ROW;

    const int b_c4 =
        (tid % B_VEC_PER_ROW) * 4;

    float accum[TM][TN] = {};

    // --------------------------------
    // 先把第 0 个 tile 拷进来
    // --------------------------------

    load_tile_async<BM, BN, BK>(
        M,
        N,
        K,
        A,
        B,
        As,
        Bs,
        0,
        0,
        block_row,
        block_col,
        a_row,
        a_c4,
        b_row,
        b_c4);

    __pipeline_commit();
    __pipeline_wait_prior(0);

    __syncthreads();

    const int tiles =
        (K + BK - 1) / BK;

    int current = 0;

    for (int tile = 0;
         tile < tiles;
         ++tile) {

        const int next =
            current ^ 1;

        const bool has_next =
            tile + 1 < tiles;

        // 把下一个 tile 异步拷进另一块缓冲区，
        // 拷贝与当前 tile 的计算重叠
        if (has_next) {

            load_tile_async<BM, BN, BK>(
                M,
                N,
                K,
                A,
                B,
                As,
                Bs,
                next,
                (tile + 1) * BK,
                block_row,
                block_col,
                a_row,
                a_c4,
                b_row,
                b_c4);

            __pipeline_commit();
        }

        // ----------------------------
        // 计算 current 缓冲区里的 tile
        // ----------------------------

        const int thread_row =
            ty * TM;

        const int thread_col =
            tx * TN;

        for (int k = 0; k < BK; ++k) {

            float reg_a[TM];
            float reg_b[TN];

            for (int i = 0; i < TM; ++i) {
                reg_a[i] =
                    As[current]
                      [thread_row + i][k];
            }

            for (int j = 0; j < TN; ++j) {
                reg_b[j] =
                    Bs[current]
                      [k][thread_col + j];
            }

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {

                    accum[i][j] +=
                        reg_a[i] *
                        reg_b[j];
                }
            }
        }

        // 等下一个 tile 的异步拷贝完成，
        // 下一轮迭代才轮到它当 current
        if (has_next) {

            __pipeline_wait_prior(0);

            __syncthreads();
        }

        current = next;
    }

    // ----------------------------
    // 写回 C
    // ----------------------------

    const int row_base =
        block_row + ty * TM;

    const int col_base =
        block_col + tx * TN;

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {

            const int row =
                row_base + i;

            const int col =
                col_base + j;

            if (row < M && col < N) {

                C[row * N + col] =
                    alpha * accum[i][j] +
                    beta * C[row * N + col];
            }
        }
    }
}

// 要求 K、N 为 4 的倍数（float4 全局访存需要 16B 对齐）
void launch_gemm_async_vectorized(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C) {

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    constexpr int TM = 4;
    constexpr int TN = 4;

    dim3 block(
        BN / TN,
        BM / TM);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    gemm_async_vectorized<
        BM,
        BN,
        BK,
        TM,
        TN>
        <<<grid, block>>>(
            M,
            N,
            K,
            alpha,
            A,
            B,
            beta,
            C);
}
