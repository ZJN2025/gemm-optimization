#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include "gemm.hpp"

// ============================================================
// 对照实验用 kernel：所有变量（tile 尺寸 / 寄存器分块 / K 分块 / 流水深度）
// 都是模板参数，骨架 = cp.async + float4 + N-stage 流水（STAGES=2 即双缓冲）。
// 每个 launcher 只改变一个变量，用来做单变量控制的性能对照。
// 与 gemm_simt_128x128 同款设计，但加载路径按 (tile 元素数 / 线程数) 循环，
// 适配任意 tile 配置（大 tile 时每个线程拷贝多个 float4）。
// 要求 K、N 为 4 的倍数（float4 全局访存需要 16B 对齐）。
// ============================================================

// 把第 tile_k 个 tile 的 A、B 子块以 float4 为单位异步拷进 stage 号缓冲区；
// 越界时不能走 16B 拷贝（会读越界），退化为逐元素处理：合法元素取值、非法元素填 0。
template <int BM, int BN, int BK, int THREADS>
__device__ __forceinline__ void load_stage_async(
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
    int tid) {

    // ----------------------------
    // A tile：沿 K 方向取 4 个连续元素
    // ----------------------------

    constexpr int A_VEC_PER_ROW = BK / 4;
    constexpr int A_VECS = BM * A_VEC_PER_ROW;

    for (int vec = tid; vec < A_VECS; vec += THREADS) {

        const int a_row = vec / A_VEC_PER_ROW;
        const int a_c4 = (vec % A_VEC_PER_ROW) * 4;

        const int a_gr = block_row + a_row;
        const int a_gc = tile_k + a_c4;

        if (a_gr < M && a_gc + 3 < K) {

            __pipeline_memcpy_async(
                &As[stage][a_row][a_c4],
                &A[a_gr * K + a_gc],
                sizeof(float4));

        } else {

#pragma unroll
            for (int i = 0; i < 4; ++i) {

                As[stage][a_row][a_c4 + i] =
                    (a_gr < M && a_gc + i < K)
                        ? A[a_gr * K + a_gc + i]
                        : 0.0f;
            }
        }
    }

    // ----------------------------
    // B tile：沿 N 方向取 4 个连续元素
    // ----------------------------

    constexpr int B_VEC_PER_ROW = BN / 4;
    constexpr int B_VECS = BK * B_VEC_PER_ROW;

    for (int vec = tid; vec < B_VECS; vec += THREADS) {

        const int b_row = vec / B_VEC_PER_ROW;
        const int b_c4 = (vec % B_VEC_PER_ROW) * 4;

        const int b_gr = tile_k + b_row;
        const int b_gc = block_col + b_c4;

        if (b_gr < K && b_gc + 3 < N) {

            __pipeline_memcpy_async(
                &Bs[stage][b_row][b_c4],
                &B[b_gr * N + b_gc],
                sizeof(float4));

        } else {

#pragma unroll
            for (int i = 0; i < 4; ++i) {

                Bs[stage][b_row][b_c4 + i] =
                    (b_gr < K && b_gc + i < N)
                        ? B[b_gr * N + b_gc + i]
                        : 0.0f;
            }
        }
    }
}

template <
    int BM,
    int BN,
    int BK,
    int TM,
    int TN,
    int STAGES>
__global__ void gemm_simt_generic(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C) {

    __shared__ float As[STAGES][BM][BK];
    __shared__ float Bs[STAGES][BK][BN];

    // block 维度固定为 (BN/TN, BM/TM)，线程总数可由此推出来
    constexpr int THREADS =
        (BM / TM) * (BN / TN);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid =
        ty * blockDim.x + tx;

    const int block_row =
        blockIdx.y * BM;

    const int block_col =
        blockIdx.x * BN;

    float accum[TM][TN] = {};

    const int tiles =
        (K + BK - 1) / BK;

    // --------------------------------
    // Prologue：预装前 STAGES-1 个 tile（K 不够多时只装能装的）
    // --------------------------------

    const int prologue =
        (STAGES - 1 < tiles) ? STAGES - 1 : tiles;

    for (int s = 0; s < prologue; ++s) {

        load_stage_async<BM, BN, BK, THREADS>(
            M, N, K, A, B, As, Bs,
            s, s * BK,
            block_row, block_col, tid);

        __pipeline_commit();
    }

    // 等最老的一组（tile 0）落地；只装了一组时就是 wait_prior(0)
    __pipeline_wait_prior(prologue - 1);

    __syncthreads();

    // --------------------------------
    // 主循环：算 tile t 的同时异步拷 tile t+STAGES-1
    // --------------------------------

    for (int tile = 0;
         tile < tiles;
         ++tile) {

        const int stage = tile % STAGES;

        if (tile + STAGES - 1 < tiles) {

            load_stage_async<BM, BN, BK, THREADS>(
                M, N, K, A, B, As, Bs,
                (tile + STAGES - 1) % STAGES,
                (tile + STAGES - 1) * BK,
                block_row, block_col, tid);

            __pipeline_commit();
        }

        // ----------------------------
        // 计算 stage 缓冲（tile t 的数据在前一轮 wait 时已落地）
        // ----------------------------

        const int thread_row = ty * TM;
        const int thread_col = tx * TN;

#pragma unroll
        for (int k = 0; k < BK; ++k) {

            float reg_a[TM];
            float reg_b[TN];

#pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] =
                    As[stage][thread_row + i][k];
            }

            // B 沿 N 方向连续，按 float4 从 smem 读
#pragma unroll
            for (int j = 0; j < TN; j += 4) {

                const float4 b4 =
                    *reinterpret_cast<const float4*>(
                        &Bs[stage][k][thread_col + j]);

                reg_b[j + 0] = b4.x;
                reg_b[j + 1] = b4.y;
                reg_b[j + 2] = b4.z;
                reg_b[j + 3] = b4.w;
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {

                    accum[i][j] +=
                        reg_a[i] *
                        reg_b[j];
                }
            }
        }

        // 等掉最老的一组（tile t+1），保证下一轮要读的缓冲已落地。
        // 尚未完成的组最多剩 min(STAGES-1, tiles-1-t) 个，
        // wait_prior(剩余-1) 恰好完成最老的那组；结尾几轮没有新组发出时自动收敛。
        const int outstanding =
            (STAGES - 1 < tiles - 1 - tile)
                ? STAGES - 1
                : tiles - 1 - tile;

        if (outstanding > 0) {

            __pipeline_wait_prior(outstanding - 1);
        }

        __syncthreads();
    }

    // ----------------------------
    // 写回 C
    // ----------------------------

    const int row_base =
        block_row + ty * TM;

    const int col_base =
        block_col + tx * TN;

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {

            const int row = row_base + i;
            const int col = col_base + j;

            if (row < M && col < N) {

                C[row * N + col] =
                    alpha * accum[i][j] +
                    beta * C[row * N + col];
            }
        }
    }
}

// 公共 launcher 模板：block = (BN/TN, BM/TM)
template <
    int BM,
    int BN,
    int BK,
    int TM,
    int TN,
    int STAGES>
void launch_gemm_config(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C) {

    dim3 block(BN / TN, BM / TM);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    gemm_simt_generic<BM, BN, BK, TM, TN, STAGES>
        <<<grid, block>>>(
            M, N, K,
            alpha, A, B,
            beta, C);
}

// ------------------------------------------------------------
// 组 1：tile 尺寸（其余固定：BK=8、4x4 分块、2-stage）
// ------------------------------------------------------------

void launch_gemm_tile_32x32(int M, int N, int K, float alpha, const float* A, const float* B,
                            float beta, float* C) {
    launch_gemm_config<32, 32, 8, 4, 4, 2>(M, N, K, alpha, A, B, beta, C);
}

void launch_gemm_tile_64x64(int M, int N, int K, float alpha, const float* A, const float* B,
                            float beta, float* C) {
    launch_gemm_config<64, 64, 8, 4, 4, 2>(M, N, K, alpha, A, B, beta, C);
}

void launch_gemm_tile_128x128(int M, int N, int K, float alpha, const float* A, const float* B,
                              float beta, float* C) {
    launch_gemm_config<128, 128, 8, 4, 4, 2>(M, N, K, alpha, A, B, beta, C);
}

// ------------------------------------------------------------
// 组 2：寄存器分块（固定 128x128x8、2-stage；
//       4x4 复用 tile_128x128，8x8 复用 gemm_simt_128x128，这里补中间档 4x8）
// ------------------------------------------------------------

void launch_gemm_rblock_4x8(int M, int N, int K, float alpha, const float* A, const float* B,
                            float beta, float* C) {
    launch_gemm_config<128, 128, 8, 4, 8, 2>(M, N, K, alpha, A, B, beta, C);
}

// ------------------------------------------------------------
// 组 3：流水深度（固定 128x128x8、8x8；
//       stages2 同时作为对 gemm_simt_128x128 的实现一致性校验）
// ------------------------------------------------------------

void launch_gemm_stages2_128x128(int M, int N, int K, float alpha, const float* A,
                                 const float* B, float beta, float* C) {
    launch_gemm_config<128, 128, 8, 8, 8, 2>(M, N, K, alpha, A, B, beta, C);
}

void launch_gemm_stages4_128x128(int M, int N, int K, float alpha, const float* A,
                                 const float* B, float beta, float* C) {
    launch_gemm_config<128, 128, 8, 8, 8, 4>(M, N, K, alpha, A, B, beta, C);
}
