#include <cuda_runtime.h>

#include "gemm.hpp"

template <
    int BM,
    int BN,
    int BK,
    int TM,
    int TN>
__global__ void gemm_double_buffer(
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

    const int a_row = tid / BK;
    const int a_col = tid % BK;

    const int b_row = tid / BN;
    const int b_col = tid % BN;

    float accum[TM][TN] = {};

    // --------------------------------
    // Load first tile
    // --------------------------------

    {
        const int gr =
            block_row + a_row;

        const int gc =
            a_col;

        As[0][a_row][a_col] =
            (gr < M && gc < K)
                ? A[gr * K + gc]
                : 0.0f;
    }

    {
        const int gr =
            b_row;

        const int gc =
            block_col + b_col;

        Bs[0][b_row][b_col] =
            (gr < K && gc < N)
                ? B[gr * N + gc]
                : 0.0f;
    }

    __syncthreads();

    const int tiles =
        (K + BK - 1) / BK;

    int current = 0;

    for (int tile = 0;
         tile < tiles;
         ++tile) {

        const int next =
            current ^ 1;

        float next_a = 0.0f;
        float next_b = 0.0f;

        const bool has_next =
            tile + 1 < tiles;

        // Issue next global loads early.
        if (has_next) {

            const int next_k =
                (tile + 1) * BK;

            const int a_gr =
                block_row + a_row;

            const int a_gc =
                next_k + a_col;

            if (a_gr < M && a_gc < K) {
                next_a =
                    A[a_gr * K + a_gc];
            }

            const int b_gr =
                next_k + b_row;

            const int b_gc =
                block_col + b_col;

            if (b_gr < K && b_gc < N) {
                next_b =
                    B[b_gr * N + b_gc];
            }
        }

        const int thread_row =
            ty * TM;

        const int thread_col =
            tx * TN;

        // Compute current tile while next
        // global loads can be outstanding.
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

        if (has_next) {

            As[next][a_row][a_col] =
                next_a;

            Bs[next][b_row][b_col] =
                next_b;
        }

        __syncthreads();

        current = next;
    }

    const int row_base =
        block_row + ty * TM;

    const int col_base =
        block_col + tx * TN;

    for (int i = 0; i < TM; ++i) {
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
void launch_gemm_double_buffer(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C) {

    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 8;

    constexpr int TM = 2;
    constexpr int TN = 2;

    dim3 block(
        BN / TN,
        BM / TM);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    gemm_double_buffer<
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