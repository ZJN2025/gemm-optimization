#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include "gemm.hpp"

template <
    int BM,
    int BN,
    int BK,
    int TM,
    int TN>
__global__ void gemm_async_copy(
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
    // Async load first tile
    // --------------------------------

    {
        const int gr =
            block_row + a_row;

        const int gc =
            a_col;

        if (gr < M && gc < K) {

            __pipeline_memcpy_async(
                &As[0][a_row][a_col],
                &A[gr * K + gc],
                sizeof(float));

        } else {

            As[0][a_row][a_col] =
                0.0f;
        }
    }

    {
        const int gr = b_row;

        const int gc =
            block_col + b_col;

        if (gr < K && gc < N) {

            __pipeline_memcpy_async(
                &Bs[0][b_row][b_col],
                &B[gr * N + gc],
                sizeof(float));

        } else {

            Bs[0][b_row][b_col] =
                0.0f;
        }
    }

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

        if (has_next) {

            const int next_k =
                (tile + 1) * BK;

            const int a_gr =
                block_row + a_row;

            const int a_gc =
                next_k + a_col;

            if (a_gr < M && a_gc < K) {

                __pipeline_memcpy_async(
                    &As[next]
                       [a_row][a_col],
                    &A[a_gr * K + a_gc],
                    sizeof(float));

            } else {

                As[next][a_row][a_col] =
                    0.0f;
            }

            const int b_gr =
                next_k + b_row;

            const int b_gc =
                block_col + b_col;

            if (b_gr < K && b_gc < N) {

                __pipeline_memcpy_async(
                    &Bs[next]
                       [b_row][b_col],
                    &B[b_gr * N + b_gc],
                    sizeof(float));

            } else {

                Bs[next][b_row][b_col] =
                    0.0f;
            }

            __pipeline_commit();
        }

        // ----------------------------
        // Compute current tile
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

        if (has_next) {

            __pipeline_wait_prior(0);

            __syncthreads();
        }

        current = next;
    }

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
void launch_gemm_async_copy(
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

    gemm_async_copy<
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