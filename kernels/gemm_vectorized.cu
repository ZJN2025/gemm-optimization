#include <cuda_runtime.h>

#include "gemm.hpp"

template <
    int BM,
    int BN,
    int BK,
    int TM,
    int TN>
__global__ void gemm_vectorized(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C) {

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid =
        ty * blockDim.x + tx;

    constexpr int THREADS =
        (BM / TM) * (BN / TN);

    const int block_row =
        blockIdx.y * BM;

    const int block_col =
        blockIdx.x * BN;

    float accum[TM][TN] = {};

    for (int tile_k = 0;
         tile_k < K;
         tile_k += BK) {

        // ----------------------------
        // Vectorized A load
        // ----------------------------

        constexpr int A_VEC_PER_ROW =
            BK / 4;

        constexpr int A_VECS =
            BM * A_VEC_PER_ROW;

        for (int vec = tid;
             vec < A_VECS;
             vec += THREADS) {

            const int r =
                vec / A_VEC_PER_ROW;

            const int c4 =
                (vec % A_VEC_PER_ROW) * 4;

            const int gr =
                block_row + r;

            const int gc =
                tile_k + c4;

            if (gr < M && gc + 3 < K) {

                float4 value =
                    *reinterpret_cast<
                        const float4*>(
                        &A[gr * K + gc]);

                As[r][c4 + 0] = value.x;
                As[r][c4 + 1] = value.y;
                As[r][c4 + 2] = value.z;
                As[r][c4 + 3] = value.w;

            } else {

                for (int i = 0; i < 4; ++i) {

                    As[r][c4 + i] =
                        (gr < M &&
                         gc + i < K)
                            ? A[gr * K +
                                gc + i]
                            : 0.0f;
                }
            }
        }

        // ----------------------------
        // Vectorized B load
        // ----------------------------

        constexpr int B_VEC_PER_ROW =
            BN / 4;

        constexpr int B_VECS =
            BK * B_VEC_PER_ROW;

        for (int vec = tid;
             vec < B_VECS;
             vec += THREADS) {

            const int r =
                vec / B_VEC_PER_ROW;

            const int c4 =
                (vec % B_VEC_PER_ROW) * 4;

            const int gr =
                tile_k + r;

            const int gc =
                block_col + c4;

            if (gr < K && gc + 3 < N) {

                float4 value =
                    *reinterpret_cast<
                        const float4*>(
                        &B[gr * N + gc]);

                Bs[r][c4 + 0] = value.x;
                Bs[r][c4 + 1] = value.y;
                Bs[r][c4 + 2] = value.z;
                Bs[r][c4 + 3] = value.w;

            } else {

                for (int i = 0; i < 4; ++i) {

                    Bs[r][c4 + i] =
                        (gr < K &&
                         gc + i < N)
                            ? B[gr * N +
                                gc + i]
                            : 0.0f;
                }
            }
        }

        __syncthreads();

        const int thread_row =
            ty * TM;

        const int thread_col =
            tx * TN;

        for (int k = 0; k < BK; ++k) {

            float reg_a[TM];
            float reg_b[TN];

            for (int i = 0; i < TM; ++i) {
                reg_a[i] =
                    As[thread_row + i][k];
            }

            for (int j = 0; j < TN; ++j) {
                reg_b[j] =
                    Bs[k][thread_col + j];
            }

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] +=
                        reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
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
void launch_gemm_vectorized(
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
    constexpr int BK = 8;

    constexpr int TM = 4;
    constexpr int TN = 4;

    dim3 block(
        BN / TN,
        BM / TM);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    gemm_vectorized<
        BM, BN, BK, TM, TN>
        <<<grid, block>>>(
            M, N, K,
            alpha,
            A, B,
            beta,
            C);
}