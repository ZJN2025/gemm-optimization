#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

template <
    int BM,
    int BN,
    int BK>
__global__ void gemm_tensor_core_optimized(
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* C) {

    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    const int tid =
        threadIdx.x;

    const int warp_id =
        tid / 32;

    constexpr int WARPS_N =
        BN / 16;

    const int warp_m =
        warp_id / WARPS_N;

    const int warp_n =
        warp_id % WARPS_N;

    const int block_row =
        blockIdx.y * BM;

    const int block_col =
        blockIdx.x * BN;

    wmma::fragment<
        wmma::matrix_a,
        16, 16, 16,
        half,
        wmma::row_major>
        a_frag;

    wmma::fragment<
        wmma::matrix_b,
        16, 16, 16,
        half,
        wmma::row_major>
        b_frag;

    wmma::fragment<
        wmma::accumulator,
        16, 16, 16,
        float>
        acc_frag;

    wmma::fill_fragment(
        acc_frag,
        0.0f);

    for (int tile_k = 0;
         tile_k < K;
         tile_k += BK) {

        // ----------------------------
        // Cooperative A load
        // ----------------------------

        for (int idx = tid;
             idx < BM * BK;
             idx += blockDim.x) {

            const int r = idx / BK;
            const int c = idx % BK;

            const int gr =
                block_row + r;

            const int gc =
                tile_k + c;

            As[r][c] =
                (gr < M && gc < K)
                    ? A[gr * K + gc]
                    : __float2half(0.0f);
        }

        // ----------------------------
        // Cooperative B load
        // ----------------------------

        for (int idx = tid;
             idx < BK * BN;
             idx += blockDim.x) {

            const int r = idx / BN;
            const int c = idx % BN;

            const int gr =
                tile_k + r;

            const int gc =
                block_col + c;

            Bs[r][c] =
                (gr < K && gc < N)
                    ? B[gr * N + gc]
                    : __float2half(0.0f);
        }

        __syncthreads();

        const half* A_warp =
            &As[warp_m * 16][0];

        const half* B_warp =
            &Bs[0][warp_n * 16];

        wmma::load_matrix_sync(
            a_frag,
            A_warp,
            BK);

        wmma::load_matrix_sync(
            b_frag,
            B_warp,
            BN);

        wmma::mma_sync(
            acc_frag,
            a_frag,
            b_frag,
            acc_frag);

        __syncthreads();
    }

    for (int i = 0;
         i < acc_frag.num_elements;
         ++i) {

        acc_frag.x[i] *= alpha;
    }

    const int out_row =
        block_row +
        warp_m * 16;

    const int out_col =
        block_col +
        warp_n * 16;

    if (beta != 0.0f) {

        wmma::fragment<
            wmma::accumulator,
            16, 16, 16,
            float>
            c_frag;

        wmma::load_matrix_sync(
            c_frag,
            C +
                out_row * N +
                out_col,
            N,
            wmma::mem_row_major);

        for (int i = 0;
             i < acc_frag.num_elements;
             ++i) {

            acc_frag.x[i] +=
                beta * c_frag.x[i];
        }
    }

    wmma::store_matrix_sync(
        C +
            out_row * N +
            out_col,
        acc_frag,
        N,
        wmma::mem_row_major);
}
void launch_gemm_tensor_core_optimized(
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* C) {

    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 16;

    // 8 warps = 256 threads
    dim3 block(256);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    gemm_tensor_core_optimized<
        BM, BN, BK>
        <<<grid, block>>>(
            M, N, K,
            alpha,
            A, B,
            beta,
            C);
}