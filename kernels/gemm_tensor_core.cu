#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void gemm_tensor_core(
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* C) {

    const int tile_row =
        blockIdx.y * 16;

    const int tile_col =
        blockIdx.x * 16;

    if (tile_row >= M ||
        tile_col >= N) {
        return;
    }

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

    for (int k = 0;
         k < K;
         k += 16) {

        const half* A_tile =
            A +
            tile_row * K +
            k;

        const half* B_tile =
            B +
            k * N +
            tile_col;

        wmma::load_matrix_sync(
            a_frag,
            A_tile,
            K);

        wmma::load_matrix_sync(
            b_frag,
            B_tile,
            N);

        wmma::mma_sync(
            acc_frag,
            a_frag,
            b_frag,
            acc_frag);
    }

    // alpha
    for (int i = 0;
         i < acc_frag.num_elements;
         ++i) {

        acc_frag.x[i] *= alpha;
    }

    if (beta != 0.0f) {

        wmma::fragment<
            wmma::accumulator,
            16, 16, 16,
            float>
            old_c;

        wmma::load_matrix_sync(
            old_c,
            C +
                tile_row * N +
                tile_col,
            N,
            wmma::mem_row_major);

        for (int i = 0;
             i < acc_frag.num_elements;
             ++i) {

            acc_frag.x[i] +=
                beta * old_c.x[i];
        }
    }

    wmma::store_matrix_sync(
        C +
            tile_row * N +
            tile_col,
        acc_frag,
        N,
        wmma::mem_row_major);
}
void launch_gemm_tensor_core(
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* C) {

    dim3 block(32);

    dim3 grid(
        (N + 15) / 16,
        (M + 15) / 16);

    gemm_tensor_core<<<grid, block>>>(
        M, N, K,
        alpha,
        A, B,
        beta,
        C);
}