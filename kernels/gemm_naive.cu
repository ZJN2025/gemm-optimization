#include <cuda_runtime.h>

#include "gemm.hpp"

// row-major
__global__ void gemm_naive(int M, int N, int K, float alpha, const float* A, const float* B,
                           float beta, float* C) {
    // 输出矩阵C的索引映射
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界条件判断
    if (row < M && col < N) {
        float sum = 0.0f;
        // 矩阵乘法循环
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        // 积写入矩阵C
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// launcher
void launch_gemm_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                       float* C) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}