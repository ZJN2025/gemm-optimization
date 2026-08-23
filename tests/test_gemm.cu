#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda_check.hpp"
#include "gemm.hpp"

void cpu_gemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
              float* C) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
    }
}

int main() {
    const int M = 64;
    const int N = 64;
    const int K = 64;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);

    std::vector<float> C_cpu(M * N, 0.0f);
    std::vector<float> C_gpu(M * N, 0.0f);

    // 初始化 A
    for (float& value : A) {
        value = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    // 初始化 B
    for (float& value : B) {
        value = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    // CPU reference
    cpu_gemm(M, N, K, alpha, A.data(), B.data(), beta, C_cpu.data());

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    const std::size_t bytes_A = static_cast<std::size_t>(M) * K * sizeof(float);

    const std::size_t bytes_B = static_cast<std::size_t>(K) * N * sizeof(float);

    const std::size_t bytes_C = static_cast<std::size_t>(M) * N * sizeof(float);

    // GPU memory
    cuda_check(cudaMalloc(&d_A, bytes_A), "cudaMalloc d_A");

    cuda_check(cudaMalloc(&d_B, bytes_B), "cudaMalloc d_B");

    cuda_check(cudaMalloc(&d_C, bytes_C), "cudaMalloc d_C");

    // Host -> Device
    cuda_check(cudaMemcpy(d_A, A.data(), bytes_A, cudaMemcpyHostToDevice), "copy A host to device");

    cuda_check(cudaMemcpy(d_B, B.data(), bytes_B, cudaMemcpyHostToDevice), "copy B host to device");

    // beta = 0，所以第一版直接清零即可
    cuda_check(cudaMemset(d_C, 0, bytes_C), "clear d_C");

    // Launch GEMM
    launch_gemm_naive(M, N, K, alpha, d_A, d_B, beta, d_C);

    // 检查 launch 本身
    cuda_check(cudaGetLastError(), "launch gemm_naive");

    // 等待 GPU 真正执行完成
    cuda_check(cudaDeviceSynchronize(), "execute gemm_naive");

    // Device -> Host
    cuda_check(cudaMemcpy(C_gpu.data(), d_C, bytes_C, cudaMemcpyDeviceToHost),
               "copy C device to host");

    float max_abs_error = 0.0f;

    for (int i = 0; i < M * N; ++i) {
        const float error = std::abs(C_cpu[i] - C_gpu[i]);

        max_abs_error = std::max(max_abs_error, error);
    }

    std::cout << "M = " << M << '\n';
    std::cout << "N = " << N << '\n';
    std::cout << "K = " << K << '\n';

    std::cout << "max absolute error = " << max_abs_error << '\n';

    if (max_abs_error < 1e-3f) {
        std::cout << "PASS\n";
    } else {
        std::cout << "FAIL\n";
    }

    cuda_check(cudaFree(d_A), "cudaFree d_A");
    cuda_check(cudaFree(d_B), "cudaFree d_B");
    cuda_check(cudaFree(d_C), "cudaFree d_C");

    return 0;
}