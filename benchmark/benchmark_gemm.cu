#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda_check.hpp"
#include "gemm.hpp"

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int warmup_iterations = 10;
    const int benchmark_iterations = 1;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);

    for (float& value : A) {
        value = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    for (float& value : B) {
        value = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    const std::size_t bytes_A = static_cast<std::size_t>(M) * K * sizeof(float);

    const std::size_t bytes_B = static_cast<std::size_t>(K) * N * sizeof(float);

    const std::size_t bytes_C = static_cast<std::size_t>(M) * N * sizeof(float);

    cuda_check(cudaMalloc(&d_A, bytes_A), "cudaMalloc d_A");

    cuda_check(cudaMalloc(&d_B, bytes_B), "cudaMalloc d_B");

    cuda_check(cudaMalloc(&d_C, bytes_C), "cudaMalloc d_C");

    cuda_check(cudaMemcpy(d_A, A.data(), bytes_A, cudaMemcpyHostToDevice), "copy A host to device");

    cuda_check(cudaMemcpy(d_B, B.data(), bytes_B, cudaMemcpyHostToDevice), "copy B host to device");

    cuda_check(cudaMemset(d_C, 0, bytes_C), "clear d_C");

    // ---------------------------
    // Warmup
    // ---------------------------

    for (int i = 0; i < warmup_iterations; ++i) {
        launch_gemm_naive(M, N, K, alpha, d_A, d_B, beta, d_C);
    }

    cuda_check(cudaGetLastError(), "warmup launch");

    cuda_check(cudaDeviceSynchronize(), "warmup synchronize");

    // ---------------------------
    // CUDA Events
    // ---------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    cuda_check(cudaEventCreate(&start), "create start event");

    cuda_check(cudaEventCreate(&stop), "create stop event");

    cuda_check(cudaEventRecord(start), "record start event");

    for (int i = 0; i < benchmark_iterations; ++i) {
        launch_gemm_naive(M, N, K, alpha, d_A, d_B, beta, d_C);
    }

    cuda_check(cudaGetLastError(), "benchmark launch");

    cuda_check(cudaEventRecord(stop), "record stop event");

    cuda_check(cudaEventSynchronize(stop), "synchronize stop event");

    float total_ms = 0.0f;

    cuda_check(cudaEventElapsedTime(&total_ms, start, stop), "calculate elapsed time");

    const float average_ms = total_ms / static_cast<float>(benchmark_iterations);

    // GEMM FLOPs ≈ 2 * M * N * K
    const double flops =
        2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    const double tflops = flops / static_cast<double>(average_ms) / 1e9;

    std::cout << "GEMM naive benchmark\n";
    std::cout << "--------------------\n";

    std::cout << "M = " << M << '\n';
    std::cout << "N = " << N << '\n';
    std::cout << "K = " << K << '\n';

    std::cout << "warmup iterations = " << warmup_iterations << '\n';

    std::cout << "benchmark iterations = " << benchmark_iterations << '\n';

    std::cout << "average latency = " << average_ms << " ms\n";

    std::cout << "performance = " << tflops << " TFLOPS\n";

    cuda_check(cudaEventDestroy(start), "destroy start event");

    cuda_check(cudaEventDestroy(stop), "destroy stop event");

    cuda_check(cudaFree(d_A), "cudaFree d_A");
    cuda_check(cudaFree(d_B), "cudaFree d_B");
    cuda_check(cudaFree(d_C), "cudaFree d_C");

    return 0;
}