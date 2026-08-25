//executable均从项目根目录执行


#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_check.hpp"
#include "gemm.hpp"

// ============================================================
// GEMM problem description
// ============================================================

struct GemmProblem {
    int M;
    int N;
    int K;
};

// ============================================================
// Common launcher interface
// ============================================================

using GemmLauncher = void (*)(int M, int N, int K, float alpha, const float* A, const float* B,
                              float beta, float* C);

// ============================================================
// Kernel registration
// ============================================================

struct GemmKernel {
    std::string name;
    GemmLauncher launcher;
};

// ============================================================
// Benchmark result
// ============================================================

struct BenchmarkResult {
    std::string kernel_name;

    int M;
    int N;
    int K;

    float average_ms;
    double tflops;
};

// ============================================================
// Benchmark one GEMM problem
// ============================================================

BenchmarkResult benchmark_one(const GemmKernel& kernel, const GemmProblem& problem,
                              int warmup_iterations, int benchmark_iterations) {
    const int M = problem.M;
    const int N = problem.N;
    const int K = problem.K;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    std::vector<float> A(static_cast<std::size_t>(M) * K);

    std::vector<float> B(static_cast<std::size_t>(K) * N);

    for (float& value : A) {
        value = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    for (float& value : B) {
        value = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for (int i = 0; i < warmup_iterations; ++i) {
        kernel.launcher(M, N, K, alpha, d_A, d_B, beta, d_C);
    }

    cuda_check(cudaGetLastError(), "warmup launch");

    cuda_check(cudaDeviceSynchronize(), "warmup synchronize");

    // --------------------------------------------------------
    // CUDA events
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    cuda_check(cudaEventCreate(&start), "create start event");

    cuda_check(cudaEventCreate(&stop), "create stop event");

    cuda_check(cudaEventRecord(start), "record start event");

    for (int i = 0; i < benchmark_iterations; ++i) {
        kernel.launcher(M, N, K, alpha, d_A, d_B, beta, d_C);
    }

    cuda_check(cudaGetLastError(), "benchmark launch");

    cuda_check(cudaEventRecord(stop), "record stop event");

    cuda_check(cudaEventSynchronize(stop), "synchronize stop event");

    float total_ms = 0.0f;

    cuda_check(cudaEventElapsedTime(&total_ms, start, stop), "calculate elapsed time");

    const float average_ms = total_ms / static_cast<float>(benchmark_iterations);

    // --------------------------------------------------------
    // TFLOPS
    // --------------------------------------------------------

    const double flops =
        2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    const double tflops = flops / static_cast<double>(average_ms) / 1e9;

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    cuda_check(cudaEventDestroy(start), "destroy start event");

    cuda_check(cudaEventDestroy(stop), "destroy stop event");

    cuda_check(cudaFree(d_A), "cudaFree d_A");

    cuda_check(cudaFree(d_B), "cudaFree d_B");

    cuda_check(cudaFree(d_C), "cudaFree d_C");

    return {kernel.name, M, N, K, average_ms, tflops};
}

// ============================================================
// Main
// ============================================================

int main() {
    const int warmup_iterations = 10;

    const int benchmark_iterations = 10;

    // --------------------------------------------------------
    // Problems
    // --------------------------------------------------------

    const std::vector<GemmProblem> problems = {
        {256, 256, 256},    {512, 512, 512},    {1024, 1024, 1024},
        {2048, 2048, 2048}, {4096, 4096, 4096},
    };

    // --------------------------------------------------------
    // Kernels
    // --------------------------------------------------------

    const std::vector<GemmKernel> kernels = {
        {
            "gemm_naive",
            launch_gemm_naive,
        },
    };

    // --------------------------------------------------------
    // Results directory
    // --------------------------------------------------------

    std::filesystem::create_directories("results");

    std::ofstream csv("results/benchmark.csv");

    if (!csv.is_open()) {
        std::cerr << "Failed to open results/benchmark.csv\n";

        return EXIT_FAILURE;
    }

    csv << "kernel," << "M," << "N," << "K," << "latency_ms," << "tflops\n";

    // --------------------------------------------------------
    // Run all benchmarks
    // --------------------------------------------------------

    for (const GemmKernel& kernel : kernels) {
        std::cout << "\n========================================\n";

        std::cout << "Kernel: " << kernel.name << '\n';

        std::cout << "========================================\n";

        for (const GemmProblem& problem : problems) {
            const BenchmarkResult result =
                benchmark_one(kernel, problem, warmup_iterations, benchmark_iterations);

            std::cout << "M=" << result.M << " N=" << result.N << " K=" << result.K
                      << " | latency=" << result.average_ms << " ms"
                      << " | performance=" << result.tflops << " TFLOPS\n";

            csv << result.kernel_name << ',' << result.M << ',' << result.N << ',' << result.K
                << ',' << result.average_ms << ',' << result.tflops << '\n';
        }
    }

    std::cout << "\nBenchmark results saved to " << "results/benchmark.csv\n";

    return 0;
}