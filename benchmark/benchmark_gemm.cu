//executable均从项目根目录执行
//
// 用法：
//   benchmark_gemm                 # 跑全部 kernel，结果写 results/benchmark.csv
//   benchmark_gemm <kernel_name>   # 只跑指定 kernel，结果写 results/benchmark_<name>.csv
//                                  # （单 kernel 模式不覆盖全量 CSV，方便 ncu 过滤前单独计时）
//   benchmark_gemm -h              # 列出可用 kernel


#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
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

// fp32 输入版本（naive / shared_tiling / vectorized / async_copy / double_buffer）
using GemmLauncher = void (*)(int M, int N, int K, float alpha, const float* A, const float* B,
                              float beta, float* C);

// fp16 输入版本（tensor_core / tensor_core_optimized）
using HalfGemmLauncher = void (*)(int M, int N, int K, float alpha, const half* A, const half* B,
                                  float beta, float* C);

// ============================================================
// Kernel registration
// ============================================================

struct GemmKernel {
    std::string name;
    GemmLauncher launcher;           // fp32 输入内核
    HalfGemmLauncher half_launcher;  // fp16 输入内核；二者取其一，另一个置 nullptr
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

template <typename DataT, typename Launcher>
BenchmarkResult benchmark_one(const std::string& name, Launcher launcher,
                              const GemmProblem& problem, int warmup_iterations,
                              int benchmark_iterations) {
    const int M = problem.M;
    const int N = problem.N;
    const int K = problem.K;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    std::vector<DataT> A(static_cast<std::size_t>(M) * K);

    std::vector<DataT> B(static_cast<std::size_t>(K) * N);

    for (DataT& value : A) {
        const float f = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

        if constexpr (std::is_same_v<DataT, half>) {
            value = __float2half(f);
        } else {
            value = f;
        }
    }

    for (DataT& value : B) {
        const float f = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

        if constexpr (std::is_same_v<DataT, half>) {
            value = __float2half(f);
        } else {
            value = f;
        }
    }

    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    DataT* d_A = nullptr;
    DataT* d_B = nullptr;
    float* d_C = nullptr;

    const std::size_t bytes_A = static_cast<std::size_t>(M) * K * sizeof(DataT);

    const std::size_t bytes_B = static_cast<std::size_t>(K) * N * sizeof(DataT);

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
        launcher(M, N, K, alpha, d_A, d_B, beta, d_C);
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
        launcher(M, N, K, alpha, d_A, d_B, beta, d_C);
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

    return {name, M, N, K, average_ms, tflops};
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    const int warmup_iterations = 10;

    const int benchmark_iterations = 10;

    // --------------------------------------------------------
    // Problems
    // --------------------------------------------------------

    // 注意：所有尺寸均为 16 的倍数，满足 tensor core 系列内核的边界要求
    const std::vector<GemmProblem> problems = {
        {256, 256, 256},    {512, 512, 512},    {1024, 1024, 1024},
        {2048, 2048, 2048}, {4096, 4096, 4096},
    };

    // --------------------------------------------------------
    // Kernels
    // --------------------------------------------------------

    const std::vector<GemmKernel> kernels = {
        {"gemm_naive", launch_gemm_naive, nullptr},
        {"gemm_shared_tiling", launch_gemm_shared_tiling, nullptr},
        {"gemm_vectorized", launch_gemm_vectorized, nullptr},
        {"gemm_async_copy", launch_gemm_async_copy, nullptr},
        {"gemm_double_buffer", launch_gemm_double_buffer, nullptr},
        {"gemm_async_vectorized", launch_gemm_async_vectorized, nullptr},
        {"gemm_simt_128x128", launch_gemm_simt_128x128, nullptr},
        {"gemm_tensor_core", nullptr, launch_gemm_tensor_core},
        {"gemm_tensor_core_optimized", nullptr, launch_gemm_tensor_core_optimized},
        {"gemm_tensor_core_mma", nullptr, launch_gemm_tensor_core_mma},
        // ---- 对照实验组（单变量控制，见 kernels/gemm_controlled.cu）----
        {"gemm_tile_32x32", launch_gemm_tile_32x32, nullptr},
        {"gemm_tile_64x64", launch_gemm_tile_64x64, nullptr},
        {"gemm_tile_128x128", launch_gemm_tile_128x128, nullptr},
        {"gemm_rblock_4x8", launch_gemm_rblock_4x8, nullptr},
        {"gemm_stages2_128x128", launch_gemm_stages2_128x128, nullptr},
        {"gemm_stages4_128x128", launch_gemm_stages4_128x128, nullptr},
    };

    // --------------------------------------------------------
    // Command line
    // --------------------------------------------------------

    std::string only_kernel;

    if (argc > 1) {
        only_kernel = argv[1];

        if (only_kernel == "-h" || only_kernel == "--help") {
            std::cout << "usage: benchmark_gemm [kernel_name]\n\n";
            std::cout << "available kernels:\n";

            for (const GemmKernel& kernel : kernels) {
                std::cout << "  " << kernel.name << '\n';
            }

            return EXIT_SUCCESS;
        }
    }

    // --------------------------------------------------------
    // Results directory
    // --------------------------------------------------------

    // 单 kernel 模式写单独的文件，不覆盖全量对比数据
    const std::string csv_path =
        only_kernel.empty() ? "results/benchmark.csv"
                            : "results/benchmark_" + only_kernel + ".csv";

    std::filesystem::create_directories("results");

    std::ofstream csv(csv_path);

    if (!csv.is_open()) {
        std::cerr << "Failed to open " << csv_path << "\n";

        return EXIT_FAILURE;
    }

    csv << "kernel," << "M," << "N," << "K," << "latency_ms," << "tflops\n";

    // --------------------------------------------------------
    // Run benchmarks
    // --------------------------------------------------------

    bool matched = false;

    for (const GemmKernel& kernel : kernels) {
        if (!only_kernel.empty() && kernel.name != only_kernel) {
            continue;
        }

        matched = true;

        std::cout << "\n========================================\n";

        std::cout << "Kernel: " << kernel.name << '\n';

        std::cout << "========================================\n";

        for (const GemmProblem& problem : problems) {
            const BenchmarkResult result =
                kernel.half_launcher != nullptr
                    ? benchmark_one<half>(kernel.name, kernel.half_launcher, problem,
                                          warmup_iterations, benchmark_iterations)
                    : benchmark_one<float>(kernel.name, kernel.launcher, problem,
                                           warmup_iterations, benchmark_iterations);

            std::cout << "M=" << result.M << " N=" << result.N << " K=" << result.K
                      << " | latency=" << result.average_ms << " ms"
                      << " | performance=" << result.tflops << " TFLOPS\n";

            csv << result.kernel_name << ',' << result.M << ',' << result.N << ',' << result.K
                << ',' << result.average_ms << ',' << result.tflops << '\n';
        }
    }

    if (!only_kernel.empty() && !matched) {
        std::cerr << "unknown kernel: " << only_kernel << "\n\n";
        std::cerr << "available kernels:\n";

        for (const GemmKernel& kernel : kernels) {
            std::cerr << "  " << kernel.name << '\n';
        }

        return EXIT_FAILURE;
    }

    std::cout << "\nBenchmark results saved to " << csv_path << "\n";

    return 0;
}
