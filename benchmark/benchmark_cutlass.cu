//executable均从项目根目录执行
//
// CUTLASS 参考基线：用 CUTLASS 官方实现跑同样的尺寸和计时方法，
// 结果写入 results/benchmark_cutlass.csv，供 tools/plot_benchmark.py 叠加对比。
//
// 用法：
//   benchmark_cutlass                 # 跑全部三种配置
//   benchmark_cutlass <config_name>   # 只跑指定配置，结果写 results/benchmark_cutlass_<name>.csv
//   benchmark_cutlass -h              # 列出可用配置
//
// 三个配置：
// - cutlass_sgemm_simt : fp32 SIMT（FFMA），与手写 fp32 kernel 同精度，可比性最强
// - cutlass_sgemm_tf32 : fp32 走 tf32 tensor core（精度略降、吞吐高），业界常用
// - cutlass_hgemm      : fp16 输入 / fp32 累积，与手写 tensor core kernel 直接对比


#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_check.hpp"

// CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

// ============================================================
// GEMM problem description
// ============================================================

struct GemmProblem {
    int M;
    int N;
    int K;
};

// ============================================================
// CUTLASS GEMM 配置（经典 Ampere 风格 kernel，RowMajor/RowMajor/RowMajor）
// ============================================================

// fp32 SIMT（精确 fp32，FFMA 计算）
using CutlassSgemmSimt = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,      // A
    float, cutlass::layout::RowMajor,      // B
    float, cutlass::layout::RowMajor,      // C
    float,                                 // accumulator
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>, // threadblock tile
    cutlass::gemm::GemmShape<32, 64, 8>,   // warp tile
    cutlass::gemm::GemmShape<1, 1, 1>,     // instruction (FFMA)
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;                                    // stages

// fp32 走 tf32 tensor core（m16n8k8）
// 注意 stages 用 2 而不是默认的 3：本机（RTX 5060 Laptop）opt-in 共享内存上限
// 只有 99KB，3-stage 需要 144KB 会启动失败；2-stage 为 96KB
using CutlassSgemmTf32 = cutlass::gemm::device::Gemm<
    cutlass::tfloat32_t, cutlass::layout::RowMajor,  // A
    cutlass::tfloat32_t, cutlass::layout::RowMajor,  // B
    float, cutlass::layout::RowMajor,                // C
    float,                                           // accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

// fp16 输入 / fp32 累积（m16n8k16），stages 同上限制
using CutlassHgemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,  // A
    cutlass::half_t, cutlass::layout::RowMajor,  // B
    float, cutlass::layout::RowMajor,            // C
    float,                                       // accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

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
// Helpers
// ============================================================

inline float random_float() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

// 把 [0,1) 的 float 随机数转成各 kernel 的输入类型
template <typename ElementT>
inline ElementT to_input_type(float value);

template <>
inline float to_input_type<float>(float value) {
    return value;
}

template <>
inline cutlass::half_t to_input_type<cutlass::half_t>(float value) {
    return cutlass::half_t::convert(value);
}

template <>
inline cutlass::tfloat32_t to_input_type<cutlass::tfloat32_t>(float value) {
    return cutlass::tfloat32_t(value);
}

// ============================================================
// Benchmark one GEMM problem with one CUTLASS configuration
// ============================================================

template <typename Gemm, typename ElementT>
BenchmarkResult benchmark_one(const std::string& name, const GemmProblem& problem,
                              int warmup_iterations, int benchmark_iterations) {
    const int M = problem.M;
    const int N = problem.N;
    const int K = problem.K;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // --------------------------------------------------------
    // Host memory（行主序，与手写 kernel 的布局一致）
    // --------------------------------------------------------

    std::vector<ElementT> A(static_cast<std::size_t>(M) * K);
    std::vector<ElementT> B(static_cast<std::size_t>(K) * N);

    for (ElementT& value : A) {
        value = to_input_type<ElementT>(random_float());
    }

    for (ElementT& value : B) {
        value = to_input_type<ElementT>(random_float());
    }

    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    ElementT* d_A = nullptr;
    ElementT* d_B = nullptr;
    float* d_C = nullptr;

    const std::size_t bytes_A = static_cast<std::size_t>(M) * K * sizeof(ElementT);
    const std::size_t bytes_B = static_cast<std::size_t>(K) * N * sizeof(ElementT);
    const std::size_t bytes_C = static_cast<std::size_t>(M) * N * sizeof(float);

    cuda_check(cudaMalloc(&d_A, bytes_A), "cudaMalloc d_A");
    cuda_check(cudaMalloc(&d_B, bytes_B), "cudaMalloc d_B");
    cuda_check(cudaMalloc(&d_C, bytes_C), "cudaMalloc d_C");

    cuda_check(cudaMemcpy(d_A, A.data(), bytes_A, cudaMemcpyHostToDevice), "copy A host to device");
    cuda_check(cudaMemcpy(d_B, B.data(), bytes_B, cudaMemcpyHostToDevice), "copy B host to device");
    cuda_check(cudaMemset(d_C, 0, bytes_C), "clear d_C");

    // --------------------------------------------------------
    // CUTLASS arguments（行主序下 lda=K, ldb=N, ldc=N）
    // --------------------------------------------------------

    Gemm gemm_op;

    typename Gemm::Arguments args(
        {M, N, K},          // problem size
        {d_A, K},           // A (M x K, row-major)
        {d_B, N},           // B (K x N, row-major)
        {d_C, N},           // C (M x N, row-major)
        {d_C, N},           // D (M x N, row-major)
        {alpha, beta});     // epilogue

    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for (int i = 0; i < warmup_iterations; ++i) {
        const cutlass::Status status = gemm_op(args);

        if (status != cutlass::Status::kSuccess) {
            std::cerr << name << " warmup failed (status " << static_cast<int>(status) << ": "
                      << cutlass::cutlassGetStatusString(status) << ")\n";
            const cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
            }
            std::exit(EXIT_FAILURE);
        }
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
        gemm_op(args);
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

    // 与 benchmark_gemm 相同的尺寸
    const std::vector<GemmProblem> problems = {
        {256, 256, 256},    {512, 512, 512},    {1024, 1024, 1024},
        {2048, 2048, 2048}, {4096, 4096, 4096},
    };

    // --------------------------------------------------------
    // Command line
    // --------------------------------------------------------

    const std::vector<std::string> config_names = {
        "cutlass_sgemm_simt", "cutlass_sgemm_tf32", "cutlass_hgemm",
    };

    std::string only_config;

    if (argc > 1) {
        only_config = argv[1];

        if (only_config == "-h" || only_config == "--help") {
            std::cout << "usage: benchmark_cutlass [config_name]\n\n";
            std::cout << "available configs:\n";

            for (const std::string& name : config_names) {
                std::cout << "  " << name << '\n';
            }

            return EXIT_SUCCESS;
        }

        const bool known = std::find(config_names.begin(), config_names.end(), only_config) !=
                           config_names.end();

        if (!known) {
            std::cerr << "unknown config: " << only_config << "\n\n";
            std::cerr << "available configs:\n";

            for (const std::string& name : config_names) {
                std::cerr << "  " << name << '\n';
            }

            return EXIT_FAILURE;
        }
    }

    const auto selected = [&](const char* name) {
        return only_config.empty() || only_config == name;
    };

    // --------------------------------------------------------
    // Results file
    // --------------------------------------------------------

    // 单配置模式写单独的文件，不覆盖全量对比数据
    const std::string csv_path =
        only_config.empty() ? "results/benchmark_cutlass.csv"
                            : "results/benchmark_cutlass_" + only_config + ".csv";

    std::filesystem::create_directories("results");

    std::ofstream csv(csv_path);

    if (!csv.is_open()) {
        std::cerr << "Failed to open " << csv_path << "\n";
        return EXIT_FAILURE;
    }

    csv << "kernel," << "M," << "N," << "K," << "latency_ms," << "tflops\n";

    const auto run_and_report = [&](const BenchmarkResult& result) {
        std::cout << result.kernel_name << " | M=" << result.M << " N=" << result.N
                  << " K=" << result.K << " | latency=" << result.average_ms << " ms"
                  << " | performance=" << result.tflops << " TFLOPS\n";

        csv << result.kernel_name << ',' << result.M << ',' << result.N << ',' << result.K
            << ',' << result.average_ms << ',' << result.tflops << '\n';
    };

    for (const GemmProblem& problem : problems) {
        if (selected("cutlass_sgemm_simt")) {
            run_and_report(benchmark_one<CutlassSgemmSimt, float>("cutlass_sgemm_simt", problem,
                                                                  warmup_iterations,
                                                                  benchmark_iterations));
        }

        if (selected("cutlass_sgemm_tf32")) {
            run_and_report(benchmark_one<CutlassSgemmTf32, cutlass::tfloat32_t>(
                "cutlass_sgemm_tf32", problem, warmup_iterations, benchmark_iterations));
        }

        if (selected("cutlass_hgemm")) {
            run_and_report(benchmark_one<CutlassHgemm, cutlass::half_t>(
                "cutlass_hgemm", problem, warmup_iterations, benchmark_iterations));
        }
    }

    std::cout << "\nCUTLASS benchmark results saved to " << csv_path << "\n";

    return 0;
}
