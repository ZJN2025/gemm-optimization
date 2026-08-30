#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "cuda_check.hpp"
#include "gemm.hpp"

// ============================================================
// Common launcher interface
// ============================================================

using GemmLauncher = void (*)(int M, int N, int K, float alpha, const float* A, const float* B,
                              float beta, float* C);

using HalfGemmLauncher = void (*)(int M, int N, int K, float alpha, const half* A, const half* B,
                                  float beta, float* C);

// ============================================================
// GEMM size
// ============================================================

struct GemmSize {
    int M;
    int N;
    int K;
};

// ============================================================
// CPU reference
// ============================================================

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

// ============================================================
// Helpers
// ============================================================

inline float random_float() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

// fp16 内核的输入在拷贝前用 __float2half 转换
template <typename DataT>
inline DataT to_input_type(float value) {
    if constexpr (std::is_same_v<DataT, half>) {
        return __float2half(value);
    } else {
        return value;
    }
}

// ============================================================
// Run one case: one kernel + one size + one (alpha, beta)
// ============================================================

template <typename DataT, typename Launcher>
bool run_one_case(const std::string& name, Launcher launcher, float tolerance, const GemmSize& size,
                  float alpha, float beta) {
    const int M = size.M;
    const int N = size.N;
    const int K = size.K;

    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    std::vector<float> A_host(M * K);
    std::vector<float> B_host(K * N);

    // C 的初值：beta != 0 时参与计算，随机初始化以覆盖 beta 路径
    std::vector<float> C_init(M * N, 0.0f);

    for (float& value : A_host) {
        value = random_float();
    }

    for (float& value : B_host) {
        value = random_float();
    }

    for (float& value : C_init) {
        value = random_float();
    }

    // CPU 参考：从与 GPU 相同的 C 初值出发
    std::vector<float> C_ref = C_init;
    cpu_gemm(M, N, K, alpha, A_host.data(), B_host.data(), beta, C_ref.data());

    // 设备端输入：fp32 内核直接用 float，fp16 内核先转成 half
    std::vector<DataT> A_input(M * K);
    std::vector<DataT> B_input(K * N);

    for (int i = 0; i < M * K; ++i) {
        A_input[i] = to_input_type<DataT>(A_host[i]);
    }

    for (int i = 0; i < K * N; ++i) {
        B_input[i] = to_input_type<DataT>(B_host[i]);
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

    cuda_check(cudaMemcpy(d_A, A_input.data(), bytes_A, cudaMemcpyHostToDevice),
               "copy A host to device");
    cuda_check(cudaMemcpy(d_B, B_input.data(), bytes_B, cudaMemcpyHostToDevice),
               "copy B host to device");

    // 与参考实现相同的 C 初值（而不是 memset 清零）
    cuda_check(cudaMemcpy(d_C, C_init.data(), bytes_C, cudaMemcpyHostToDevice),
               "copy C init host to device");

    // --------------------------------------------------------
    // Launch
    // --------------------------------------------------------

    launcher(M, N, K, alpha, d_A, d_B, beta, d_C);

    cuda_check(cudaGetLastError(), "launch " + name);
    cuda_check(cudaDeviceSynchronize(), "synchronize " + name);

    // --------------------------------------------------------
    // Compare
    // --------------------------------------------------------

    std::vector<float> C_gpu(M * N);

    cuda_check(cudaMemcpy(C_gpu.data(), d_C, bytes_C, cudaMemcpyDeviceToHost),
               "copy C device to host");

    cuda_check(cudaFree(d_A), "cudaFree d_A");
    cuda_check(cudaFree(d_B), "cudaFree d_B");
    cuda_check(cudaFree(d_C), "cudaFree d_C");

    float max_abs_error = 0.0f;
    float max_ref = 0.0f;

    for (int i = 0; i < M * N; ++i) {
        max_abs_error = std::max(max_abs_error, std::abs(C_ref[i] - C_gpu[i]));
        max_ref = std::max(max_ref, std::abs(C_ref[i]));
    }

    // 相对误差：与参考值量级挂钩
    const float limit = tolerance * std::max(1.0f, max_ref);

    const bool pass = max_abs_error <= limit;

    std::cout << "  M=" << M << " N=" << N << " K=" << K << " alpha=" << alpha
              << " beta=" << beta << " | max_abs_error=" << max_abs_error << " (limit=" << limit
              << ") " << (pass ? "PASS" : "FAIL") << '\n';

    return pass;
}

// ============================================================
// Run all cases for one kernel
// ============================================================

template <typename DataT, typename Launcher>
bool run_kernel_tests(const std::string& name, Launcher launcher, float tolerance,
                      const std::vector<GemmSize>& sizes) {
    std::cout << "\n----------------------------------------\n";
    std::cout << "Kernel: " << name << '\n';
    std::cout << "----------------------------------------\n";

    bool all_pass = true;

    for (const GemmSize& size : sizes) {
        // beta = 0：纯 A*B；beta != 0：覆盖 C = alpha*A*B + beta*C 路径
        all_pass &= run_one_case<DataT>(name, launcher, tolerance, size, 1.0f, 0.0f);
        all_pass &= run_one_case<DataT>(name, launcher, tolerance, size, 0.5f, 0.5f);
    }

    std::cout << "Kernel " << name << ": " << (all_pass ? "PASS" : "FAIL") << '\n';

    return all_pass;
}

// ============================================================
// Main
// ============================================================

int main() {
    // 容差说明：
    // - fp32 内核：参考值与 GPU 都按 fp32 累积，相对误差应远小于 1e-3
    // - fp16 内核：输入舍入到 fp16 本身引入约 5e-4 的相对误差，容差放宽到 1e-2
    const float fp32_tolerance = 1e-3f;
    const float fp16_tolerance = 1e-2f;

    // 尺寸选择说明：
    // - 100 不整除任何 tile 尺寸，用来压边界
    // - gemm_vectorized 用 float4 访存，要求 K、N 为 4 的倍数（16 字节对齐）
    // - tensor core 系列要求 M、N、K 为 16 的倍数（optimized 版还要求 M%32、N%64 为 0）
    const std::vector<GemmSize> fp32_sizes = {
        {64, 64, 64},
        {100, 100, 100},
        {256, 256, 256},
    };

    const std::vector<GemmSize> fp16_sizes = {
        {64, 64, 64},
        {128, 256, 128},
        {256, 256, 256},
    };

    bool all_pass = true;

    all_pass &= run_kernel_tests<float>("gemm_naive", launch_gemm_naive, fp32_tolerance,
                                        fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_shared_tiling", launch_gemm_shared_tiling,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_vectorized", launch_gemm_vectorized,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_async_copy", launch_gemm_async_copy,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_double_buffer", launch_gemm_double_buffer,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_async_vectorized", launch_gemm_async_vectorized,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_simt_128x128", launch_gemm_simt_128x128,
                                        fp32_tolerance, fp32_sizes);

    // 对照实验组（同一模板的不同实例，全部过一遍正确性）
    all_pass &= run_kernel_tests<float>("gemm_tile_32x32", launch_gemm_tile_32x32,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_tile_64x64", launch_gemm_tile_64x64,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_tile_128x128", launch_gemm_tile_128x128,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_rblock_4x8", launch_gemm_rblock_4x8,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_stages2_128x128", launch_gemm_stages2_128x128,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<float>("gemm_stages4_128x128", launch_gemm_stages4_128x128,
                                        fp32_tolerance, fp32_sizes);

    all_pass &= run_kernel_tests<half>("gemm_tensor_core", launch_gemm_tensor_core,
                                       fp16_tolerance, fp16_sizes);

    all_pass &= run_kernel_tests<half>("gemm_tensor_core_optimized",
                                       launch_gemm_tensor_core_optimized, fp16_tolerance,
                                       fp16_sizes);

    all_pass &= run_kernel_tests<half>("gemm_tensor_core_mma", launch_gemm_tensor_core_mma,
                                       fp16_tolerance, fp16_sizes);

    std::cout << "\n========================================\n";
    std::cout << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << '\n';
    std::cout << "========================================\n";

    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
