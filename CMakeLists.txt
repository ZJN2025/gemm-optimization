cmake_minimum_required(VERSION 3.24)

project(
    gemm_optimization
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(
    gemm_kernels
    kernels/gemm_naive.cu
    kernels/gemm_shared_tiling.cu
    kernels/gemm_vectorized.cu
    kernels/gemm_async_copy.cu
    kernels/gemm_double_buffer.cu
    kernels/gemm_async_vectorized.cu
    kernels/gemm_simt_128x128.cu
    kernels/gemm_tensor_core.cu
    kernels/gemm_tensor_core_optimized.cu
)
target_include_directories(
    gemm_kernels
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
add_executable(
    test_gemm
    tests/test_gemm.cu
)
target_link_libraries(
    test_gemm
    PRIVATE
    gemm_kernels
)
add_executable(
    benchmark_gemm
    benchmark/benchmark_gemm.cu
)

target_link_libraries(
    benchmark_gemm
    PRIVATE
    gemm_kernels
)

# ------------------------------------------------------------------
# CUTLASS（性能参考基线）
# configure 时从 GitHub 下载 v4.7.1，缓存在 build/_deps，之后重配置不再重复下载
# ------------------------------------------------------------------
include(FetchContent)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    cutlass
    URL
    https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.7.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP
    TRUE
)
FetchContent_MakeAvailable(cutlass)

add_executable(
    check_smem
    tools/check_smem.cu
)
add_executable(
    benchmark_cutlass
    benchmark/benchmark_cutlass.cu
)
target_include_directories(
    benchmark_cutlass
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
# CUTLASS 头文件里的 constexpr __host__ 函数需要放宽 constexpr 限制
target_compile_options(
    benchmark_cutlass
    PRIVATE
    --expt-relaxed-constexpr
)
target_link_libraries(
    benchmark_cutlass
    PRIVATE
    nvidia::cutlass::cutlass
)

# 测试不依赖工作目录，可在任意目录运行；benchmark 需从项目根目录执行（结果写入 results/）
enable_testing()
add_test(
    NAME
    gemm_tests
    COMMAND
    test_gemm
)
