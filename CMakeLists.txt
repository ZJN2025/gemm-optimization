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
