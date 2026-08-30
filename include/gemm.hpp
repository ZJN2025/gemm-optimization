#pragma once

#include <cuda_fp16.h>

// 所有 GEMM kernel 的 launcher 声明都集中在这个头文件里，
// 测试和 benchmark 只需包含本头文件即可调用任意版本的 GEMM。
//
// 约定：
// - A、B、C 均为行主序 (row-major)
// - 计算语义统一为 C = alpha * A * B + beta * C
// - M 为 A 的行数 / C 的行数，N 为 B 的列数 / C 的列数，K 为内积维

// ----------------------------
// fp32 输入（fp32 计算）
// ----------------------------

void launch_gemm_naive(int M, int N, int K, float alpha, const float* A, const float* B,
                       float beta, float* C);

void launch_gemm_shared_tiling(int M, int N, int K, float alpha, const float* A, const float* B,
                               float beta, float* C);

// 使用 float4 向量化访存，要求 K、N 为 4 的倍数（保证 16 字节对齐）
void launch_gemm_vectorized(int M, int N, int K, float alpha, const float* A, const float* B,
                            float beta, float* C);

// 用 cp.async (__pipeline_memcpy_async) 异步拷贝下一块 tile
void launch_gemm_async_copy(int M, int N, int K, float alpha, const float* A, const float* B,
                            float beta, float* C);

// 软件流水：先发起下一块 tile 的全局内存读取，再计算当前 tile
void launch_gemm_double_buffer(int M, int N, int K, float alpha, const float* A, const float* B,
                               float beta, float* C);

// cp.async + float4 向量化 + 双缓冲；要求 K、N 为 4 的倍数（16 字节对齐）
void launch_gemm_async_vectorized(int M, int N, int K, float alpha, const float* A,
                                  const float* B, float beta, float* C);

// 128x128 tile + 每线程 8x8 寄存器分块 + cp.async 双缓冲（精确 fp32，
// 与 CUTLASS 经典 simt sgemm 同款设计）；要求 K、N 为 4 的倍数（16 字节对齐）
void launch_gemm_simt_128x128(int M, int N, int K, float alpha, const float* A, const float* B,
                              float beta, float* C);

// ----------------------------
// 对照实验组（见 kernels/gemm_controlled.cu）：同一骨架只改一个变量，
// 骨架 = cp.async + float4 + N-stage 流水，K、N 需为 4 的倍数
// ----------------------------

// tile 尺寸组：BK=8、4x4 分块、2-stage 固定，只改 BM=BN
void launch_gemm_tile_32x32(int M, int N, int K, float alpha, const float* A, const float* B,
                            float beta, float* C);
void launch_gemm_tile_64x64(int M, int N, int K, float alpha, const float* A, const float* B,
                            float beta, float* C);
void launch_gemm_tile_128x128(int M, int N, int K, float alpha, const float* A, const float* B,
                              float beta, float* C);

// 寄存器分块组：128x128x8、2-stage 固定（4x4 = tile_128x128，8x8 = gemm_simt_128x128）
void launch_gemm_rblock_4x8(int M, int N, int K, float alpha, const float* A, const float* B,
                            float beta, float* C);

// 流水深度组：128x128x8、8x8 固定；stages2 同时校验通用模板与手写版实现一致
void launch_gemm_stages2_128x128(int M, int N, int K, float alpha, const float* A,
                                 const float* B, float beta, float* C);
void launch_gemm_stages4_128x128(int M, int N, int K, float alpha, const float* A,
                                 const float* B, float beta, float* C);

// ----------------------------
// fp16 输入（tensor core / wmma，fp32 累积）
// ----------------------------

// 注意：要求 M、N、K 均为 16 的倍数
void launch_gemm_tensor_core(int M, int N, int K, float alpha, const half* A, const half* B,
                             float beta, float* C);

// 注意：要求 M 为 32 的倍数、N 为 64 的倍数（warp 级 16x16 输出不做边界裁剪）
void launch_gemm_tensor_core_optimized(int M, int N, int K, float alpha, const half* A,
                                       const half* B, float beta, float* C);

// fp16 升级版：cp.async + ldmatrix + mma.sync m16n8k16 完整数据通路；
// 要求 K、N 为 8 的倍数（16B 对齐），M 任意
void launch_gemm_tensor_core_mma(int M, int N, int K, float alpha, const half* A, const half* B,
                                 float beta, float* C);
