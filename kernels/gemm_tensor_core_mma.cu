#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "gemm.hpp"

// ============================================================
// fp16 tensor core 升级版：完整的数据通路
//   cp.async(16B) -> smem（为 ldmatrix 定制的 8x8 块布局）-> ldmatrix -> mma.sync m16n8k16
// 相比 wmma 版（gemm_tensor_core / gemm_tensor_core_optimized）的区别：
//   - 取数向量化：ldmatrix 一条指令把 8x8 tile 搬进 fragment，替代 wmma 的慢速 load
//   - smem 布局为 ldmatrix 定做：8x8 块连续 128B，x4 变体一次搬 4 个块
//   - 直接写 mma.sync PTX（wmma 只是它的封装）
// 配置：64x64x16 tile、4 warps、2-stage 流水。
// 要求 K、N 为 8 的倍数（16B 对齐）；M 任意（越界零填充 + 写回判界）。
// ============================================================

// ldmatrix.m8n8.x4：一次搬 4 个 8x8 fp16 块（块间距 128B）。
// 注意语义：32 个 lane 各提供一个地址——lane i 提供第 (i/8) 块矩阵的第 (i%8) 行地址，
// 而不是"8 个线程给地址、硬件自动偏移"。
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                            uint32_t& r3, uint32_t smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_addr));
}

// .trans 变体：地址语义与普通版相同（lane i 提供第 (i/8) 块的第 (i%8) 行地址，16B 对齐），
// 区别只在结果寄存器布局——输出是转置后的 fragment（正好匹配 mma 的 B 操作数布局）
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                  uint32_t& r3, uint32_t smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_addr));
}

// mma.sync m16n8k16：A(m16k16, row) x B(k16n8, col) -> C(m16n8, f32 累加)
// 每线程：A 4 个 uint32（8 个 half）、B 2 个 uint32（4 个 half）、C/D 4 个 float
__device__ __forceinline__ void mma_16x8x16(float& c0, float& c1, float& c2, float& c3,
                                           uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 把第 tile_k 个 tile 以 16B（8 个 half）为单位异步拷进 stage 缓冲的 8x8 块布局：
//   As[stage][k_blk][m_blk][8][8]：每块 128B，m 方向连续 —— 正好是 ldmatrix.x4 的间距
//   Bs[stage][k_blk][n_blk][8][8]：n 方向连续 —— 正好是 ldmatrix.x4.trans 的间距
// 越界退化逐元素（合法取值、非法填 0）。
template <int BM, int BN, int BK, int THREADS>
__device__ __forceinline__ void load_stage_async(
    int M,
    int N,
    int K,
    const half* A,
    const half* B,
    half (*As)[BK / 8][BM / 8][8][8],
    half (*Bs)[BK / 8][BN / 8][8][8],
    int stage,
    int tile_k,
    int block_row,
    int block_col,
    int tid) {

    // ----------------------------
    // A tile：16B 块 = 同一行 8 个连续的 k
    // ----------------------------

    constexpr int A_CHUNKS = BM * (BK / 8);

    for (int v = tid; v < A_CHUNKS; v += THREADS) {

        const int a_m = v / (BK / 8);
        const int a_kb = v % (BK / 8);

        const int a_gr = block_row + a_m;
        const int a_gc = tile_k + a_kb * 8;

        half* dst = &As[stage][a_kb][a_m / 8][a_m % 8][0];
        const half* src = &A[a_gr * K + a_gc];

        if (a_gr < M && a_gc + 7 < K) {

            __pipeline_memcpy_async(dst, src, 16);

        } else {

#pragma unroll
            for (int i = 0; i < 8; ++i) {

                dst[i] =
                    (a_gr < M && a_gc + i < K)
                        ? src[i]
                        : __float2half(0.0f);
            }
        }
    }

    // ----------------------------
    // B tile：16B 块 = 同一行 8 个连续的 n
    // ----------------------------

    constexpr int B_CHUNKS = (BK / 8) * 8 * (BN / 8);

    for (int v = tid; v < B_CHUNKS; v += THREADS) {

        const int b_kb = v / (8 * (BN / 8));
        const int b_row = (v / (BN / 8)) % 8;
        const int b_nb = v % (BN / 8);

        const int b_gr = tile_k + b_kb * 8 + b_row;
        const int b_gc = block_col + b_nb * 8;

        half* dst = &Bs[stage][b_kb][b_nb][b_row][0];
        const half* src = &B[b_gr * N + b_gc];

        if (b_gr < K && b_gc + 7 < N) {

            __pipeline_memcpy_async(dst, src, 16);

        } else {

#pragma unroll
            for (int i = 0; i < 8; ++i) {

                dst[i] =
                    (b_gr < K && b_gc + i < N)
                        ? src[i]
                        : __float2half(0.0f);
            }
        }
    }
}

template <
    int BM,
    int BN,
    int BK>
__global__ void gemm_tensor_core_mma_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* C) {

    constexpr int THREADS = 128;

    __shared__ half As[2][BK / 8][BM / 8][8][8];
    __shared__ half Bs[2][BK / 8][BN / 8][8][8];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    // 4 warps 排成 2x2：warp tile 32x32
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // 每个 warp：2 个 m16 x 4 个 n8 = 8 个 mma，每个 4 个 float 累加器
    float accum[2][4][4] = {};

    const int tiles = (K + BK - 1) / BK;

    // --------------------------------
    // Prologue：预装 tile 0
    // --------------------------------

    const int prologue = (1 < tiles) ? 1 : tiles;

    for (int s = 0; s < prologue; ++s) {

        load_stage_async<BM, BN, BK, THREADS>(
            M, N, K, A, B, As, Bs,
            s, s * BK,
            block_row, block_col, tid);

        __pipeline_commit();
    }

    __pipeline_wait_prior(prologue - 1);

    __syncthreads();

    // --------------------------------
    // 主循环：算 tile t 的同时异步拷 tile t+1
    // --------------------------------

    for (int tile = 0; tile < tiles; ++tile) {

        const int stage = tile % 2;

        if (tile + 1 < tiles) {

            load_stage_async<BM, BN, BK, THREADS>(
                M, N, K, A, B, As, Bs,
                stage ^ 1, (tile + 1) * BK,
                block_row, block_col, tid);

            __pipeline_commit();
        }

        // ----------------------------
        // 计算：每个 k 步 4 次 ldmatrix + 8 次 mma
        // ----------------------------

        // A：本 warp 的 4 个 m8 块，位于 warp_m*4 起；lane i 提供第 (i/8) 块的第 (i%8) 行
        const uint32_t a_saddr0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[stage][0][warp_m * 4 + lane / 8][lane % 8][0]));
        const uint32_t a_saddr1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[stage][1][warp_m * 4 + lane / 8][lane % 8][0]));

        // B：本 warp 的 4 个 n8 块，位于 warp_n*4 起；地址语义与 A 相同（行地址），
        // 转置发生在结果寄存器布局上（正好匹配 mma 的 B 操作数）
        const uint32_t b_saddr0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[stage][0][warp_n * 4 + lane / 8][lane % 8][0]));
        const uint32_t b_saddr1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[stage][1][warp_n * 4 + lane / 8][lane % 8][0]));

        // 每个 k_blk 各搬一次：a_?[j] = m 块 j 在该 k_blk 的 fragment
        uint32_t a0[4];
        uint32_t a1[4];
        uint32_t b0[4];
        uint32_t b1[4];

        ldmatrix_x4(a0[0], a0[1], a0[2], a0[3], a_saddr0);
        ldmatrix_x4(a1[0], a1[1], a1[2], a1[3], a_saddr1);
        ldmatrix_x4_trans(b0[0], b0[1], b0[2], b0[3], b_saddr0);
        ldmatrix_x4_trans(b1[0], b1[1], b1[2], b1[3], b_saddr1);

        // mma(m, n) 的 A 操作数 = {a0[2m], a0[2m+1], a1[2m], a1[2m+1]}（(m0,k0),(m1,k0),(m0,k1),(m1,k1)）
        //            B 操作数 = {b0[n], b1[n]}（(n,k0),(n,k1)）
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {

                mma_16x8x16(
                    accum[m][n][0], accum[m][n][1],
                    accum[m][n][2], accum[m][n][3],
                    a0[m * 2], a0[m * 2 + 1],
                    a1[m * 2], a1[m * 2 + 1],
                    b0[n], b1[n]);
            }
        }

        // 等下一块 tile 落地
        if (tile + 1 < tiles) {

            __pipeline_wait_prior(0);

            __syncthreads();
        }
    }

    // ----------------------------
    // 写回 C：mma 的 C 布局为每线程 2x2
    //   (c0,c1) = 行 lane/4、列 2*(lane%4)+{0,1}
    //   (c2,c3) = 行 lane/4+8、同列
    // ----------------------------

    const int warp_row_base = block_row + warp_m * 32;
    const int warp_col_base = block_col + warp_n * 32;

#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {

            const int row = warp_row_base + m * 16 + lane / 4;
            const int col = warp_col_base + n * 8 + (lane % 4) * 2;

            float v0 = alpha * accum[m][n][0];
            float v1 = alpha * accum[m][n][1];
            float v2 = alpha * accum[m][n][2];
            float v3 = alpha * accum[m][n][3];

            if (beta != 0.0f) {
                if (row < M && col < N) {
                    v0 += beta * C[row * N + col];
                }
                if (row < M && col + 1 < N) {
                    v1 += beta * C[row * N + col + 1];
                }
                if (row + 8 < M && col < N) {
                    v2 += beta * C[(row + 8) * N + col];
                }
                if (row + 8 < M && col + 1 < N) {
                    v3 += beta * C[(row + 8) * N + col + 1];
                }
            }

            if (row < M && col + 1 < N) {

                *reinterpret_cast<float2*>(&C[row * N + col]) =
                    make_float2(v0, v1);

            } else if (row < M && col < N) {

                C[row * N + col] = v0;
            }

            if (row + 8 < M && col + 1 < N) {

                *reinterpret_cast<float2*>(&C[(row + 8) * N + col]) =
                    make_float2(v2, v3);

            } else if (row + 8 < M && col < N) {

                C[(row + 8) * N + col] = v2;
            }
        }
    }
}

// 要求 K、N 为 8 的倍数（16B 对齐）
void launch_gemm_tensor_core_mma(
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* C) {

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    dim3 block(128);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    gemm_tensor_core_mma_kernel<BM, BN, BK>
        <<<grid, block>>>(
            M, N, K,
            alpha, A, B,
            beta, C);
}
