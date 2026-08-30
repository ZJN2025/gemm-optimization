# gemm-optimization

CUDA GEMM 优化练习项目：从最朴素的「一个线程算一个输出元素」出发，逐步加入共享内存分块、
向量化访存、cp.async 异步拷贝、软件流水和 Tensor Core（wmma），对比每一步带来的性能提升。

统一接口约定：`C = alpha * A * B + beta * C`，A、B、C 均为行主序 (row-major)，
所有 kernel 通过 `launch_gemm_xxx(M, N, K, alpha, A, B, beta, C)` 形式的 launcher 调用。

## 环境

| 项 | 版本/型号 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop（sm_120） |
| CUDA | 13.0 |
| CMake | ≥ 3.24 |
| 标准 | C++17 / CUDA 17 |
| 绘图 | Python 3 + matplotlib（项目自带 `.venv`） |

## 快速开始

```bash
# 编译
cmake -S . -B build && cmake --build build -j

# 正确性测试（任意目录均可执行；8 个 kernel 全 PASS 且退出码为 0）
./build/test_gemm
# 或 ctest --test-dir build

# 性能测试（必须从项目根目录执行，结果写入 results/benchmark.csv）
./build/benchmark_gemm

# 只跑单个 kernel（ncu 分析前单独计时很方便；结果写 results/benchmark_<name>.csv，不覆盖全量 CSV）
./build/benchmark_gemm gemm_simt_128x128
./build/benchmark_gemm -h    # 列出可用 kernel

# 网络不稳导致 configure 在下载 CUTLASS 时卡住/失败时的替代方案：
# 用已解压的源码目录跳过下载（_deps/cutlass-src 只要 configure 成功过一次就会缓存）
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_CUTLASS=$(pwd)/build/_deps/cutlass-src

# CUTLASS 参考基线（同样从项目根目录执行，结果写入 results/benchmark_cutlass.csv；
# CUTLASS v4.7.1 由 CMake 在 configure 时自动下载，缓存在 build/_deps）
./build/benchmark_cutlass
./build/benchmark_cutlass cutlass_hgemm   # 只跑指定配置

# 生成对比图（主线 kernel 一张 + 对照实验一张；参考基线数据在 benchmark_cutlass.csv 里）
.venv/bin/python tools/plot_benchmark.py

# 严谨模式：跑 3 次取中位数（本机不支持 nvidia-smi 锁频，用中位数抑制时钟漂移）
.venv/bin/python tools/median_benchmark.py                          # benchmark_gemm 3 次
.venv/bin/python tools/median_benchmark.py --cmd ./build/benchmark_cutlass \
    --out results/benchmark_cutlass.csv                            # 参考基线 3 次
```

## 项目结构

```
.
├── include/
│   ├── gemm.hpp                # 所有 launcher 的统一声明（新 kernel 必须在此登记）
│   └── cuda_check.hpp          # CUDA 错误检查辅助
├── kernels/                    # kernel 实现，每个文件 = __global__ kernel + launcher
│   ├── gemm_naive.cu
│   ├── gemm_shared_tiling.cu
│   ├── gemm_vectorized.cu
│   ├── gemm_async_copy.cu
│   ├── gemm_double_buffer.cu
│   ├── gemm_async_vectorized.cu
│   ├── gemm_tensor_core.cu
│   └── gemm_tensor_core_optimized.cu
├── tests/
│   └── test_gemm.cu            # CPU 参考实现对拍（边界尺寸 + alpha/beta 路径）
├── benchmark/
│   ├── benchmark_gemm.cu       # CUDA event 计时，输出 TFLOPS，写 results/benchmark.csv
│   └── benchmark_cutlass.cu    # CUTLASS 参考基线，写 results/benchmark_cutlass.csv
├── tools/
│   ├── plot_benchmark.py       # 把 benchmark.csv 画成对比图
│   └── check_smem.cu           # 探测本机共享内存上限（静态 / opt-in 两档，换卡后复测用）
├── docs/
│   ├── development-process.md  # 开发流程复盘：每个阶段的动机、结果、踩坑（演示/面试用）
│   └── knowledge-points.md     # 知识点整理：内存层次、分块、bank conflict 等，含 trade-off
├── results/                    # benchmark 产物（CSV + PNG）
└── profiler/                   # nsys / ncu 报告（.nsys-rep / .ncu-rep）
```

## Kernel 总览

| kernel | 关键技术 | 输入 | tile 配置 |
|---|---|---|---|
| `gemm_naive` | 一个线程算一个输出元素，无任何优化 | fp32 | 无 |
| `gemm_shared_tiling` | 共享内存分块，减少重复全局访存 | fp32 | 16×16 tile |
| `gemm_vectorized` | float4 向量化加载 + 每线程 4×4 寄存器分块（高 ILP） | fp32 | 64×64×8 |
| `gemm_async_copy` | cp.async 异步拷贝 + 双缓冲 smem | fp32 | 32×32×8 |
| `gemm_double_buffer` | 软件流水：先发下一 tile 的全局读取，再算当前 tile | fp32 | 32×32×8 |
| `gemm_async_vectorized` | 上述三者结合：cp.async + float4 + 双缓冲 | fp32 | 64×64×16 |
| `gemm_simt_128x128` | 128×128 大 tile + 每线程 8×8 寄存器分块 + cp.async 双缓冲（CUTLASS 经典 simt sgemm 同款设计） | fp32 | 128×128×8 |
| `gemm_tensor_core` | wmma 16×16×16，直接取数不经过 smem | fp16 | 16×16×16 |
| `gemm_tensor_core_optimized` | wmma + smem 中转 + 8 warp 协作加载 | fp16 | 32×64×16 |
| `gemm_tensor_core_mma` | 完整数据通路：cp.async → 8×8 块布局 smem → ldmatrix → mma.sync m16n8k16 | fp16 | 64×64×16 |

### 对照实验组（单变量控制）

主线 kernel 演进时经常同时改变多个变量（tile、分块、异步拷贝……），横向对比会混杂。
对照实验组全部由同一个全参数模板 kernel（`kernels/gemm_controlled.cu`，
骨架 = cp.async + float4 + N-stage 流水）实例化，**每个实验只改一个变量**：

| 实验 | 固定 | 只改变 | 对照组 kernel |
|---|---|---|---|
| tile 尺寸 | BK=8、4×4 分块、2-stage | BM=BN ∈ {32, 64, 128} | tile_32x32 / tile_64x64 / tile_128x128 |
| cp.async 有无 | 64×64×8、4×4 分块 | 加载路径（普通 load vs cp.async+双缓冲） | gemm_vectorized / tile_64x64 |
| BK（K 方向分块） | 64×64、4×4 分块、cp.async | BK ∈ {8, 16} | tile_64x64 / gemm_async_vectorized |
| 寄存器分块 | 128×128×8、2-stage | TM×TN ∈ {4×4, 4×8, 8×8} | tile_128x128 / rblock_4x8 / gemm_simt_128x128 |
| 流水深度 | 128×128×8、8×8 | STAGES ∈ {2, 4} | stages2_128x128 / stages4_128x128 |

`stages2_128x128` 同时是**实现一致性校验**：它与手写的 `gemm_simt_128x128` 配置完全相同，
若两者分数接近，说明通用模板的加载路径没有偷工减料。对照实验结果见下文「对照实验数据」。

### 尺寸约束

| kernel | 要求 | 原因 |
|---|---|---|
| `gemm_vectorized` | K、N 为 4 的倍数 | float4 全局访存需 16 字节对齐 |
| `gemm_async_vectorized` | K、N 为 4 的倍数 | 同上 |
| `gemm_simt_128x128` | K、N 为 4 的倍数 | 同上 |
| `gemm_tensor_core` | M、N、K 为 16 的倍数 | wmma tile 16×16，K 循环不判边界 |
| `gemm_tensor_core_optimized` | M 为 32 的倍数、N 为 64 的倍数 | warp 级 16×16 输出未做边界裁剪 |
| `gemm_tensor_core_mma` | K、N 为 8 的倍数 | 16B 对齐；M 任意（越界零填充 + 写回判界） |

## 性能结果

> 数据快照（2026-08-30，RTX 5060 Laptop，**3 次运行取中位数**——本机不支持 nvidia-smi
> 锁频，用中位数抑制笔记本 GPU 的时钟漂移；单次原始数据在 `results/benchmark_run<i>.csv`）。
> 完整数据见 `results/benchmark.csv` 和 `results/benchmark_cutlass.csv`，图见
> `results/benchmark_analysis.png`。

TFLOPS @ 4096×4096×4096（大尺寸才能体现 kernel 真实水平，256 那列受启动开销主导）：

| kernel | TFLOPS | 相对 naive |
|---|---|---|
| gemm_naive | 1.10 | 1.0× |
| gemm_shared_tiling | 1.53 | 1.4× |
| gemm_vectorized | 6.56 | 6.0× |
| gemm_async_copy | 2.87 | 2.6× |
| gemm_double_buffer | 3.88 | 3.5× |
| gemm_async_vectorized | 8.70 | 7.9× |
| gemm_simt_128x128 | 10.37 | 9.4× |
| gemm_tensor_core | 9.11 | 8.3× |
| gemm_tensor_core_optimized | 7.58 | 6.9× |
| gemm_tensor_core_mma | 19.48 | 17.7× |
| cutlass_sgemm_simt（参考） | 12.29 | 11.2× |
| cutlass_sgemm_tf32（参考） | 18.68 | 17.0× |
| cutlass_hgemm（参考） | 36.29 | 33.0× |
| cublas_sgemm（参考） | 11.72 | 10.7× |
| cublas_sgemm_tf32（参考） | 18.75 | 17.0× |
| cublas_hgemm（参考） | 37.34 | 33.9× |

几个值得注意的结论：

1. **fp32 的第一跳来自 vectorized**（1.5 → 6.6 TFLOPS）：float4 访存 + 64×64 大 tile + 每线程 4×4
   寄存器分块的组合拳，而不是某个单点技巧。
2. **async_copy / double_buffer 跑不过 vectorized**：它们的 tile 只有 32×32、每线程 2×2 分块，
   覆盖访存延迟的收益抵不过计算密度低的损失——「用了 cp.async 就一定快」是错的，tile 配置才是大头。
3. **async_vectorized（8.7 TFLOPS）验证了组合收益**：cp.async + float4 + 双缓冲搭上 64×64 配置后，
   超过了 vectorized 和两个 wmma 内核。
4. **simt_128x128（10.4 TFLOPS）是 fp32 最佳**：达到 cutlass simt 的 84%、cublas sgemm 的 88%。
5. **tensor_core_mma（19.5 TFLOPS）是 fp16 升级的成果**：wmma 直取 9.1 → cp.async + ldmatrix +
   mma.sync 数据通路 19.5（2.1×），超过 cutlass/cuBLAS 的 tf32 路径；离 cublas hgemm 的 37.3
   还有 52% 差距——它们还有 warp 特化、更大 tile、更深流水、TMA 这几张牌。

### 对照实验数据（单变量控制，4096 实测）

| 实验 | 对照组 | TFLOPS | 结论 |
|---|---|---|---|
| tile 尺寸（4×4 分块固定） | 32×32 → 64×64 → 128×128 | 6.71 → 8.23 → **7.53** | **tile 不是越大越好**：4×4 分块下 64×64 最优，128×128 因 1024 线程/block 拉低 occupancy 反而回落 |
| cp.async 有无（64×64×8、4×4 固定） | vectorized → tile_64x64 | 6.56 → 8.23 | cp.async+双缓冲的**纯贡献 +25%**（配置相同时） |
| BK（64×64、4×4 固定） | BK=8 → BK=16 | 8.23 → 8.70 | BK=16 略好（+6%），barrier 减半的边际收益很小 |
| 寄存器分块（128×128×8 固定） | 4×4 → 4×8 → 8×8 | 7.53 → 6.84 → **10.37** | **非单调**：4×8 这种"长条"分块因线程映射/bank conflict 反而最差；8×8 的 FFMA:访存比优势才是真金 |
| 流水深度（128×128×8、8×8 固定） | 2-stage → 4-stage | 10.00 → 10.05 | **4-stage 几乎无收益（+0.5%）**：2-stage + 高 ILP 已遮住延迟，验证收益递减 |
| 实现一致性校验 | 手写 simt_128x128 → 模板 stages2_128x128 | 10.37 → 10.00 | 同配置差 3.6%，确认模板化实现没有偷工减料 |

### 与 CUTLASS 的对比

CUTLASS v4.7.1（CMake configure 时自动下载），同一张卡、同样尺寸、同样计时方法，
完整数据在 `results/benchmark_cutlass.csv`，图中以灰色虚线/点线叠加：

1. **cutlass_sgemm_simt（精确 fp32，FFMA）12.3 TFLOPS vs 手写 simt_128x128 10.3**——两者同精度、
   同款设计（128×128 tile / 8×8 分块），剩余 19% 差距主要来自 CUTLASS 的 swizzle 布局
   （消除 smem bank conflict）、更深的流水和向量化 epilogue。
2. **cutlass_sgemm_tf32 18.4 TFLOPS**——fp32 走 tf32 tensor core，精度略降（10 位尾数）、
   吞吐大增，是工业界的常用折中。
3. **cutlass_hgemm 35.1 TFLOPS vs 手写 tensor_core 7.9——4.5×**，差距来自 cp.async + ldmatrix
   向量化取数 + 多 stage 流水 + warp 特化等系统性工程。

两个与本机环境相关的说明：

- RTX 5060 Laptop 的 opt-in 共享内存上限只有 **99KB**（数据中心卡是 227KB），CUTLASS 默认的
  3-stage 配置（144KB）会启动失败，因此这里用 2-stage（96KB）。
- sm_120 上 CUTLASS 官方的 tcgen05 新内核目前只支持 FP8/FP4 类型，fp16/tf32 走的是经典
  Ampere 风格 mma.sync 路径，所以这里的 CUTLASS 数据还不是这张卡的理论上限。

## 测试说明

`test_gemm` 对每个 kernel 跑 3 组尺寸 × 2 组 (alpha, beta)，与 CPU 参考实现对拍：

- fp32 内核：64 / 100 / 256 三组尺寸——100 不整除任何 tile，专门压边界
- fp16 内核：64 / 128 / 256（满足 16 倍数约束）
- 两组参数分别覆盖 `beta = 0`（纯 A·B）和 `beta ≠ 0`（C 初值参与运算）两条路径
- 容差为相对误差：fp32 内核 1e-3，fp16 内核 1e-2（输入舍入到 fp16 本身引入 ~5e-4 误差）

## Profiler 分析

> TODO：nsys / ncu 结果待用 Nsight 跑完后补充到这里，报告文件放进 `profiler/` 目录。

建议的采集命令（benchmark 会跑全部 kernel，用 `-k` 过滤目标）：

```bash
# Nsight Compute：单 kernel 的访存 / 计算 / 停顿分析
ncu --set full -k regex:gemm_async_vectorized -o profiler/gemm_async_vectorized \
    ./build/benchmark_gemm

# Nsight Systems：整体时间线（kernel 重叠、拷贝、启动开销）
nsys profile -o profiler/timeline ./build/benchmark_gemm
```

目前 `profiler/` 里有 `gemm_naive` 的 full 和 roofline 报告（`.ncu-rep`，可直接用 Nsight Compute 打开）。

## 如何新增一个 kernel

1. 新建 `kernels/gemm_xxx.cu`：`__global__` kernel + `launch_gemm_xxx` launcher，
   launcher 命名统一 `launch_` 前缀，签名与 `include/gemm.hpp` 的声明完全一致，
   文件里 `#include "gemm.hpp"`（让定义和声明互相做类型检查）；
2. `include/gemm.hpp` 添加声明，如有尺寸约束写进注释；
3. `CMakeLists.txt` 的 `gemm_kernels` 源文件列表加一行（漏了会 undefined reference）；
4. `tests/test_gemm.cu` 注册一行：
   `run_kernel_tests<float>("gemm_xxx", launch_gemm_xxx, fp32_tolerance, fp32_sizes);`
   （fp16 内核用 `<half>` + `fp16_tolerance`，测试尺寸需满足 kernel 约束）；
5. `benchmark/benchmark_gemm.cu` 注册一行：
   `{"gemm_xxx", launch_gemm_xxx, nullptr},`（fp16 内核反过来：`{name, nullptr, launch_xxx}`）。

然后走一遍完整流程：

```bash
cmake -S . -B build && cmake --build build -j   # 编译
./build/test_gemm                               # 正确性：全 PASS 才能进下一步
./build/benchmark_gemm                          # 性能（项目根目录执行）
.venv/bin/python tools/plot_benchmark.py        # 更新对比图
```
