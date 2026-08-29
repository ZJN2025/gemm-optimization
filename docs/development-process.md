# 项目开发流程（演示用）

> 本文按时间顺序还原整个项目的开发过程：每个阶段解决什么问题、结果如何、踩了什么坑、
> 学到了什么。最后附一份现场演示脚本。所有性能数据均为 RTX 5060 Laptop 实测
> （M=N=K=4096，完整数据见 `results/benchmark.csv`）。

## 项目定位

一个 CUDA GEMM 手写优化练习：从最朴素的实现出发，逐步加入共享内存分块、向量化访存、
cp.async 异步拷贝、软件流水、寄存器分块和 Tensor Core，最终与 CUTLASS 对比。
统一接口 `C = alpha * A * B + beta * C`（行主序），全部 kernel 由 `launch_gemm_xxx` 形式的
launcher 调用。

## 总览：9 个 kernel 的演进

| # | kernel | 关键手段 | TFLOPS @4096 | 相对 naive |
|---|---|---|---|---|
| 1 | gemm_naive | 无（基线） | 1.11 | 1.0× |
| 2 | gemm_shared_tiling | smem 分块 | 1.52 | 1.4× |
| 3 | gemm_vectorized | float4 + 64×64 + 4×4 分块 | 6.50 | 5.9× |
| 4 | gemm_async_copy | cp.async + 双缓冲（32×32） | 2.88 | 2.6× |
| 5 | gemm_double_buffer | 软件流水（32×32） | 3.87 | 3.5× |
| 6 | gemm_async_vectorized | cp.async + float4 + 双缓冲（64×64） | 8.68 | 7.8× |
| 7 | gemm_simt_128x128 | 128×128 + 8×8 寄存器分块 | 10.31 | 9.3× |
| 8 | gemm_tensor_core | wmma fp16，直接取数 | 6.85 | 6.2× |
| 9 | gemm_tensor_core_optimized | wmma + smem 中转 | 6.80 | 6.1× |

参考基线（CUTLASS v4.7.1）：simt sgemm 12.25 / tf32 sgemm 18.43 / hgemm 35.08。

---

## 阶段 0：先把标尺立起来（框架搭建）

**做了什么**：在写第一个 kernel 之前，先搭好三样东西：

1. **正确性标尺**：`tests/test_gemm.cu` —— CPU 参考实现对拍。每个 kernel 跑多组尺寸
   （含 100 这种不整除任何 tile 的边界尺寸）× 两组 (alpha, beta)，fp32 内核相对误差容差
   1e-3、fp16 内核 1e-2（输入舍入到 fp16 本身就引入 ~5e-4 误差）。任一失败返回非零退出码，
   可以挂进 ctest。
2. **性能标尺**：`benchmark/benchmark_gemm.cu` —— CUDA event 计时（warmup 10 次 + 正式 10 次），
   统一输出 `kernel,M,N,K,latency_ms,tflops` 到 CSV；后来加了单 kernel 命令行参数
   （`./build/benchmark_gemm gemm_xxx`），方便 ncu 分析前单独计时。
3. **可视化**：`tools/plot_benchmark.py` —— 把 CSV 画成 TFLOPS / 延迟 / 加速比三张面板。

**演示要点**：强调「先有正确性标尺，再有性能标尺，最后才谈优化」——后面 9 个 kernel
没有一个是拍脑袋通过的，每个都先过测试再过 benchmark。这也是为什么敢在 100×100×100
这种尺寸上压边界：边界处理错一点，测试立刻能抓出来。

## 阶段 1：gemm_naive —— 基线

一个线程算一个输出元素，A、B 全部从全局内存现取。

- **结果**：1.11 TFLOPS（4096），且 256~4096 几乎一条直线——完全被内存带宽卡死。
- **分析**：每个 A 元素被重复读了 N 次、每个 B 元素被重复读了 M 次（都在全局内存上）。
  算术强度极低，落在 roofline 模型的带宽墙下。
- **演示要点**：这是所有优化的出发点。记住 1.11 这个数，后面每个阶段都和它比。

## 阶段 2：gemm_shared_tiling —— 共享内存分块

把 A、B 切成 16×16 的 tile 放进共享内存，块内复用数据；越界元素填 0。

- **结果**：1.52 TFLOPS，只涨了 38%。
- **为什么涨得少**：共享内存加载是标量的（无向量化）、每个线程只算 1 个输出
  （无寄存器分块）、存在 bank conflict——这三件事在下一个阶段一次性解决。
- **踩坑**：写代码时把 `blockIdx.x` 敲成了 `blockIdx,x`（语法错误），launcher 名字也拼错过。
  教训：launcher 签名必须和头文件声明一致，`#include "gemm.hpp"` 让编译器帮你对。

## 阶段 3：gemm_vectorized —— 第一个大跳跃

三个手段一起上：**float4 向量化加载**（访存指令数 ÷4）、**64×64 大 tile**（复用率更高）、
**每线程 4×4 寄存器分块**（ILP 翻倍）。

- **结果**：6.50 TFLOPS，**4.3× 跳跃**——整个 fp32 路线上最大的一次单步提升。
- **演示要点**：这是全项目最重要的一个结论——**性能来自组合拳而不是单点技巧**。
  三个手段互相成全：float4 需要 16B 对齐（K、N 为 4 的倍数），大 tile 给足数据量，
  4×4 分块给足指令级并行。单独拆开任何一个都拿不到这个数。
- **边界处理设计**：越界的 float4 退化为逐元素加载（合法元素取值、非法元素填 0），
  这样 M、N 不整除 tile 也能算对，代价是边界块慢一点。

## 阶段 4 & 5：gemm_async_copy / gemm_double_buffer —— 反直觉的两步

分别用 cp.async（硬件异步拷贝，不经寄存器中转）和软件流水（先发下一块 tile 的全局读，
再算当前 tile）来隐藏访存延迟。但这两版沿用了 32×32 tile + 每线程 2×2 分块。

- **结果**：2.88 / 3.87 TFLOPS——**都比 vectorized 慢**。
- **演示要点**：这是全项目第二个重要结论——**「用了 cp.async 就一定快」是错的**。
  异步拷贝省的是延迟，而 32×32 + 2×2 的配置算力密度太低（每线程只有 4 个累加器），
  延迟根本没成为瓶颈，省了个寂寞。tile 配置 > 单点技巧。面试讲这一段比讲"我都优化了"
  更有说服力。

## 阶段 6：gemm_async_vectorized —— 验证组合收益

把阶段 4 的异步拷贝/双缓冲搬上阶段 3 的 64×64 配置（BK=16）。

- **结果**：8.68 TFLOPS，超过 vectorized 33%，也超过了两个 tensor core 内核。
- **演示要点**：同一个技巧，放在对的配置上才兑现收益——阶段 4 的"反例"和这里正好
  构成对照实验，论证 tile 配置才是大头。

## 阶段 7：gemm_simt_128x128 —— 经典 SIMT 设计

对齐 CUTLASS 经典 simt sgemm 的设计：128×128 tile、**每线程 8×8 = 64 个累加器**、
cp.async 双缓冲。8×8 分块把 FFMA:smem 访存比从 2:1 提到 4:1（每个 k 步 64 次 FFMA
只需要 16 次 smem 读取），FFMA 成为绝对主导。

- **结果**：10.31 TFLOPS，fp32 最佳，**达到 CUTLASS simt 的 84%**。
- **资源账**：64 累加器 + 操作数 ≈ 128 个寄存器/线程 → 2 blocks/SM（25% 占用率）、
  零 spill；16KB smem 远低于上限。寄存器分块在这里同时扮演「减少访存」和
  「提供 ILP 掩盖延迟」两个角色。
- **演示要点**：和 cutlass_sgemm_simt（12.25）同精度、同款设计，剩余 19% 差距来自
  CUTLASS 的 swizzle smem 布局（消除 bank conflict）、更深流水和向量化 epilogue——
  这些是明确的下一步，也是 ncu 可以验证的（看 smem bank conflict 计数）。

## 阶段 8：gemm_tensor_core 两版 —— 精度换吞吐，但"优化"没兑现

- 基础版：wmma 16×16×16（fp16 输入 / fp32 累积），**直接从全局内存取数**，零 smem、
  零同步开销 → 6.85 TFLOPS。
- "优化版"：加 smem 中转 + 8 warp 协作加载 → 6.80 TFLOPS，**没有任何提升**。
- **演示要点**：第三个重要结论——**优化必须用数据验证，不能想当然**。基础版由 L2 直接
  供数已经足够快；优化版多了标量 smem 加载 + 每 tile 两次 `__syncthreads()`，
  staging 开销正好吃掉复用收益。优化方向：smem 加载向量化 / cp.async、加大 tile 减少
  barrier。

## 阶段 9：与 CUTLASS 对比

同一张卡、同样尺寸、同样计时方法，引入 CUTLASS v4.7.1 作参考基线
（`benchmark/benchmark_cutlass.cu`，三种配置：simt sgemm / tf32 sgemm / hgemm）。

- **结果**：simt 12.25（同精度，差 19%）、tf32 18.43（精度略降）、hgemm 35.08（差 3.4×）。
- **环境坑（很好的排查故事）**：CUTLASS 默认 3-stage 配置（144KB smem）在这台卡上
  启动失败（`invalid argument`）——写了个最小诊断程序，发现 **5060 Laptop 的 opt-in
  共享内存上限只有 99KB**（数据中心卡是 227KB），于是改用 2-stage（96KB）。
  另外 sm_120 上 CUTLASS 的 tcgen05 新内核目前只支持 FP8/FP4，fp16/tf32 走经典
  mma.sync 路径——所以这组参考数据还不是这张卡的理论上限。

---

## 现场演示脚本（约 15 分钟）

**1. 环境与构建（1 分钟）**

```bash
cmake -S . -B build && cmake --build build -j   # 首次 configure 会下载 CUTLASS（~40MB）
```

**2. 正确性 + 可视化（2 分钟）**

```bash
./build/test_gemm                       # 9 个 kernel 全 PASS（含边界尺寸、beta 路径）
./build/benchmark_gemm                  # 全量跑一遍，写 results/benchmark.csv
.venv/bin/python tools/plot_benchmark.py
```

打开 `results/benchmark_analysis.png`，指着第三张面板讲加速比：naive → shared_tiling →
vectorized 的三级跳，async_copy/double_buffer 的回落，async_vectorized 和 simt_128x128
的爬升，CUTLASS 灰线的位置。

**3. 讲演进修路线（6 分钟）**：按上面的阶段表走，每个 kernel 一句话说手段、报一个数。
重点讲三个"反直觉"结论（阶段 3 的组合拳、阶段 4/5 的反例、阶段 8 的优化没兑现）。

**4. 单 kernel 深挖（3 分钟）**：

```bash
./build/benchmark_gemm gemm_simt_128x128    # 单独计时
ncu --set full -k regex:gemm_simt_128x128 ./build/benchmark_gemm gemm_simt_128x128
```

看 ncu 里的 Speed of Light、smem bank conflict、warp stall，和 CUTLASS simt 对照。

**5. 收尾（3 分钟）**：讲下一步方向——smem swizzle 布局消除 bank conflict、加深流水
（4-stage）、epilogue 向量化；以及 fp16 侧对齐 CUTLASS hgemm 的 4.5× 差距
（ldmatrix、warp 特化）。

## 面试高频问题（按项目内容自答）

- 「vectorized 为什么是最大的单步提升？」→ 阶段 3 的组合拳解释 + FFMA:访存比。
- 「cp.async 怎么用的，为什么 4/5 两版没变快？」→ 阶段 4/5 的反例 + tile 配置才是大头。
- 「8×8 分块和 4×4 分块差在哪？」→ 64 vs 16 个累加器，访存比翻倍，寄存器压力 128 但零 spill。
- 「tensor_core 版为什么比不过手写 SIMT？」→ 我的 wmma 版取数没优化；CUTLASS hgemm 35 TFLOPS
  证明 tensor core 潜力在，差距是工程量（ldmatrix/cp.async/warp 特化），不是硬件。
- 「怎么保证正确性？」→ 阶段 0 的测试框架：CPU 对拍、边界尺寸、beta 路径、相对容差。
- 「遇到什么坑？」→ 99KB smem 限制的诊断过程、blockIdx.x typo、half_t 类型包装。
