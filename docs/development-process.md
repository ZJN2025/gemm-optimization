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
| 1 | gemm_naive | 无（基线） | 1.10 | 1.0× |
| 2 | gemm_shared_tiling | smem 分块 | 1.53 | 1.4× |
| 3 | gemm_vectorized | float4 + 64×64 + 4×4 分块 | 6.56 | 6.0× |
| 4 | gemm_async_copy | cp.async + 双缓冲（32×32） | 2.87 | 2.6× |
| 5 | gemm_double_buffer | 软件流水（32×32） | 3.88 | 3.5× |
| 6 | gemm_async_vectorized | cp.async + float4 + 双缓冲（64×64） | 8.70 | 7.9× |
| 7 | gemm_simt_128x128 | 128×128 + 8×8 寄存器分块 | 10.37 | 9.4× |
| 8 | gemm_tensor_core | wmma fp16，直接取数 | 9.11 | 8.3× |
| 9 | gemm_tensor_core_optimized | wmma + smem 中转 | 7.58 | 6.9× |
| 10 | gemm_tensor_core_mma | cp.async + ldmatrix + mma.sync | 19.48 | 17.7× |

另有 6 个单变量对照配置（阶段 10）：tile 尺寸组 / cp.async 有无组 / BK 组 / 寄存器分块组 /
流水深度组，见 `kernels/gemm_controlled.cu`。

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
  零同步开销 → 9.11 TFLOPS。
- "优化版"：加 smem 中转 + 8 warp 协作加载 → 7.58 TFLOPS，**没有任何提升**。
- **演示要点**：第三个重要结论——**优化必须用数据验证，不能想当然**。基础版由 L2 直接
  供数已经足够快；优化版多了标量 smem 加载 + 每 tile 两次 `__syncthreads()`，
  staging 开销正好吃掉复用收益。优化方向：smem 加载向量化 / cp.async、加大 tile 减少
  barrier。

## 阶段 8.5：gemm_tensor_core_mma —— fp16 完整数据通路

wmma 只是"能跑通"级别，真正该做的是现代 tensor core 数据通路：**cp.async 把 16B 块
（8 个 half）拷进为 ldmatrix 定制的 8×8 块布局 smem → ldmatrix 一次搬 4 块 fragment →
mma.sync m16n8k16 直接算**。64×64×16 tile、4 warps、2-stage。

- **结果**：**19.48 TFLOPS，wmma 直取的 2.1×**，超过 cutlass/cuBLAS 的 tf32 路径。
- **调试故事（很好的面试素材）**：第一版直接报 `misaligned address`——ldmatrix 的
  `.trans` 变体我按"列地址"传参（2B 对齐），查 cutlass 的 cute 源码发现它要求的是
  16B 对齐的**行地址**；修完地址后结果仍错，用一个"已知输入探针"（A[m][k]=m、B[k][n]=n，
  期望 C=16·m·n）打出整块输出矩阵，从错位模式反推出 **ldmatrix.x4 的地址语义**：
  32 个 lane 各提供一个地址（lane i → 第 i/8 块矩阵的第 i%8 行），而不是"8 个线程
  给地址、硬件自动 +128B"。修正 `lane/8` 块偏移后 256 个元素全对。教训：**fragment
  布局这类硬件契约，先查库的实现，再用探针实验闭环验证**，凭记忆写必踩坑。
- **剩余差距**：cublas_hgemm 37.3，我们还有 52% 差距——对方有 warp 特化（生产者/
  消费者分工）、128×256 大 tile、更深流水和 TMA，这是明确的下一步路线图。

## 阶段 9：与 CUTLASS / cuBLAS 对比

同一张卡、同样尺寸、同样计时方法，引入两套参考基线（`benchmark/benchmark_cutlass.cu`）：
CUTLASS v4.7.1（simt / tf32 / hgemm）+ cuBLAS（sgemm / sgemm_tf32 / hgemm）。

- **结果**（中位数）：cutlass simt 12.29 / tf32 18.68 / hgemm 36.29；
  cublas sgemm 11.72 / tf32 18.75 / hgemm 37.34。
- **方法论升级**：本机不支持 nvidia-smi 锁频（GeForce 笔记本卡），改跑 **3 次取中位数**
  （`tools/median_benchmark.py`）——此前两次单跑 wmma 基础版差 15% 就是时钟漂移。
  从此所有文档数字都是中位数。
- **环境坑（很好的排查故事）**：CUTLASS 默认 3-stage 配置（144KB smem）在这台卡上
  启动失败（`invalid argument`）——写了个最小诊断程序，发现 **5060 Laptop 的 opt-in
  共享内存上限只有 99KB**（数据中心卡是 227KB），于是改用 2-stage（96KB）。
  另外 sm_120 上 CUTLASS 的 tcgen05 新内核目前只支持 FP8/FP4，fp16/tf32 走经典
  mma.sync 路径——所以这组参考数据还不是这张卡的理论上限。

## 阶段 10：对照实验（单变量控制）—— 让每个结论站得住

**为什么需要这一步**：回顾主线演进，很多对比是"多变量一起变"的：async_copy 相对 vectorized
同时改了 tile 尺寸（32×32 vs 64×64）、寄存器分块（2×2 vs 4×4）和加载路径（cp.async vs 普通
load）——"它输了"这个结论是对的，但**输在哪个变量上说不清**，面试讲起来就不硬气。

**怎么做**：写一个全参数模板 kernel（`kernels/gemm_controlled.cu`，骨架 = cp.async +
float4 + N-stage 流水，所有变量 BM/BN/BK/TM/TN/STAGES 都是模板参数），用一组 launcher
实例化出**只改一个变量**的对照组：

| 实验 | 固定 | 只改变 | 回答什么问题 |
|---|---|---|---|
| tile 尺寸 | BK=8、4×4、2-stage | 32→64→128 | tile 越大越快吗？快多少？ |
| cp.async 有无 | 64×64×8、4×4 | 普通 load vs cp.async+双缓冲 | 单看 cp.async 值多少钱？ |
| BK | 64×64、4×4、cp.async | 8 vs 16 | K 分块更深有收益吗？ |
| 寄存器分块 | 128×128×8、2-stage | 4×4 → 4×8 → 8×8 | 累加器越多越快吗？ |
| 流水深度 | 128×128×8、8×8 | 2-stage → 4-stage | 更深的流水还有油水吗？ |

另外 `stages2_128x128` 与手写的 `gemm_simt_128x128` 配置完全相同，是一组**实现一致性
校验**：两者分数接近才说明模板的加载路径没偷工减料。

**结果**（4096 实测，完整数据在 results/benchmark.csv）：

| 实验 | 数据点 | 结论 |
|---|---|---|
| tile 尺寸 | 32×32: 6.71 → 64×64: 8.23 → 128×128: 7.53 | **tile 不是越大越好**：4×4 分块下 64×64 最优；128×128 要 1024 线程/block，occupancy 掉到 1 block/SM，复用增益被抵消 |
| cp.async 有无 | vectorized: 6.56 → tile_64x64: 8.23 | 配置完全相同时，cp.async+双缓冲纯贡献 **+25%**（主线里 32×32 的 async_copy"无效"不是 cp.async 的锅，是 tile 配置的锅） |
| BK | BK=8: 8.23 → BK=16: 8.70 | 只 +6%：barrier 减半的边际收益很小 |
| 寄存器分块 | 4×4: 7.53 → 4×8: 6.84 → 8×8: 10.37 | **非单调**："长条"分块 4×8 因线程映射和 bank conflict 反而最差；8×8 才是 FFMA:访存比翻倍的赢家 |
| 流水深度 | 2-stage: 10.00 → 4-stage: 10.05 | **4-stage 几乎无收益（+0.5%）**：2-stage + 8×8 的高 ILP 已经把延迟遮住了——"收益递减"有了本机数据背书 |
| 实现一致性 | 手写 simt_128x128: 10.37 → 模板 stages2_128x128: 10.00 | 同配置差 3.6%，确认模板化实现没有偷工减料，跨实现对比可信 |

**演示要点**：这段放在讲完主线之后，作为"方法论"收尾——主线讲"我做了什么"，对照实验讲
"我怎么证明每个手段各自值多少"。面试官问"你怎么确定是 XX 带来的提升"时，直接指这张表。
两个非单调的发现（tile 128×128 回落、4×8 分块最差）尤其值得主动讲：**优化没有免费午餐，
每个参数都有拐点，只有对照实验能定位拐点**。

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
- 「cp.async 怎么用的，为什么 4/5 两版没变快？」→ 阶段 4/5 的反例 + 对照实验给出纯贡献 +25%。
- 「8×8 分块和 4×4 分块差在哪？」→ 64 vs 16 个累加器，访存比翻倍，寄存器压力 128 但零 spill。
- 「fp16 的 19.5 是怎么从 9.1 涨上来的？」→ cp.async + ldmatrix + mma.sync 数据通路，
  取数向量化 + smem 块布局为 ldmatrix 定做；离 cublas hgemm 37.3 还差 52%，差在 warp 特化、
  大 tile、深流水、TMA。
- 「ldmatrix 有什么坑？」→ x4 变体是 32 个 lane 各提供一个地址（lane i → 第 i/8 块的第 i%8 行），
  .trans 也要求 16B 行地址；我用探针实验（A[m][k]=m、B[k][n]=n → C=16mn）定位了这两处。
- 「怎么保证正确性？」→ 阶段 0 的测试框架：CPU 对拍、边界尺寸、beta 路径、相对容差。
- 「性能数字可信吗？」→ 本机不支持锁频，全部数字是 3 次运行的中位数，原始数据留档。
- 「遇到什么坑？」→ 99KB smem 限制的诊断过程、blockIdx.x typo、half_t 类型包装。
