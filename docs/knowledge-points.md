# 知识点整理（含 trade-off 考量）

> 以本项目 9 个 kernel 为素材整理的 CUDA GEMM 优化知识点。每个点都给出：
> 是什么 → 在本项目里的体现 → trade-off 考量。数据为 RTX 5060 Laptop 实测。

## 1. 性能模型：为什么 naive 只有 1 TFLOPS

**内存层次**：全局内存（GB 级，~400 周期延迟）→ L2（~几十 MB）→ 共享内存（每 SM 几十 KB，
~20 周期）→ 寄存器（每 SM 64K×4B，1 周期）。GEMM 优化的本质是**把数据尽量放在
离计算单元近的地方、并让每次取数被尽可能多次计算复用**。

**算术强度 / roofline**：GEMM 的算术强度 = 2MNK / (2MNK 次访存的字节数)。naive 实现里
每个 A 元素被重复读 N 次、每个 B 元素被重复读 M 次，且全部发生在全局内存上——
算术强度 ≈ 1 FLOP/4B，稳稳落在带宽墙下，算力再高也没用。所以 naive 的 1.11 TFLOPS
基本等于「全局内存带宽 / 每 FLOP 字节数」。

**本项目对应**：gemm_naive 1.11 → shared_tiling 1.52（省掉部分全局重复读）→
vectorized 6.50（复用率 + 向量化一起上）→ simt_128x128 10.31。

## 2. 共享内存分块（tiling）

**是什么**：把 A、B 切成 tile 放进 smem，块内所有线程共享，把 M×N×K 的全局访存降到
(M×K×⌈N/BN⌉ + K×N×⌈M/BM⌉)。

**本项目**：16×16（shared_tiling）→ 64×64（vectorized）→ 128×128（simt_128x128）。

**trade-off：tile 越大，复用越高，但 smem 占用和 barrier 同步也越多**
- smem 是稀缺资源：这台 5060 Laptop 的 opt-in 上限只有 **99KB**（数据中心卡 227KB），
  而且 smem 大小直接限制 occupancy（每 SM 能放几个 block）。
- tile 越大单块数据越多，但每次 `__syncthreads()` 前所有线程都要等最慢的那个；
  tile 太大 → 每个 tile 计算太久 → 等待时间占比上升。
- 对照实验（固定 4×4 分块、BK=8、2-stage，只改 tile）：32×32: 6.67 → 64×64: 8.37 →
  128×128: **7.52**——**非单调**！128×128 要 1024 线程/block，occupancy 掉到 1 block/SM，
  复用增益被抵消。结论修正为：tile 大小存在拐点，拐点位置取决于分块方式
  （8×8 分块时 128×128 才是最优，见 simt_128x128 的 10.43）。

## 3. 向量化访存（float4）

**是什么**：一次 load 指令取 16 字节（float4），访存指令数 ÷4，LSU 压力骤降。
要求地址 16 字节对齐。

**本项目**：vectorized / async_vectorized / simt_128x128 的全局加载全部是 float4；
simt_128x128 连 smem 读 B 也是 float4（B 沿 N 连续）。

**trade-off：对齐约束换性能**
- 全局 float4 要求行主序下 **K、N 为 4 的倍数**（行首地址 16B 对齐 + 行内偏移 16B 对齐），
  否则要么改布局（拷贝/转置），要么退化为逐元素。
- 越界 tile 不能整段 16B 拷（会读越界），必须**退化路径**：逐元素判界、非法填 0。
  代价是边界块的代码路径变慢，换来任意尺寸的正确性。
- 对 smem 的 float4 读取还有隐藏要求：读出的 4 个元素必须在 smem 里连续（受布局限制，
  见 bank conflict 一节）。

## 4. 寄存器分块与指令级并行（ILP）

**是什么**：每个线程一次算 TM×TN 个输出，累加器全在寄存器里；这些 FFMA 相互独立，
GPU 可以在等待访存时先算别的，隐藏延迟。

**本项目**：4×4=16 累加器（vectorized）→ 8×8=64 累加器（simt_128x128）。

**trade-off：累加器换访存与延迟，但挤占寄存器**
- 对照实验（固定 128×128×8、2-stage，只改分块）：4×4: 7.52 → 4×8: **6.87** → 8×8: **10.43**。
  **非单调**——4×8 这种"长条"分块反而最差（线程映射变化导致更差的 bank conflict 模式 +
  FFMA:访存比没质变）；只有 8×8 把 **FFMA:smem 访存比**从 2:1 提到 4:1（每个 k 步 64 次
  FFMA 只配 16 次 smem 读取），量变才引发质变。
- 代价是寄存器压力：64 累加器 + 操作数 ≈ 128 regs/线程（刚好零 spill），
  直接决定 occupancy——128 regs × 256 线程 = 32K regs，每 SM 64K → **2 blocks/SM（25%）**。
- 再往上（如 12×12 分块）会 spill 到 local memory（其实是全局内存），性能崩盘。
  所以寄存器分块不是越大越好：**ILP vs occupancy 的平衡点**就是零 spill 的最大分块。
  本项目 8×8 恰好卡在 128 regs 上，是这条 trade-off 的活例子。

## 5. cp.async 异步拷贝

**是什么**：Ampere 起的 `cp.async`（本项目用 `__pipeline_memcpy_async`）把全局→共享的
拷贝变成异步操作：数据**不经寄存器中转**、不阻塞计算，配合 `commit/wait` 组管理。

**本项目**：async_copy、async_vectorized、simt_128x128 的加载路径。

**trade-off：省的是延迟，不是带宽**
- cp.async 只解决「等数据」的问题。如果 kernel 本来就是计算/带宽瓶颈（延迟被别的东西
  盖住），cp.async 的收益就是零甚至负的——async_copy（32×32 + 2×2 分块）2.88 TFLOPS
  跑不过 vectorized 6.50 就是这个道理。
- 它必须和**足够深的流水**配合：拷贝要与计算重叠，重叠的前提是有独立的缓冲区
  （见下一节）和足够多的并行计算。
- 16B 拷贝要求全局与 smem 两侧都 16B 对齐；跨 tile 边界的元素不能 16B 拷，
  退化路径逐元素处理。

## 6. 软件流水 / 双缓冲

**是什么**：把 tile 的「加载 → 计算」串行流水拆成重叠：算第 t 块的同时加载第 t+1 块。
用两份（或多份）smem 缓冲区交替。

**本项目**：double_buffer（寄存器预取版，无 cp.async）、async_copy / async_vectorized /
simt_128x128（cp.async 版，2-stage）。

**trade-off：流水深度 vs smem 占用 vs 同步复杂度**
- stage 数越多，访存延迟隐藏越充分，但 smem 按 stage 数线性增长：本项目 128×128×8 的
  2-stage 是 16KB，4-stage 就是 32KB——在这台 99KB 的卡上仍放得下，但在小卡上
  可能直接挤掉 occupancy。
- stage 变多，`wait_prior` 的组管理、barrier 顺序都更复杂，出错窗口变大；
  且每一 stage 的收益是递减的（2-stage 已经遮掉了大部分延迟）。
- 反例：tensor_core_optimized 的失败部分原因就是流水设计不当——smem 中转阶段
  用了两次 `__syncthreads()`，barrier 本身成了开销。

## 7. Bank conflict（smem 的隐藏杀手）

**是什么**：smem 分 32 个 bank（4B 宽）。一条指令里若 warp 的多个线程访问同一 bank 的
**不同地址**，访问被串行化 n 路（n-way conflict），smem 带宽 ÷n。访问**同一地址**则是
广播，不冲突。

**本项目逐 kernel 的冲突账**：
- simt_128x128 读 A：8 个标量（ty 两组 → 2-way conflict，其余广播）；
- 读 B：2 个 float4（tx 与 tx+4 同 bank → 4-way conflict）；
- 因为 8×8 分块让 FFMA 主导（第 4 节），这些冲突没有成为瓶颈——但 CUTLASS simt 比我们
  快的那 19% 主要就来自它用 **swizzle 布局**把冲突清零了。

**trade-off：布局变换 vs 加载路径复杂度**
- 消除 conflict 的手段：行 padding（只能修部分模式）、xor swizzle / interleave 布局
  （改物理列序）、转置存储。这些手段**改变 smem 物理布局**，会导致：
  - cp.async 16B 拷贝要求目标地址连续——swizzle 后的目标不连续，要么改用 4B 拷贝，
    要么先自然布局再转置（多一次 smem 往返 + barrier）；
  - 读写两方都要记得换算物理地址，代码复杂度明显上升。
- 什么时候值得做：等 FFMA:访存比优化到位、ncu 显示 smem 利用率触顶之后。

## 8. Tensor Core / mma.sync / ldmatrix

**是什么**：专用矩阵乘单元。分层理解：
- **wmma API**：封装层，`load_matrix_sync` 从全局内存直接取数，布局由 API 定；
- **mma.sync PTX**（m16n8k16 等）：真正的指令，操作数是每线程 2~8 个 half 的 fragment；
- **ldmatrix**：smem → fragment 的向量化搬运指令，一条指令搬 8×8 fp16 块
  （x4 变体一次 4 块），`.trans` 变体输出转置后的 fragment（正好匹配 mma 的 B 操作数）。

**本项目实测（4096，中位数）**：
- wmma 直取（gemm_tensor_core）9.11、wmma + 标量 smem 中转 7.58；
- **cp.async + ldmatrix + mma.sync（gemm_tensor_core_mma）19.48**——数据通路的差距是 2.1×；
- cublas hgemm 37.34：剩余 52% 差在 warp 特化、128×256 大 tile、更深流水、TMA。

**ldmatrix 的两个硬件契约（本项目踩坑实录）**：
1. **地址语义**：x4 变体是 **32 个 lane 各提供一个地址**——lane i 提供第 (i/8) 块矩阵的
   第 (i%8) 行地址（16B 对齐），不是"8 个线程给地址、硬件自动 +128B"。写错后结果
   错得很有规律，用一个「已知输入探针」（A[m][k]=m、B[k][n]=n → C=16·m·n）打出整块
   输出即可从错位模式反推。
2. **.trans 也要 16B 行地址**：转置发生在结果寄存器布局上，地址侧与非 trans 相同。

**trade-off：精度与布局约束换吞吐**
- **精度**：fp16 尾数 11 位 → 相对误差 ~5e-4（测试容差因此放到 1e-2）；tf32 是
  10 位尾数（cuBLAS tf32 18.75 TFLOPS 就是这个 trade-off）。要精确 fp32 就只能走
  FFMA 路线（cutlass simt 12.29）。
- **布局**：smem 布局要为 ldmatrix 定做（本项目：8×8 块连续 128B，m/n 方向连续）——
  恰好 16B 的 cp.async 全局块映射 1:1 到块行，两头的向量化都对上了。
- **数据路径 trade-off（最有意思的一条）**：wmma 直取（L2 供数、零同步）反而快过
  "标量 smem 中转"——中转只有在其加载（向量化/cp.async）和消费（ldmatrix）都高效时
  才划算，否则 staging + barrier 的开销 > 复用收益。
- 环境边界：sm_120 上 CUTLASS 的 tcgen05（第五代 TC）内核目前只支持 FP8/FP4 类型，
  fp16/tf32 只能走经典 mma.sync 路径。

## 9. Occupancy 与延迟隐藏

**是什么**：每 SM 同时驻留的 warp 数。限制来自寄存器（64K/SM）、smem、线程数上限。
足够的 warp 才能通过切换隐藏访存/算术延迟。

**本项目**：simt_128x128 = 128 regs + 16KB smem → 2 blocks/SM = 512 线程 = 16 warps
（25% 占用率）。但每个线程有 64 个独立累加器，ILP 极强，25% 的占用率足以打满 FFMA——
**高 ILP 可以弥补低 occupancy**，这是现代 GEMM 的典型取舍：为了零 spill 的 8×8 分块，
宁肯只放 2 个 block。

**trade-off：occupancy vs 每线程工作量**
- 提高 occupancy 通常意味着减寄存器/smem → 每线程分块变小 → ILP 和复用变差。
- 降低 occupancy 拿每线程高 ILP，前提是分块够大、不 spill。
- 判据永远只有一个：测。ncu 的 warp stall 和 issue 利用率能告诉你卡在哪。

## 10. 正确性与测试方法

**为什么要有 CPU 对拍**：GPU 结果和 CPU 参考的浮点累加**顺序不同**（tile 化、寄存器
分块都会重排求和），逐位相等不现实，所以用**相对误差**判通过：误差 ≤ 容差 ×
max(1, |参考值|)。本项目 fp32 容差 1e-3、fp16 容差 1e-2。

**要测什么**：
- 边界尺寸（100 不整除任何 tile）：测零填充、退化加载路径、写回判界；
- beta ≠ 0：C 初值参与运算的路径（tensor core 内核有专门分支）；
- 对齐约束（K、N %4）要么在测试尺寸里覆盖，要么在文档里写明。

**trade-off：容差定多少**
- 太松（如 1e-1）会漏掉真 bug（比如某行漏算、填充错误）；
- 太紧（如 fp16 也用 1e-3）会让「输入舍入」这种固有误差误报失败。
- 正确做法：先算固有误差量级（fp16 舍入 ~5e-4/元素，K=256 累计 ~1e-3 相对），
  容差取它的 10 倍量级。

## 11. 工程化与工具链知识

- **统一接口的价值**：`gemm.hpp` 集中声明所有 launcher → 测试/benchmark 靠函数指针
  表批量注册，新增 kernel 只需 4 个文件各加一行。类型检查、防拼写错误。
- **CUDA event 计时**：warmup（预热 JIT/时钟）+ 多次取平均；单 kernel 模式写独立 CSV，
  不污染全量数据。
- **本机环境边界**：5060 Laptop opt-in smem 99KB；CUDA 13.0 + sm_120；
  CUTLASS 4.x 的 `cutlass::half_t` 是包装类型（用 `half_t::convert(float)` 构造，
  不是 `__half` 别名）。
- **性能数据要同机同尺寸同方法**对比：CUTLASS/cuBLAS 对比的可靠性来自「同一张卡、同样尺寸、
  同样计时方法、同一天跑」。
- **时钟漂移的应对**：消费级/笔记本卡通常不支持 nvidia-smi 锁频，单次测量的数字会飘
  （本项目同 kernel 两次单跑差过 15%）。替代方案：重复 3 次取中位数（tools/median_benchmark.py），
  原始数据留档（results/*_run<i>.csv），文档只引中位数。

## 12. 对照实验与单变量控制（性能调优的方法论）

**是什么**：性能优化是"改一个变量 → 测 → 看结论"的循环。如果一次改动牵扯多个变量，
测出来的差异就无法归因——这是调优最大的方法论陷阱。

**本项目怎么做的**：把 kernel 写成全参数模板（BM/BN/BK/TM/TN/STAGES 全部模板化，
见 `kernels/gemm_controlled.cu`），同一份代码实例化出只差一个变量的对照组：
tile 尺寸组（32/64/128，其余固定）、cp.async 有无组、BK 组、寄存器分块组、流水深度组。
每组内两条曲线之差就是该变量的"纯贡献"。

**具体操作要点**：
- 先列变量清单，再设计最小改动组——两两之间只允许一个变量不同；
- 同机、同尺寸、同计时方法、尽量同一次运行内测完（排除时钟漂移）；
- 用「实现一致性校验」排除实现差异：同配置的两个实现分数接近，才能跨实现对比
  （本项目 stages2_128x128 vs 手写 simt_128x128 就是这组校验）；
- 每组实验只问一个问题："tile 越大越快吗？"而不是"哪个配置最好"——
  后者是 sweep，前者才是对照。

**trade-off**：对照实验的代价是**代码量**（每个对照组都是一个实例）和**运行时间**
（kernel 数量 × 尺寸组合），收益是每个结论都可归因、面试可讲。
值得为它付出的复杂度边界：模板化一次（成本一次性），之后所有对照都只是加一行实例化。

## Trade-off 汇总速查

| 取舍 | 左侧 | 右侧 | 本项目结论 |
|---|---|---|---|
| tile 大小 | 复用率↑ | smem↑ / 同步开销↑ / 线程数↑ | 对照实验：64×64 最优，128×128 回落（4×4 分块下）；拐点随分块方式移动 |
| 寄存器分块 | FFMA:访存比↑、ILP↑ | 寄存器↑ → occupancy↓、可能 spill | 对照实验：8×8 ≫ 4×4 > 4×8（非单调）；8×8 恰好 128 regs 零 spill |
| 向量化 | 指令数÷4 | K、N 需 4 对齐 + 越界退化路径 | 换 3× 收益，值得 |
| cp.async | 延迟隐藏 | 只对延迟瓶颈有效；需配合大配置 | 对照实验：同配置纯贡献 +27%；32×32 上无效是 tile 的锅 |
| BK（K 分块） | barrier 减半 | smem×BK | 对照实验：BK=16 比 8 仅 +3% |
| 流水深度 | 重叠更充分 | smem×stage、复杂度↑ | 对照实验：4-stage 比 2-stage 仅 +0.5%，收益递减实锤 |
| swizzle 布局 | bank conflict→0 | cp.async 目标不连续、代码复杂 | 与 cutlass simt 差 16% 的主要来源，下一步 |
| tensor core 数据通路 | 吞吐 4×+ | fp16 精度损失、布局约束 | 通路配齐（cp.async+ldmatrix+mma.sync）后 19.5 = cublas hgemm 的 52% |
| smem 中转 | 全局访存↓ | barrier + staging 开销 | 标量中转反而输——只有配 ldmatrix 才划算 |
| occupancy | 延迟隐藏 | 每线程工作量↓ | 高 ILP 弥补低 occupancy；但 1024 线程/block 的 4×4 配置验证了过低也会输 |
