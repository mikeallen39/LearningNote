# Megatron Core MoE 官方文档详细讲解

这篇笔记基于 NVIDIA Megatron Core 最新官方文档：

- Mixture of Experts
  - https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html

写作时间：`2026-03-27`

这篇文档非常重要，但也有一个需要先讲清楚的前提：

**它的主体是 MoE 训练框架文档，不是推理框架文档。**

所以里面的大部分设计、参数和优化，默认语境是：

- 大规模 MoE 训练
- 多维并行
- 显存、通信、吞吐的联合优化

不过，即使你更关心推理，这篇文档仍然非常有价值，因为它系统梳理了：

- MoE 的并行组织方式；
- token dispatch 路径；
- EP / TP / PP / CP 的取舍；
- MoE 的核心性能瓶颈；
- 以及很多后来被推理系统继承的优化思想。

这篇笔记的目标不是逐段翻译，而是：

- 按文档结构完整梳理内容；
- 解释每一块到底在说什么；
- 补足工程语境；
- 帮你判断哪些内容对推理也有启发。

## 1. 这篇官方文档整体在讲什么

Megatron Core 给 MoE 的定位非常明确：

**它是一个 production-ready 的大规模 MoE 训练框架。**

官方文档围绕 6 个主题展开：

1. 最新版本新增了什么
2. 这个 MoE 栈支持哪些模型和能力
3. 如何快速启动一个 MoE 训练
4. 如何系统地做并行与性能选型
5. 关键特性如何配置
6. 常用参数都有哪些

如果把它压缩成一句话，它真正想表达的是：

**Megatron Core 不是只提供一个“MoE 层”，而是提供一整套面向大规模 MoE 训练的并行、通信、内存、算子和工程实践。**

## 2. What’s New：最近几个版本到底新增了什么

这一节非常重要，因为它能看出 Megatron Core 团队最近在押注什么方向。

### 2.1 dev 分支（2026-01）

文档写到开发分支最近增加了：

- pipeline-aware 的细粒度 activation offloading
- `Qwen3-Next` 支持
- `DeepSeek-V3.2` 支持
- Muon 和 layer-wise distributed optimizer
- 更细粒度 scope 的 CUDA Graph 支持

这说明它的演进方向非常清楚：

- 一方面继续扩模型覆盖；
- 一方面继续压榨训练系统的内存与调度开销；
- 一方面往更复杂、更细粒度的并行运行时走。

### 2.2 v0.15（2025-11）

主要新增：

- Flex Dispatcher 支持 `HybridEP` backend
- 支持 `FSDP + EP` 组合

这说明 Megatron Core 已经不满足于“单一 dispatcher + 单一并行形态”，而是在尝试：

- 更灵活的 expert dispatch backend
- 更复杂的参数分片与专家并行组合

### 2.3 v0.14（2025-09）

这一版很关键，主要包括：

- 用 batch-level overlap 隐藏 `EP A2A` 通信
- fine-grained recomputation 支持 FP8
- MoE router fusion kernel
- MTP 和 MLA 的 CP 支持

这几个点很能说明问题：

- MoE 的通信和小算子已经成为主战场；
- 不是只写快一点的 GEMM，而是要把 router、dispatch、recompute、CP 一起纳入系统设计；
- MTP / MLA 这些“更现代”的结构开始进入主线支持。

### 2.4 v0.13（2025-07）

主要包括：

- optimizer state 支持 BF16
- 更灵活的虚拟流水并行布局
- Hybrid Shard Data Parallel 支持
- 细粒度 recomputation
- 更省内存的 token permutation 设计

这里面最值得你注意的是最后一点：

**Megatron Core 直接把 token permutation 当成一个独立优化问题来处理。**

这其实非常前沿，因为今天 MoE 的性能瓶颈很多时候已经不是“专家算得慢”，而是：

- token 怎么排
- 怎么发
- 怎么恢复
- 中间张量怎么少落地

### 2.5 v0.12（2025-05）

主要包括：

- DeepEP 支持
- MTP 支持
- dropless MoE 的 CUDA Graph 支持

这几乎是今天很多现代 MoE 系统的主线组合：

- 更高效的 dispatcher
- 更复杂的模型结构
- 更低 runtime 开销

## 3. Overview：Megatron Core MoE 到底支持什么

这一节本质上是一个能力总表。

### 3.1 模型支持

官方列出的重点模型包括：

- DeepSeek-V2
- DeepSeek-V3（含 MTP）
- Qwen2-57B-A14B
- Qwen3-30B-A3B
- Qwen3-235B-A22B
- Mixtral-8x7B
- Mixtral-8x22B

这说明它主要瞄准的是今天最典型的开源 MoE 路线：

- Mixtral 系
- DeepSeek 系
- Qwen 系

### 3.2 核心 MoE 功能

官方列出的核心能力包括：

- dropless MoE
- Top-K router
- load balancing losses

这里的重点是：

**Megatron Core 默认把“无丢 token 的高性能 MoE”当成核心能力，而不是一种附加实验功能。**

### 3.3 高级并行

官方明确支持：

- `EP + DP + TP + PP + SP`
- `CP`
- `Parallel Folding`
- distributed optimizer

这说明它不是单维并行框架，而是一个多维并行组合框架。

### 3.4 性能优化

这一节列出来的优化项非常值得注意：

- memory-efficient token permutation
- fine-grained recomputation
- MLA TP support
- GroupedGEMM
- DP/PP/TP communication overlap
- shared expert overlap
- router fusion
- token permutation/unpermutation fusion
- cuDNN fused attention

这份列表本身就已经在告诉你：

**MoE 的真实瓶颈早就不是“写一个快点的 expert MLP”这么简单。**

官方已经把问题拆成：

- 通信
- 重排
- overlap
- 小算子融合
- activation memory
- attention backend

### 3.5 硬件与精度支持

文档特别提到：

- DeepEP 对 H100 / B200 的支持
- GroupedGEMM 包括 FP8 / MXFP8 支持
- FP8 训练完整支持

这说明它在 MoE 上的设计，已经明显偏向 Hopper / Blackwell 时代的训练栈。

### 3.6 开发体验

包括：

- MoE model zoo
- distributed checkpointing
- upcycling
- MCore2HF converter
- per-layer logging

这一块说明 Megatron Core 并不是“只有内核和并行”，它也在补齐模型迁移、检查点和生态互通。

## 4. Quick Start：官方建议你怎么快速起一个 MoE

Quick Start 分成 3 块：

1. 最小可训练配置
2. 预定义模型配置
3. 通用性能建议

### 4.1 最小训练配置

官方给的最小示例，本质上就是一套最核心参数：

- 设置专家数
- 设置 shared expert hidden size
- 设置 router 类型与 top-k
- 设置 aux loss
- 设置 token dispatcher

最核心的几个参数是：

- `--num-experts`
- `--moe-shared-expert-intermediate-size`
- `--moe-router-load-balancing-type`
- `--moe-router-topk`
- `--moe-aux-loss-coeff`
- `--moe-token-dispatcher-type`

这套最小例子其实在表达一个很重要的设计：

**Megatron Core 里，MoE 结构主要是通过脚本参数定义的，而不是你手写一大堆专门模型代码。**

### 4.2 预定义配置

官方说已经提供了多种流行 MoE 模型的预定义配置，可直接参考：

- Mixtral 8x7B
- Mixtral 8x22B
- DeepSeek-V3
- Qwen3-30B-A3B
- Qwen3-235B-A22B

这个信息很重要，因为实际工程里：

- “能训起来”和“训得快”差很多；
- 预定义 config 往往比你自己从零拼更可靠。

### 4.3 通用性能建议

官方推荐在几乎所有 workload 上优先打开的一些功能：

- `flex + deepep`
- `--moe-grouped-gemm`
- `--moe-router-fusion`
- `--moe-permute-fusion`
- cross entropy fusion
- distributed optimizer
- overlap-grad-reduce
- overlap-param-gather
- tp-comm-overlap
- manual gc

这一段特别值得注意，因为它本质上已经给出了一套“现代 MoE 高性能训练默认组合”：

- 好 dispatcher
- 好 expert kernel
- 好 fusion
- 好 overlap
- 好 runtime 管理

如果你后面在看别的系统，例如 SGLang、vLLM、DeepEP、TensorRT-LLM，会发现思路高度相通。

## 5. Best Practices：官方的并行选型方法论

这一节是整篇文档里最值得反复看的部分之一。

它不是零散堆参数，而是在教你如何做**系统性的并行选型**。

### 5.1 Step 1：先找显存能承受的可行并行映射

官方把显存消耗拆成三部分：

- activation memory
- weight / gradient memory
- optimizer states memory

然后对不同并行方式给出一个粗略比较：

- `TP`：activation、weight、optimizer 都能按 `1/N` 切，但每层通信高
- `EP`：激活内存不一定显著下降，MoE layer 的权重按 `1/N` 切，通信中等
- `PP`：权重和 optimizer 按 `1/N` 切，但 activation 侧不一定等比例省
- `CP`：activation 能按 `1/N` 切，但权重不切
- `DP`：activation 和权重都不切，但 optimizer 可通过 distributed optimizer 降低

这张表最重要的结论不是数字本身，而是：

**不同并行方式省的是不同类型的内存，付出的通信代价也完全不同。**

官方还提供了 `--fake-init-process-group`，允许你在单卡上模拟分布式初始化，用来寻找显存允许的可行映射。

这是一个非常工程化的功能，说明他们很清楚：

真正难的不是写并行名字，而是先知道哪些组合根本不会 OOM。

### 5.2 Step 2：如何选“最优并行策略”

这里官方给了 5 条明确指南。

#### Guideline 1：尽量减少模型并行，尽量放大数据并行

核心思想：

- TP / EP / PP 都会引入通信；
- DP 通常更容易给吞吐；
- 所以只要不 OOM，就尽量把模型并行度压低。

这是 Megatron Core 的一个重要思想：

**模型并行不是越大越好，而是“刚好够放下 + 再把剩余空间还给 DP”。**

#### Guideline 2：EP 和 TP 尽量留在 NVLink 域内

官方明确建议：

- 让 `EP × TP` 尽量限制在单机内
- 不要轻易把高通信并行扩到跨节点

原因非常直接：

- EP 和 TP 都是高通信并行
- NVLink 明显比跨节点互联更强

这条建议其实非常值得迁移到推理里：

**高频 collective 和 A2A，尽量留在最高带宽域内。**

#### Guideline 3：跨节点扩展优先考虑 PP

官方明确说：

- 当需要跨节点扩展时，优先考虑 PP，而不是继续把 TP/EP 扩到跨节点

这点非常重要。它其实也在呼应我们前面聊过的结论：

**PP 的点对点通信往往比跨节点的大规模 `all_reduce` / `all_to_all` 更容易管理。**

#### Guideline 4：MoE expert 层优先 EP，而不是 TP

这是整篇文档里最关键的一句之一。

官方给的理由包括：

- 更好的 GEMM 效率
- 更低的通信
- 更简单的计算图
- 更容易做通信计算重叠
- 当 `EP = num_experts` 时，本地 token permutation 可以消除

官方还直接给了例子：

- Mixtral 8x7B 中，`EP8 × TP1` 比 `EP4 × TP2` 更好

这基本就是：

**Megatron Core 官方明确站队“对 expert 层优先 EP，不要轻易上 TP”。**

#### Guideline 5：长序列时启用 CP

官方建议：

- 当序列长度达到 `8K` 及以上时，考虑打开 CP

这说明他们对 CP 的定位非常明确：

- 不是默认总开；
- 是针对长序列的重要工具。

### 5.3 Step 3：根据 profiling 的瓶颈有针对性地开优化

官方把瓶颈分成 4 类：

- memory bottleneck
- communication bottleneck
- CPU overhead bottleneck
- computation bottleneck

这其实就是一个非常成熟的系统优化框架。

#### Memory Bottleneck

如果你因为 OOM 被迫把并行拉得很大，官方建议优先尝试：

- selective recomputation
- activation offloading
- optimizer offloading

#### Communication Bottleneck

如果 profiling 里 collective 太重，官方建议按通信类型开 overlap：

- DP gradient reduce：`--overlap-grad-reduce`
- DP param gather：`--overlap-param-gather`
- TP comm：`--tp-comm-overlap`
- EP A2A：`--overlap-moe-expert-parallel-comm --delay-wgrad-compute`
- PP send/recv：通过 VPP 增强 overlap

这说明官方是明确把通信看成分 parallel dimension 管理的，而不是一个总开关。

#### CPU Overhead Bottleneck

如果 Nsight 上看到 GPU kernel 之间有明显 gap，说明 CPU launch 不够快。

官方建议：

- 手动 GC
- CUDA Graph
- 减少 kernel 数

这非常像今天很多推理系统在做的事情。

#### Computation Bottleneck

如果没有明显通信或 CPU 瓶颈，但 GPU 利用率仍然不高，说明你算子太碎或效率低。

官方建议：

- 开 fusion
- 用 FP8

这和 MoE 的现实完全一致：

**很多 fine-grained MoE 的问题，不是 FLOPs 少，而是 kernel 太碎。**

## 6. Feature Documentation：官方把哪些能力作为“核心特性”

这一部分进入了更细的功能层。

### 6.1 Router / Load Balancing

文档支持多种 router 负载均衡方式：

- `aux_loss`
- `sinkhorn`
- `seq_aux_loss`
- `none`
- 以及新的 expert bias 路线

这里最值得关注的不是参数名，而是官方已经把负载均衡看成可切换策略，而不是固定实现。

这说明：

**router 不只是选 top-k，它还是整体性能与稳定性的控制点。**

### 6.2 Token Dispatching

这部分非常关键。

官方把 dispatcher 分成几类：

### alltoall

- 基于 NCCL 的 All-to-All token exchange
- 适合标准 `EP > 1`

这是最传统、最标准的 EP 路线。

### flex + DeepEP

官方写得很清楚：

- 会移除跨节点通信里的冗余 token
- 融合节点内和节点间通信
- 特别适合 cross-node EP 和 fine-grained MoE，例如 DeepSeek-V3

这说明 Megatron Core 已经承认：

**标准 A2A 不够了，dispatcher 本身就是优化重点。**

### flex + HybridEP

官方把它描述成：

- 基于 TMA 和 IBGDA
- 占用更少 SM
- 原生支持 MNNVL

这说明他们在继续把 dispatcher 往更底层硬件能力上压。

### allgather

官方也保留了 `allgather` dispatcher，用于：

- TP-only setup
- 小 EP
- 大 top-k

这点很重要，因为它说明：

**Megatron Core 并没有把所有场景都一刀切成 A2A。**

### 6.3 MoE Parallel Folding

这是整篇文档里理论与工程结合最强的一部分之一。

它解决的问题是：

- 传统框架常限制 `EP ≤ DP`
- attention 和 MoE layer 被迫共享同一套 TP/CP 结构
- 结果导致两边都不是最优

Parallel Folding 的核心思想是：

- attention 层用一套并行映射
- MoE 层用另一套并行映射

文档里给出的形式是：

- attention：`TP × CP × DP × PP`
- MoE：`ETP × EP × EDP × PP`

这里最重要的工程意义是：

**attention 和 MoE layer 的最优并行方式本来就不一样，不应该强行绑死。**

官方列出的收益包括：

1. 打破 `EP ≤ DP` 的限制
2. 降低最小 GPU 需求
3. attention 和 MoE 可以独立优化
4. 让高带宽通信留在 NVLink 域内

这实际上就是一个异构并行映射框架。

如果你理解了这点，就会更容易理解：

- 为什么 MoE 不应该默认用大 TP
- 为什么 attention 和 expert 要分开看
- 为什么 CP 对 attention 很重要，但对 expert 不一定有意义

## 7. Memory Optimization：为什么 Megatron Core 特别重视“省内存”

官方强调一个核心现实：

**MoE 即使每个 token 只激活少量 experts，模型仍然要维护全部 expert 参数。**

所以内存问题不会因为“稀疏激活”自动消失。

文档给出的主要方向包括：

### 7.1 Fine-grained Recomputation

不是整层重算，而是只重算某些模块，例如：

- `mla_up_proj`
- `layernorm`
- `moe_act`

这比全量 activation checkpoint 更细，也更灵活。

### 7.2 Fine-grained Activation Offloading

它和 recomputation 的 tradeoff 不一样：

- recomputation：拿算力换显存
- offloading：拿 GPU-CPU 带宽换显存

官方强调要点在于：

- 用异步 D2H / H2D
- 尽量把传输延迟藏在计算后面
- 支持和 PP/VPP 配合

这说明他们不是简单“把张量搬到 CPU”，而是认真把 offload 当作 pipeline 设计问题来做。

### 7.3 Precision-aware Optimizer

把优化器状态从 FP32 压到 BF16，可减少一半 optimizer memory。

### 7.4 Optimizer Offloading

进一步把 optimizer states 放到 CPU。

这些东西虽然是训练语境，但你从中能看出官方的整体思路：

**MoE 的显存优化不是一个手段，而是一整套层次化方案。**

## 8. Communication Optimization：Megatron Core 如何看待通信优化

官方明确说：

**分布式训练会引入来自不同并行维度的通信开销，而正确思路是把通信和计算重叠。**

这部分分成四类。

### 8.1 DP Communication Overlap

在 distributed optimizer 下，DP 会引入：

- gradient 的 `reduce-scatter`
- parameter 的 `all-gather`

官方支持：

- `--overlap-grad-reduce`
- `--overlap-param-gather`

### 8.2 TP Communication Overlap

官方写得很清楚：

- TP + SP 会引入 activation `all-gather` 和 `reduce-scatter`
- 可以通过 bulk 或 pipelined 方式做 overlap

启用条件：

- `tensor_model_parallel_size >= 2`
- `--sequence-parallel`

这说明 TP overlap 不是免费午餐，而是建立在 SP 与依赖关系管理上的。

### 8.3 PP Communication Overlap

PP 的通信是 stage 间的 P2P send/recv。

官方指出：

- 开启 VPP 后，1F1B 阶段的 overlap 会自动变得更强

这和他们前面的“多机优先考虑 PP”形成了呼应。

### 8.4 EP Communication Overlap

官方明确指出：

**如果不优化，EP 的 All-to-All 可能吃掉 30% 到 40% 的训练时间。**

这是一个非常值得记住的数字。

他们提供两类能力：

- `EP A2A Overlap`
- `Shared Expert Overlap`

前者通过相邻 microbatch 的前后向融合来藏 A2A；后者让 shared expert 计算和 token transfer 并发。

这说明：

**Megatron Core 已经把 MoE 通信优化看成一等公民，而不是附加特性。**

## 9. Compute Optimization：为什么它会特别强调 GroupedGEMM 和 Fusion

官方这里的逻辑非常直接：

**fine-grained MoE 会产生大量小算子，导致 GPU 利用率低。**

因此需要：

### 9.1 Grouped GEMM

把多个 expert GEMM 批处理到一次 kernel 调用中。

核心收益：

- 减少 launch overhead
- 提高 GPU 利用率

### 9.2 Router Fusion

把 router projection、top-k、softmax、aux loss 等融合起来。

### 9.3 Permute Fusion

把 token permutation / unpermutation 融成更少 kernel。

这是非常典型的现代 MoE 优化思路：

- router 自己单独优化
- permutation 自己单独优化
- expert GEMM 自己单独优化

也就是说，官方已经完全承认：

**MoE 的时间不会只花在 MLP 上。**

## 10. FP8 Training：官方怎么看 FP8 在 MoE 里的价值

Megatron Core 对 FP8 的论述非常系统。

它说 FP8 会同时改善三个“性能墙”：

- memory
- communication
- compute

### 10.1 内存收益

- activation 可减半
- 可以消除 BF16 权重副本

### 10.2 通信收益

- EP dispatch 体积减半
- parameter all-gather 体积减半

### 10.3 计算收益

- Hopper / Blackwell 上 FP8 Tensor Core 更快

官方还进一步区分了三种 recipe：

- per-tensor
- blockwise
- MXFP8

并明确推荐：

**Hopper 上生产训练优先使用 blockwise FP8。**

这个判断非常关键，因为它说明官方不是只支持 FP8，而是已经形成了“生产推荐路径”。

此外，官方还提供了两类 MoE-specific FP8 优化：

- routing map padding
- FP8 primary weights

这进一步说明：

**Megatron Core 的 FP8 不是通用训练附带支持，而是已经深入到了 MoE 细节。**

## 11. CUDA Graph：官方怎么把图捕获引入 MoE

文档提到：

- 支持 `local` 和 `transformer_engine` 两种 graph 实现
- 可以通过 `--cuda-graph-scope` 控制 scope

在前面的 CPU overhead 章节里也提到：

- 可以只 graph 某些 scope，例如 `attn`、`moe_router`、`moe_preprocess`

这说明它不是要求“一次 graph 整个模型”，而是倾向于：

**对合适的局部子路径分 scope 捕获，以减少 CPU launch jitter。**

这和推理系统里常见的 partial graph capture 思路非常像。

## 12. Arguments Reference：这部分参数该怎么读

官方在后半部分给了完整参数分类。

这里最重要的不是记住每个 flag，而是建立“参数分层”的理解。

### 12.1 Core Arguments：定义 MoE 结构

典型参数包括：

- `--num-experts`
- `--expert-model-parallel-size`
- `--moe-ffn-hidden-size`
- `--expert-tensor-parallel-size`
- `--moe-layer-freq`

其中一个非常关键的点是：

官方明确写到：

**`expert-tensor-parallel-size` 推荐在 fine-grained MoE 中设为 1。**

这和前面“expert 优先 EP 而不是 TP”的建议完全一致。

### 12.2 Router Arguments：定义路由行为

包括：

- `--moe-router-load-balancing-type`
- `--moe-router-topk`
- `--moe-router-score-function`
- `--moe-router-pre-softmax`
- `--moe-router-num-groups`
- `--moe-router-group-topk`

从这些参数你就能看出，Megatron Core 的 router 已经很灵活：

- top-k 可配
- score function 可配
- pre-softmax 可配
- group-limited routing 也可配

### 12.3 Loss and Regularization

主要包括：

- `--moe-aux-loss-coeff`
- `--moe-z-loss-coeff`
- `--moe-input-jitter-eps`

也就是说，路由稳定性和负载均衡，不只是结构问题，也是损失设计问题。

### 12.4 Token Dispatching

这一类参数直接控制 token 如何移动：

- `--moe-token-dispatcher-type`
- `--moe-enable-deepep`
- `--moe-expert-capacity-factor`
- `--moe-pad-expert-input-to-capacity`
- `--moe-token-drop-policy`
- `--moe-permute-fusion`

这组参数其实已经构成了一个“小系统”：

- dispatcher 类型
- capacity 控制
- 是否 pad
- drop 策略
- permutation 是否融合

### 12.5 Performance Optimization

包括：

- `--moe-grouped-gemm`
- `--overlap-moe-expert-parallel-comm`
- `--delay-wgrad-compute`
- `--moe-shared-expert-intermediate-size`
- `--moe-shared-expert-overlap`

这里最值得注意的是：

shared expert 已经被当成和 routed expert 不同的一类对象来单独优化。

### 12.6 Memory and Checkpointing

包括：

- `--moe-layer-recompute`
- `--moe-use-upcycling`
- `--moe-upcycling-granularity`

### Upcycling 是什么

它的核心思想是：

**从 dense 模型出发，逐步迁移或扩展到 MoE 模型。**

这个能力对于模型扩容和迁移非常有价值。

### 12.7 Miscellaneous

包括：

- `--moe-per-layer-logging`
- `--moe-router-force-load-balancing`

前者适合诊断，后者偏实验性。

## 13. Examples：官方示例脚本说明了什么

文档最后给了 Mixtral 8x7B 的完整训练脚本示例。

这段示例的重点不是具体数值，而是你能看出官方推荐的组合：

- `EP=8`
- `TP=1`
- `PP=4`
- 开启 grouped gemm
- 开启 permute fusion
- 用 `alltoall` dispatcher
- 开 overlap-grad-reduce / overlap-param-gather
- 用 distributed optimizer
- 开 sequence parallel

这套组合再次说明一个极其关键的工程结论：

**在 Megatron Core 的官方实践里，MoE 模型并不是默认要靠大 TP 来跑。**

更常见的路线是：

- experts 用 EP
- attention 相关并行控制在合理范围
- 再配合 PP、SP、distributed optimizer 和各种 overlap

## 14. 我对这篇官方文档的整体理解

如果把整篇文档抽象成几个核心判断，我会总结成下面 8 点。

### 14.1 它不是“MoE 层使用说明”，而是“MoE 训练系统使用说明”

这篇文档真正讲的是：

- 并行映射
- dispatcher
- overlap
- memory
- router
- FP8
- runtime

而不是只讲某个算子。

### 14.2 官方已经明确站在“expert 优先 EP”这一边

这一点非常明确，而且是反复出现的：

- expert layer 优先 EP
- `expert-tensor-parallel-size` 推荐设为 1
- Mixtral 示例也是 `EP8 × TP1`

### 14.3 MoE 的优化重点已经从“单点算子”转向“系统协同”

官方最强调的东西包括：

- dispatcher
- permutation
- overlap
- grouped gemm
- router fusion
- shared expert overlap

这说明 MoE 的性能问题本质上是系统问题。

### 14.4 通信不是一个整体，而是按并行维度拆开管

文档把通信明确拆成：

- DP overlap
- TP overlap
- PP overlap
- EP overlap

这非常成熟，也很实用。

### 14.5 Parallel Folding 是这篇文档最有理论价值的部分之一

因为它真正承认了：

**attention 和 MoE layer 的最优并行方式天然不同。**

这点对理解今天很多推理系统也很有帮助。

### 14.6 FP8 在它这里已经不是“实验功能”，而是主线优化

尤其是：

- blockwise FP8
- routing map padding
- FP8 dispatch

这些都说明它已经进入生产级路线。

### 14.7 文档虽然面向训练，但对推理也很有启发

例如：

- expert 优先 EP 而不是 TP
- token permutation 是独立瓶颈
- dispatcher 是核心模块
- overlap 必须按 parallel dimension 设计

这些结论在推理里同样成立。

### 14.8 这篇文档最重要的工程思想

如果只记一句话，我会记：

**MoE 的最优实现不是“开更多并行度”，而是先找到可行显存映射，再按瓶颈逐维度打开合适的并行与优化。**

## 15. 对推理读者来说，最值得提炼的内容

如果你更关心推理，而不是训练，这篇文档最值得带走的其实是下面这些判断：

1. expert 层通常优先 EP，而不是大 TP。
2. token dispatch / permutation / combine 是独立瓶颈。
3. TP、EP、PP、CP 不该混成一团看，应该分通信维度看。
4. shared expert 和 routed expert 应该分开理解。
5. overlap 必须分 `DP / TP / PP / EP` 来做。
6. 长序列问题和 expert 问题不是同一个问题，CP 和 EP 的价值不同。
7. MLA / MTP 这类新结构会改变最优并行映射。
8. 真正成熟的系统一定会把 memory、communication、compute、CPU overhead 一起纳入考虑。

## 参考资料

- Megatron Core MoE 官方文档
  - https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
