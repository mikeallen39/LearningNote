# MoE/EP 关键技术拆解

MoE/EP 真正难的地方，不是“把专家分到多张卡上”这么简单，而是把原本规则的 dense GEMM，变成了一个稀疏路由、不规则通信、小算子、动态负载的问题。工程上，瓶颈往往已经不在 attention 或 FFN 本身，而在 token 怎么发、怎么排、怎么重叠、怎么防止热点专家把整机拖慢。

下面按 6 个核心技术点展开。

## 1. AllToAll token 分发

### 它是什么

每个 token 先经过 router，选出 top-k expert。若这些 expert 分布在不同 GPU 上，就要把 token hidden states 发到对应 rank；专家算完后，再把结果发回原 rank 做 combine。这就是 MoE 里的 dispatch / combine，本质上是一个稀疏 many-to-many 的 AllToAll。

### 难点在哪里

- 同一批 token 被路由到各 expert 的数量每次都不同，send size 是运行时才知道的。
- top-k 会复制 token，导致通信量不规则放大。
- decode 场景 batch 小、消息碎、延迟敏感；prefill 场景 batch 大、带宽敏感，两种优化目标不一致。
- 跨机时还会遇到 NVLink、PCIe、RDMA 这类异构链路，瓶颈位置并不相同。

### 为什么难

普通 collective 更适合规则 tensor；MoE dispatch 则需要先按 expert 分类，再发给不同 rank，再在对端恢复局部连续布局。这中间包含计数、prefix-sum、pack、通信、unpack 多步，任何一步设计不对，都会把后续专家计算饿住。

### 当前前沿解法

- **DeepEP**：把 dispatch / combine 单独视作 MoE 专用通信问题，分别做高吞吐和低时延路径，并支持异构域 forwarding。
- **SGLang**：将 `deepep` 集成为 MoE A2A backend，支持面向 prefill 和 decode 的不同模式。
- **Megatron-Core**：将 `alltoall` dispatcher 和更灵活的 dispatcher 作为正式能力提供。

前沿方向很明确：不是泛化 collective，而是为 MoE token dispatch 做专门的通信内核和分层调度。

## 2. permutation / unpermutation

### 它是什么

dispatch 之前或之后，要把 token 按 expert 重新排列，让属于同一个 expert 的 token 在内存里尽量连续，这叫 permutation。专家算完以后，再把输出恢复到原 token 顺序，并按 top-k 权重合并，这叫 unpermutation。

### 难点在哪里

- 这类操作通常不是算力瓶颈，而是内存搬运瓶颈。
- 它涉及 gather / scatter、index mapping、重复 token、权重融合，访存模式很碎。
- 如果 top-k 大于 1，同一个 token 会进入多个 expert bucket，恢复时还要正确加权累加。
- 在很多实现里，permute / unpermute 的时间并不比专家 GEMM 短。

### 为什么难

这类问题更像一个稀疏重排系统，而不是标准矩阵乘。很难靠单纯堆 FLOPs 解决，必须靠 layout 设计、kernel fusion、减少中间张量和中间 pass。

### 当前前沿解法

- **Megatron-Core Permute Fusion**：把 permute / unpermute 合成更少的 kernel。
- **Memory efficient token permutation**：把部分概率乘法从 unpermute 阶段往后挪，减少中间访存与额外 pass。
- 越来越多框架把 combine 逻辑尽量塞进后续算子，避免重复落地和重复读写。

领先实现普遍在做两件事：融合重排 kernel，以及把 combine 逻辑尽量延后或内联进后续计算。

## 3. 通信计算重叠

### 它是什么

理想情况下，当一部分 token 还在 dispatch / combine 通信时，GPU 已经开始做别的有用计算，比如 attention、shared experts、上一批或下一批的 MoE compute，而不是整卡等待通信结束。

### 难点在哪里

- 很多实现虽然表面异步，但通信和计算实际争的是同一批 SM、同一片 HBM、同一个 stream 优先级。
- MoE 依赖链很强：router 决定 dispatch，dispatch 完了才能 expert compute，compute 完了才能 combine，不是简单加个 async 就能隐藏延迟。
- 真正有价值的 overlap，必须建立在正确的执行时序和资源隔离上。

### 为什么难

你需要同时控制通信何时启动、哪些张量先到先算、哪些计算可以和通信真正并行、如何避免 overlap 反而让 kernel 互相抢资源。这已经接近 runtime / scheduler 问题，而不只是 kernel 问题。

### 当前前沿解法

- **DeepEP hook-based overlap**：通过 hook 机制实现通信与计算重叠，并强调不占用 SM 做无效轮询。
- **SGLang TBO / SBO**：前者更偏 batch 级流水，后者更偏单 batch 内的 hook 驱动重叠。
- **vLLM Dual Batch Overlap**：把两个 batch 的调度和执行交错，以隐藏 A2A 延迟。
- **Megatron-Core**：提供 EP communication overlap 相关能力。

当前前沿不是简单“异步化”，而是把 MoE 通信纳入整机调度，把 batch 级和单 batch 级 overlap 一起做。

## 4. grouped GEMM / 小专家优化

### 它是什么

每个 expert 分到的 token 数往往不多，尤其是 decode 或负载不均衡时，一个 expert 只吃到很小的矩阵。若对每个 expert 单独发起小 GEMM，kernel launch 和低 occupancy 会严重拖垮性能。grouped GEMM 的目标，就是把多个 expert 的小矩阵乘尽量合并成更大的执行单元。

### 难点在哪里

- 每个 expert 的 token 数量并不相同，shape 天然不规则。
- 有的 expert 吃到的 token 很少，有的很多，甚至有的为空。
- 你既希望合并执行，又不希望为了对齐做过多 padding。
- expert 权重彼此独立，不能像普通 batch GEMM 那样简单堆叠。

### 为什么难

这是典型的“理论 FLOPs 很多，但 GPU 实际很闲”的问题。MoE 的算子粒度太碎，launch overhead、warp 利用率、shared memory 使用和权重读取方式都会成为主导开销。

### 当前前沿解法

- **Megatron-Core Grouped GEMM**：把多个 expert 的小 GEMM 聚合成更少的大 kernel，提高 GPU 利用率。
- **SGLang**：在 MoE runner backend 上支持面向 expert compute 的不同后端，本质目标也是把 expert 计算做成高利用率执行。
- 前沿路线通常不是单点优化，而是三件事配合：
  - token 预重排，形成更好的连续布局；
  - grouped kernel / grouped MLP，减少碎片化执行；
  - 低精度计算，如 FP8、W8A8，进一步缓解带宽和寄存器压力。

对小专家特别有效的，通常不是某一个孤立技巧，而是重排、grouped kernel、quantization 三者一起发挥作用。

## 5. 负载均衡

### 它是什么

router 会让不同 token 选择不同 expert，但真实流量往往会让少数 expert 特别热，形成 straggler。MoE 的整体吞吐和时延通常取决于最慢 expert 或最慢 rank，所以负载均衡直接决定吞吐上限和 tail latency。

### 难点在哪里

- 训练时负载均衡，并不等于部署时自然平衡。
- 线上流量分布和训练分布可能完全不同。
- prefill 和 decode 的负载模式并不一致。
- 即使平均上看起来平衡，短时间窗口内也可能出现明显热点 expert。

### 为什么难

这是一个动态系统问题。不能只看静态参数量，还要看时间窗内 token 命中分布、top-k 共现模式、不同 batch 的偏斜程度，以及真实请求结构。很多时候，模型自身路由不均与部署放置不合理会叠加。

### 当前前沿解法

- 训练侧常见手段包括 auxiliary loss、capacity factor 等。
- 推理侧越来越多框架开始做运行时负载均衡。
- **vLLM EPLB**：根据运行时统计结果调整 expert 布局，并支持冗余专家。
- **SGLang EPLB**：也提供运行时专家布局优化能力。
- **redundant experts / expert replication**：对热点专家做副本，避免单点过热。

现在比较先进的思路不是强行改 router，而是训练期平衡、推理期重排或复制、周期性 rebalance 一起做。

## 6. 拓扑感知调度

### 它是什么

不是所有 GPU 间通信都一样快。单机内可能有 NVLink 域，跨机可能走 RDMA，不同路径带宽和时延差异很大。拓扑感知调度就是让 EP 和 token dispatch 尽量顺着“便宜链路”走，把最频繁的 token 交换限制在高速域内。

### 难点在哪里

- MoE 的通信模式是由 routing 动态决定的，而硬件拓扑是静态异构图。
- 如果 expert placement 不考虑拓扑，就可能出现热点 expert 总在跨域接收 token 的情况。
- prefill 更看重吞吐，decode 更看重时延，最优路径不一定一致。

### 为什么难

这不是单层优化，而是 placement、group 划分、runtime 调度、通信库能力一起耦合的结果。需要同时决定：

- expert 放在哪些 GPU；
- EP group 如何划分；
- prefilling 与 decoding 分别走什么路径；
- 是否先域内聚合再域间发；
- 热点 expert 是否需要复制到多个域。

这些决策相互耦合，而且会随着 batch size 和流量模式变化。

### 当前前沿解法

- **DeepEP**：明确支持异构域优化，例如面向 `NVLink -> RDMA` 的 forwarding，以及面向 decode 的低时延纯 RDMA kernel。
- 实际最佳实践通常是：
  - 尽量把 EP 限制在单机高速互联域内；
  - 做分层 dispatch / combine，优先域内再域间；
  - prefill 使用高吞吐策略，decode 使用低时延策略；
  - 热点 expert 优先放在流量中心，必要时复制。

所以拓扑感知调度的前沿不是某一个单独 kernel，而是通信库、placement 和 runtime scheduler 的协同设计。

## 总结

这 6 个点里，真正决定 MoE/EP 上限的通常不是某一个 kernel，而是下面这条链是否闭环：

- 路由足够平衡；
- token 重排足够省；
- dispatch / combine 足够快；
- 小专家 GEMM 足够高效；
- 通信和计算能真正重叠；
- expert placement 符合硬件拓扑。

现在开源前沿已经越来越收敛到一个共识：MoE 推理优化本质上是“通信系统 + runtime 调度 + kernel fusion”的联合问题，不再只是“写一个更快的 GEMM”。

## 参考资料

- DeepEP
  - https://github.com/deepseek-ai/DeepEP
- Megatron-Core MoE
  - https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/moe.html
- vLLM Expert Parallel Deployment
  - https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- SGLang Expert Parallelism
  - https://docs.sglang.io/advanced_features/expert_parallelism.html
