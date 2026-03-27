# 推理并行方式与通信开销总览

这篇笔记的目标是把两个问题放在一起回答清楚：

1. **当前推理里常用的并行方式有哪些，它们各自在切什么？**
2. **这些并行方式分别会用到哪些通信算子，不同通信算子的开销怎么比较？**

为了避免概念混乱，先强调一个总原则：

**并行方式决定“切什么”，通信算子决定“切开以后怎么重新协同”。**

例如：

- `TP` 决定按层内张量切；
- `PP` 决定按层深切；
- `EP` 决定按 experts 切；
- `CP / DCP` 决定按序列或 KV cache 切；
- 真正的代价则落在 `all_reduce / all_gather / reduce_scatter / all_to_all / send-recv` 这些通信原语上。

## 一句话结论

如果只看推理主路径，今天最常见的几种并行方式可以粗略分成：

- **DP / 请求并行**：主路径通信最少；
- **TP**：主路径通信最频繁，最容易被 `all_reduce` 主导；
- **PP**：主路径更偏 `send/recv`，通常比跨节点大 TP 更友好；
- **EP**：主路径更偏 `all_to_all`，MoE 性能核心在 dispatch / combine；
- **CP / DCP**：主路径通信形态取决于 attention 变体和实现，常见是 `all_gather` 或 `send/recv` 风格的 KV 交换。

如果只比较通信算子，在**同等总数据量、拓扑良好、实现成熟**的前提下，可以先记一个非常粗略但很有用的经验顺序：

`send/recv < broadcast ≈ reduce < all_gather ≈ reduce_scatter < all_reduce < all_to_all / all_to_allv`

但必须强调：

- 这不是绝对排序；
- 小消息更看时延，大消息更看带宽；
- `all_to_all` 在 MoE 中通常还叠加 permutation / pack / unpack，所以实际体感往往比公式更“贵”；
- `all_reduce` 虽然在理论带宽上不一定最差，但在 TP 中因为**每层都是强同步点**，所以常常最显眼。

## 1. 当前推理中常用的并行方式

### 1.1 DP：数据并行 / 请求并行

### 它在切什么

DP（Data Parallelism）在推理里最常见的形态，是：

- 每个 rank 持有完整模型副本；
- 不同请求或不同 batch 分片落到不同 rank 上。

这类并行本质上是在切：

- 请求集合；
- batch；
- 服务副本吞吐。

### 通信算子

在最理想、最典型的推理路径里：

- **主路径几乎没有模型并行通信**

常见的只是：

- 调度层同步；
- 健康检查 / 控制面通信；
- 某些框架里的 load balancing / metadata 同步；
- MoE 场景下跨 DP rank 的 expert 对齐或 dummy forward 协调。

### 开销特点

- 单请求 latency 往往最稳定；
- 吞吐扩展最自然；
- 代价主要是**权重复制**和**KV cache 复制**，不是主路径 collective。

### 什么时候常用

- 模型能放下；
- 在线 serving；
- 高吞吐；
- 希望避免大 TP 通信。

### 1.2 TP：张量并行

### 它在切什么

TP（Tensor Parallelism）是把单层线性层或 attention 内部张量按 hidden 维、输出维、输入维切开。

它本质上切的是：

- 层内矩阵；
- 每张卡负责的局部 GEMM。

### 通信算子

TP 里最常见的通信算子是：

- `all_reduce`
- `all_gather`
- `reduce_scatter`

具体模式通常是：

- column/row parallel 线性层之后做 `all_reduce`
- 恢复完整 hidden 或 logits 时做 `all_gather`
- 一些实现里用 `reduce_scatter + all_gather` 组织 shard 流

如果结合 sequence parallel，还会显式引入：

- activation `all_gather`
- activation `reduce_scatter`

### 开销特点

- **主路径通信最频繁**
- decode 场景下特别容易被 `all_reduce` 主导
- TP 越大，局部 GEMM 越碎
- 对 MLA / MQA / GQA 模型，KV cache 还可能 duplication

所以 TP 的典型特征不是“单次通信一定最大”，而是：

**每层几乎都要同步。**

### 什么时候常用

- 模型或层太大，单卡放不下；
- 单机内有很强的高速互联；
- dense 模型；
- prefill 为主，且 batch 足够大。

### 1.3 PP：流水并行

### 它在切什么

PP（Pipeline Parallelism）按层深切模型，把不同 layer block 放到不同 GPU 或节点上。

它本质上切的是：

- 网络深度；
- layer stage。

### 通信算子

PP 的主路径通信通常不是 collective，而是：

- `send`
- `recv`
- 或等价的 point-to-point activation transfer

也可以理解成：

- 阶段间传 hidden states / residual states

### 开销特点

- 通信模式更简单；
- 通常比跨节点大 TP 更友好；
- 更依赖流水调度和气泡控制；
- 单请求路径会更长，但跨节点通信压力常小于 TP 的强同步 collective。

vLLM 官方博客甚至明确提到：相比 TP 的昂贵 `all-reduce`，PP 只需要更便宜的 point-to-point 通信。

### 什么时候常用

- 模型太大，单节点放不下；
- 节点之间互联不强；
- GPU 数量不能整齐地切 TP；
- 多机部署。

### 1.4 EP：专家并行

### 它在切什么

EP（Expert Parallelism）只对 MoE expert 层生效。

它切的是：

- routed experts；
- expert 权重；
- token 到 expert 的执行位置。

attention 并不会因为开了 EP 就自动一起切。

### 通信算子

EP 里最典型的通信是：

- `all_to_all`
- 或 `all_to_allv`

很多框架里还会出现：

- `allgather + reducescatter` 形式模拟 all-to-all
- naive `broadcast` 后备路径

除了通信原语本身，还一定伴随：

- token permutation
- pack / unpack
- combine / unpermute

### 开销特点

- 核心代价是 token dispatch / combine；
- 更容易受负载均衡影响；
- 更容易受拓扑影响；
- 在 MoE 里往往是吞吐上限关键瓶颈。

和 TP 最大的不同是：

- TP 的通信是“每层同步”
- EP 的通信是“路由 token 到专家”

### 什么时候常用

- MoE 模型；
- expert 参数太大；
- 希望把 expert 负载分散到更多 GPU；
- 愿意引入 A2A 和更复杂调度来换吞吐。

### 1.5 CP：上下文并行 / 长序列并行

### 它在切什么

CP（Context Parallelism）主要在长上下文场景下，把序列长度维切开。

它切的是：

- sequence length；
- query/key/value 的 token 维。

### 通信算子

CP 的通信模式不是唯一的，和具体 attention 路径有关。

常见两类：

1. **partial query, full key/value**
   - 先把分布在各 GPU 上的 KV 聚合回来
   - 常见是 `all_gather`

2. **partial query, partial key/value**
   - 每张卡只持有一部分 KV
   - 通过 ring-attention 之类的方式 chunk-by-chunk 交换 KV
   - 常见是 `send/recv` 风格的 ring 通信

### 开销特点

- 更适合长 prefill；
- 目标通常是降低 TTFT 或降低长序列激活压力；
- full-KV 路径更像大 `all_gather`
- ring 路径更像多轮 `send/recv`

### 什么时候常用

- 超长上下文；
- TP 不适合继续增大；
- 需要切 sequence 而不是切 hidden。

### 1.6 DCP：Decode Context Parallel

### 它在切什么

DCP（Decode Context Parallel）是专门面向 decode 的上下文并行。

它切的是：

- decode 阶段 KV cache 的 token / context 维。

本质目标是：

- 减少 KV cache duplication；
- 让更多请求同时驻留在 KV cache 中；
- 在 MLA / GQA / 少 kv-head 模型上缓解大 TP 带来的 duplication。

### 通信算子

DCP 的通信不是固定单一 primitive，而是 attention backend 相关。

从机制上看，常见会涉及：

- attention 阶段的 KV 交换；
- backend-specific `send/recv`；
- 或局部 `all_gather` / ring 传递。

更准确地说：

**DCP 的主成本不是一个标准名字的 collective，而是“为了跨 rank 读取分片 KV cache 的 attention 通信”。**

### 开销特点

- 它用更多通信换更少 KV duplication；
- 更偏 decode throughput / 容量优化；
- 并不是“无代价地把 KV 切开”。

### 什么时候常用

- MLA / GQA / kv-head 很少；
- `tp_size > kv_heads`；
- decode 阶段 KV cache 是主瓶颈。

### 1.7 SP：序列并行

### 它在切什么

SP（Sequence Parallelism）通常作为 TP 的辅助优化出现。

它主要切的是：

- sequence dimension 上的一些轻算子；
- 例如 LayerNorm、Dropout 等激活路径。

### 通信算子

常见是：

- `all_gather`
- `reduce_scatter`

### 开销特点

- 常用于缓解 TP 带来的 activation memory；
- 一般不是单独讨论的主并行方式；
- 更像 TP 的配套手段。

### 什么时候常用

- TP 已开启；
- hidden 很大；
- 希望降低 activation memory；
- TP + EP 组合时常被要求一起启用。

## 2. 不同通信算子到底在做什么

下面把推理里最常见的通信原语单独拿出来看。

### 2.1 send / recv

### 语义

一个 rank 把数据发给另一个 rank。

### 常见场景

- PP stage 间 hidden 传递；
- ring attention / ring KV 交换；
- 某些定制 attention backend。

### 典型特点

- 最简单；
- 延迟通常最低；
- 容易做成流水；
- 但如果需要构造“所有人和所有人交换”，会变复杂。

### 2.2 broadcast

### 语义

从一个 root 把同一份数据复制给所有 rank。

### 常见场景

- 初始化；
- 少量控制信息同步；
- 一些 naive all-to-all fallback。

### 典型特点

- 比 all-reduce 轻；
- 更适合单源多播；
- 一般不是 TP 主路径核心瓶颈。

### 2.3 reduce

### 语义

把所有 rank 的输入按某种规约操作聚合到一个 root。

### 常见场景

- 统计信息聚合；
- 非主路径控制逻辑；
- 很少作为推理主干热点。

### 典型特点

- 只把结果留在一个 rank；
- 比 all-reduce 少一步结果复制。

### 2.4 all_gather

### 语义

每个 rank 提供一个 shard，最后所有 rank 都拿到拼接后的完整结果。

### 常见场景

- TP 下恢复完整 hidden；
- 恢复完整 logits；
- CP 下聚合 full KV；
- SP 下恢复激活。

### 典型特点

- 强同步，但比 all-reduce 少规约计算；
- 带宽开销明显；
- 常见于“先切开，后拼回”的路径。

### 2.5 reduce_scatter

### 语义

先对所有 rank 的输入做规约，再把规约结果按 shard 分发给各 rank。

### 常见场景

- TP / SP 中维持 shard layout；
- 某些实现中替代一部分 all-reduce；
- EP combine 的局部规约路径。

### 典型特点

- 可以看作 all-reduce 的一半；
- 常用于避免过早恢复完整张量；
- 有利于后续继续在 shard 上计算。

### 2.6 all_reduce

### 语义

对所有 rank 的输入做规约，并把完整结果发回所有 rank。

### 常见场景

- TP 主路径；
- attention / MLP 的部分结果聚合；
- 某些 DP 同步。

### 典型特点

- 推理里最典型的强同步 collective；
- 在 TP 中非常常见；
- NCCL 官方明确指出：`ReduceScatter + AllGather` 等价于 `AllReduce`。

它最难受的地方不是单次一定最慢，而是：

**在 TP 中，它每层都出现。**

### 2.7 all_to_all / all_to_allv

### 语义

每个 rank 把不同子块发给不同 rank，每个 rank 同时从所有 rank 接收属于自己的那部分数据。

### 常见场景

- MoE token dispatch / combine；
- EP 主路径；
- 某些稀疏路由场景。

### 典型特点

- 语义最复杂；
- 对拓扑最敏感；
- 常常伴随变长消息、pack/unpack、permutation；
- 在 MoE 中通常是最棘手的通信问题。

## 3. 通信开销怎么比较

这里给一个更工程化的比较方式。

### 3.1 先看两个维度：时延项和带宽项

任何通信原语的开销，粗略都可以拆成：

- **时延项**：协议启动、同步、轮次、软件栈开销
- **带宽项**：真正搬了多少字节

因此：

- 小消息更看时延；
- 大消息更看带宽；
- decode 更容易暴露时延项；
- prefill 更容易暴露带宽项。

### 3.2 一个粗略但非常有用的公式视角

在常见的 `alpha-beta` 模型里，可以把通信代价粗略写成：

- `alpha`：每轮通信的固定启动时延
- `beta * N`：传输 `N` 字节的带宽代价

在理想 ring / tree 模型下，可以得到一些**粗略推断**：

- `send/recv`：`alpha + beta * N`
- `all_gather`：大约 `O((p-1) * alpha + (p-1)/p * beta * N_total)`
- `reduce_scatter`：大约和 `all_gather` 同阶
- `all_reduce`：大约等价于 `reduce_scatter + all_gather`
- `all_to_all`：理论上也可写成类似线性模型，但在 MoE 实际实现里往往明显更差，因为它不是规则拼接，而是稀疏 many-to-many 数据交换

这里的公式只是**帮助建立量纲感**，不是绝对性能预测公式。

### 3.3 在同等总数据量下的粗略相对成本

下面这个表是一个更贴近工程直觉的对比。

| 通信算子 | 同步强度 | 常见瓶颈 | 粗略成本感受 | 说明 |
| --- | --- | --- | --- | --- |
| `send/recv` | 低到中 | 单链路时延 / 带宽 | 最低 | 适合 PP、ring-attention |
| `broadcast` | 中 | root 扇出 | 低 | 不是推理主路径主角 |
| `reduce` | 中 | root 汇聚 | 低到中 | 结果只保留在一个 rank |
| `all_gather` | 中到高 | 带宽、完整张量恢复 | 中 | 常见于恢复 hidden / logits |
| `reduce_scatter` | 中到高 | 带宽、规约 | 中 | 有利于保留 shard layout |
| `all_reduce` | 高 | 强同步、慢 rank、每层重复 | 中到高 | TP 中最常见热点 |
| `all_to_all` | 高 | many-to-many、拓扑、pack/unpack | 高 | EP / MoE 的核心难点 |
| `all_to_allv` | 很高 | 变长消息、负载不均 | 最高 | MoE 动态路由里尤其棘手 |

如果只给一句经验判断：

**同等总数据量下，`all_reduce` 往往比 `all_gather` / `reduce_scatter` 更“重”，而 `all_to_all` 在 MoE 实际系统中通常最难优化。**

### 3.4 为什么 all_reduce 体感经常比公式更贵

因为它的体感成本不只是搬字节，还包括：

- 所有 rank 必须到齐；
- 最慢 rank 决定 collective 结束时间；
- TP 中它每层都出现；
- decode 时本地计算太小，固定同步开销被放大。

所以很多 profiling 里，`all_reduce` 看起来像“最慢通信”，本质上可能是：

**通信 + 等待 + 结构性同步点过密**

而不只是链路本身慢。

### 3.5 为什么 all_to_all 在 MoE 中常常更难

因为它不仅是 collective，还叠加了：

- token routing 的不规则性；
- top-k 导致的 token 复制；
- permutation / unpermutation；
- pack / unpack；
- expert 负载不均；
- 跨域拓扑差异。

所以在 MoE 里，`all_to_all` 的真实代价常常不是一个“干净的 collective benchmark 数字”，而是：

**通信 + 排布 + 数据重排 + 慢专家等待**

这也是为什么现在会有 DeepEP、FlashInfer、NIXL-EP 等专门优化 MoE A2A 的通信后端。

## 4. 并行方式和通信算子的对应关系

可以把主流推理并行方式和常见通信原语做一个映射。

| 并行方式 | 切分对象 | 主通信算子 | 热点开销 |
| --- | --- | --- | --- |
| DP / 请求并行 | 请求 / batch / 副本 | 主路径通常无；少量控制同步 | 权重/KV 复制，不是主路径 collective |
| TP | 层内张量 / hidden | `all_reduce`、`all_gather`、`reduce_scatter` | 每层强同步，decode 最敏感 |
| PP | 层深 | `send/recv` | stage 间 hidden 传递、流水气泡 |
| EP | experts / token-to-expert | `all_to_all` / `all_to_allv` / AG+RS 模拟 | dispatch/combine、拓扑、负载均衡 |
| CP（prefill） | sequence 长度 | `all_gather` 或 ring `send/recv` | 长序列 prefill，TTFT 和内存 |
| DCP（decode） | KV cache 的 token 维 | backend-specific KV 交换 | KV duplication 与额外通信折中 |
| SP | sequence 上部分激活 | `all_gather`、`reduce_scatter` | 激活内存与 shard 保持 |

## 5. 现在做推理并行时最实用的判断

### 5.1 如果模型能放下，优先 DP

原因很简单：

- 主路径通信最少；
- latency 更稳；
- 并发扩展自然。

### 5.2 如果单卡放不下，再考虑 TP 或 PP

- 单机内高速互联好：先看 TP
- 多机或互联一般：PP 往往比跨节点大 TP 更友好

### 5.3 如果是 MoE，不要默认沿用 dense 的 TP 思路

MoE 更常见的路线是：

- attention：`DP` 或小 `TP`
- experts：`EP`

因为你不想同时承担：

- attention / MLP 的 `all_reduce`
- expert dispatch 的 `all_to_all`

### 5.4 如果是长上下文或 MLA / GQA，必须把 CP / DCP 纳入考虑

- 长 prefill：看 CP
- KV cache 压力大：看 DCP
- 少 kv-head 模型：警惕 TP 带来的 KV duplication

## 6. 最后的总结

推理并行里最重要的不是死记并行名字，而是理解下面这层关系：

**切分方式决定通信模式，通信模式决定系统上限。**

从工程角度，今天最值得记住的几个事实是：

- `DP`：主路径通信最少，最适合吞吐扩展；
- `TP`：最容易被 `all_reduce` 主导，decode 尤其敏感；
- `PP`：通信更像 `send/recv`，跨节点常比大 TP 更稳；
- `EP`：核心是 `all_to_all`，MoE 优化本质是 dispatch/combine 优化；
- `CP / DCP`：本质是在用更多 attention 通信换更长上下文或更少 KV duplication。

如果只记一句话：

**推理系统里的通信开销，不是“哪种 collective 理论最快”，而是“哪种并行方式让最昂贵的 collective 反复出现在主路径上”。**

## 参考资料

- NCCL Collective Operations
  - https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2223/user-guide/doc/usage/collectives.html
- Megatron Core Parallelism Guide
  - https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/parallelism-guide.html
- vLLM Parallelism and Scaling
  - https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
- vLLM Expert Parallel Deployment
  - https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- vLLM Context Parallel Deployment
  - https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/
- vLLM Pipeline Parallelism Blog
  - https://vllm.ai/blog/llama31
