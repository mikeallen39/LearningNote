# Decode 吞吐优化与时延优化

在 LLM 推理里，很多人会把“decode 变快”当成一个目标，但工程上其实至少有两个不同目标：

- **提升 decode 吞吐**：单位时间内生成更多 token，也就是总 token/s 更高；
- **降低 decode 时延**：单个请求每生成一个 token 更快，也就是单 token latency 更低。

这两个目标相关，但并不等价，很多优化甚至是冲突的。理解它们的差异，是做并行策略、KV cache、调度和内核优化的前提。

## 一句话结论

- **吞吐优化**关注的是“同时服务更多 token”；
- **时延优化**关注的是“让每一步更短”；
- 同一个优化在吞吐视角下是收益，在时延视角下可能反而是代价；
- 因此 decode 优化不能只问“快不快”，而要先问“优化的是 throughput 还是 latency”。

## 1. 为什么 decode 比 prefill 更难优化

decode 和 prefill 的根本差异在于有效工作量。

- prefill：一次处理整段 prompt，当前层会处理 `B x S` 个 token；
- decode：每步通常只为每个活跃请求生成 `1` 个 token，当前层更接近处理 `B x 1` 个 token。

这会导致几个直接后果：

- decode 每步的有效 token batch 远小于 prefill；
- 本地 GEMM 更小，更难摊薄固定开销；
- kernel launch、通信同步、host 调度等固定开销会被放大；
- `all_reduce`、`all_gather`、`all_to_all` 的相对占比更容易升高；
- latency 和 throughput 会更明显地受调度与 KV cache 管理影响。

所以 decode 优化的重点，往往不再只是“写更快的 matmul”，而是：

- 减少每步同步；
- 减少每步数据搬运；
- 增强 KV cache 利用率；
- 改善调度；
- 提高重叠效率。

## 2. 提升 decode 吞吐在优化什么

提升 decode 吞吐，目标是：

**在单位时间内生成更多 token。**

这意味着你关心的是：

- token/s
- requests/s
- GPU 平均利用率
- 在更高并发下的稳定产出能力

吞吐优化通常不要求单个请求更快，而要求整个系统同时处理更多活跃序列时仍然效率高。

## 3. 降低 decode latency 在优化什么

降低 decode 时延，目标是：

**让单个请求的每一步生成尽可能短。**

这意味着你关心的是：

- 单 token latency
- time-to-next-token
- 交互式体验
- 小 batch 下的响应速度

时延优化通常更关注：

- 每层同步点有多少；
- 每步 launch 开销大不大；
- 有没有慢 rank；
- 有没有不必要的 gather / reduce；
- 单步数据路径是否足够短。

## 4. 提升 decode 吞吐的常见方法

### 4.1 提高并发与 batch 利用率

这是吞吐优化里最基础、也最有效的一类手段。

典型做法：

- continuous batching
- request merging
- 更好的 scheduler
- 分离 prefill 与 decode 队列

核心思想是：

**让每一步 decode 能同时处理更多活跃序列，从而摊薄固定开销。**

### 4.2 提高 KV cache 利用率

很多吞吐瓶颈不在算力，而在 KV cache。

常见方向：

- PagedAttention
- prefix caching
- 减少碎片
- 降低 duplication
- 合理的 block 管理

如果 KV cache 管理差，就算算子本身够快，也上不去并发。

### 4.3 用 EP 承载 MoE experts

在 MoE 模型里，吞吐优化往往更依赖 expert 侧策略，而不是单纯扩大 TP。

更典型的方向是：

- experts 用 EP 分散到更多卡；
- 做 token dispatch / combine 优化；
- 做负载均衡；
- 做通信计算重叠；
- 在必要时复制热点专家。

因为 MoE 的吞吐上限，常常不是 attention 算不动，而是 expert 权重带宽和路由通信在限制。

### 4.4 双 batch / 多 batch overlap

吞吐场景下非常重要的手段，是把：

- 当前 batch 的计算
- 前一个或后一个 batch 的通信

尽量重叠起来。

典型方向：

- dual batch overlap
- two-batch overlap
- single-batch overlap
- dispatch/combine 与 shared experts / attention overlap

它们未必直接降低单个请求时延，但能明显提高整体 token/s。

### 4.5 量化

吞吐优化中，量化往往非常值钱，因为它同时影响：

- 显存占用
- 带宽占用
- 每卡可承载并发
- 某些 kernel 的吞吐率

例如：

- FP8
- W8A8
- KV cache quantization

这些优化不一定让单步 latency 线性下降，但通常能让系统承接更多并发。

### 4.6 speculative decoding

这类方法的价值主要体现在：

- 提高平均生成速度；
- 用更少的 target model step 生成更多 token；
- 在高吞吐场景下特别有吸引力。

它不一定总是最优的低时延方案，但常常是强有力的吞吐工具。

## 5. 降低 decode latency 的常见方法

### 5.1 减少每步同步点

这通常是时延优化的第一优先级。

做法包括：

- attention 避免大 TP；
- 减少 `all_reduce`；
- 避免不必要的 `all_gather`；
- 让本地 shard 能被后续继续消费；
- 尽量不要每层恢复完整张量。

原因很简单：

decode 每步有效 token batch 很小，固定同步开销特别容易主导总时间。

### 5.2 控制 TP 规模

对 decode latency 来说，TP 不是不能用，但通常不应开太大。

常见经验：

- `TP=1` 往往最好；
- 放不下时，`TP=2` 是较常见折中；
- 再往上，局部 GEMM 变碎、`all_reduce` 增多，收益常常恶化。

尤其对 MoE + MLA / GQA 模型，attention 做大 TP 通常很容易让 latency 变差。

### 5.3 缩短 host 和 runtime 路径

单步时延里，host 侧开销经常被忽视。

需要关注：

- kernel launch 开销；
- host-device 同步；
- stream 管理；
- Python / runtime 调度成本；
- graph capture / replay。

常见做法：

- CUDA Graph
- 静态图
- fused launch
- 减少 host-side wait

这类优化对 decode latency 特别重要，因为单步计算本来就不大。

### 5.4 优化 KV cache 访问路径

decode 时 attention 往往更偏 memory-bound。

所以时延优化很依赖：

- 更高效的 KV cache layout；
- 更高效的 paged attention kernel；
- 更低的 cache miss；
- 更少的重复搬运；
- 减少 KV duplication。

对 MLA / GQA 来说，必要时还要结合：

- DCP
- context parallel

否则 TP 可能会在 decode 阶段浪费大量 KV cache。

### 5.5 避免慢 rank

只要有 collective，同步路径就会被最慢 rank 决定。

所以时延优化里必须关注：

- rank skew；
- 某张卡是否额外忙；
- 拓扑是否不理想；
- 某些通信是否和别的流抢资源；
- 是否有卡在等待其他 rank。

很多 profiling 里看起来“通信很慢”，本质其实是“有 rank 没到齐”。

### 5.6 decode 专项算法优化

如果目标是交互式时延，下面这些往往非常重要：

- speculative decoding
- early accept / reject 优化
- prefill / decode 分离调度
- 更小的 decode batch
- priority scheduling

这些优化未必最大化总吞吐，但可以显著改善下一 token 响应时间。

## 6. 哪些优化更偏吞吐，哪些更偏时延

可以用下面这个粗略分类来理解。

| 优化项 | 更偏吞吐 | 更偏时延 | 说明 |
| --- | --- | --- | --- |
| continuous batching | 是 | 否 | 提高并发，可能增加单请求等待 |
| larger effective batch | 是 | 否 | 更能摊薄固定成本，但不一定适合交互式 |
| 小 TP / 少同步 | 否 | 是 | 直接减少每步同步点 |
| EP + overlap | 是 | 部分 | 对 MoE 吞吐通常更关键 |
| KV cache 优化 | 是 | 是 | 两边都重要 |
| 量化 | 是 | 是 | 但收益方向和幅度依 workload 不同 |
| speculative decoding | 是 | 是 | 不同实现偏重点不同 |
| CUDA Graph / 静态图 | 否 | 是 | 对单步 launch 开销很有效 |
| prefix caching | 是 | 部分 | 吞吐收益更常见 |
| DCP / CP | 是 | 部分 | 更偏容量和长上下文 decode 效率 |

这张表不是绝对的，但能说明一个核心事实：

**很多提升 throughput 的手段，并不会自然降低 latency；很多降低 latency 的手段，也不会自然让系统吞吐最大化。**

## 7. TP 为什么在 decode 中经常显得“不划算”

这和你前面的问题直接相关。

TP 的问题不是“完全不能用”，而是：

- 每层都会有同步；
- decode 每步计算太小；
- `all_reduce` 固定开销难以摊薄；
- TP 越大，局部 GEMM 越碎；
- 对 MLA / MQA / GQA，KV cache 还可能 duplication；
- 如果 MoE 已经有 EP 通信，再叠加 TP 通信往往更难看。

所以 decode 场景下常见的更优选择通常是：

- attention 用 `DP` 或小 `TP`
- experts 用 `EP`
- KV cache 压力大时看 `DCP`

这不是说 TP 一定错，而是说：

**decode 优化的工作点，通常不在“大 TP”。**

## 8. 一套更实用的工程判断

如果你在调 decode，可以先问自己下面几个问题。

### 8.1 如果目标是低 latency

先看：

1. 每步是不是有太多同步点？
2. `all_reduce` 是否主导？
3. attention 是否开了过大的 TP？
4. 是否存在慢 rank？
5. host-side launch 和同步开销是否过大？

如果这些问题明显，那么优先级通常是：

- 缩 TP
- 减同步
- 优化 runtime
- 优化单步 attention / KV cache 路径

### 8.2 如果目标是高 throughput

先看：

1. GPU 是否真的被并发吃满？
2. KV cache 是否成了容量瓶颈？
3. MoE experts 是否已经用 EP 分摊？
4. overlap 是否做起来了？
5. quantization 是否能换来更高并发？

如果这些问题明显，那么优先级通常是：

- 提高并发
- 优化 KV cache 管理
- 强化 EP
- 做 batch overlap
- 上量化

## 9. 总结

decode 优化里最容易犯的错误，是把“提吞吐”和“降 latency”混成一个目标。

更准确的理解应该是：

- **吞吐优化**：让系统同时生成更多 token；
- **时延优化**：让单个请求更快拿到下一个 token。

因此：

- 吞吐更看重 batching、KV cache 利用率、EP、overlap、量化；
- 时延更看重少同步、小 TP、runtime 开销、单步 attention 路径、避免慢 rank。

一句话概括：

**decode 优化不是单一问题，而是“吞吐目标”和“时延目标”驱动下的两套不同工程路线。**

## 参考延伸

- [attention-dp-vs-tp.md](./attention-dp-vs-tp.md)
- [tp-allreduce-bottleneck.md](./tp-allreduce-bottleneck.md)
- [vllm-moe-parallelism-notes.md](./vllm-moe-parallelism-notes.md)
- [moe.md](./moe.md)
