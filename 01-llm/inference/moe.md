# MoE 模型并行推理最佳实践

## 核心结论

目前 MoE 模型推理的主流最佳实践通常不是把整个模型简单地做大 TP，而是：

- attention 主要走 `TP` 或 `DP`
- experts 主要走 `EP`
- token dispatch / combine 使用高性能 `all-to-all`
- 需要做 expert load balancing
- 尽量把 EP 组限制在单机高速互联域内，例如 NVLink 域
- prefill 和 decode 分别调优

一句话概括就是：

**MoE 优先 EP，TP 只保留给 attention 或显存兜底。**

## 为什么 MoE 不宜无脑加 TP

MoE 里的 attention 和 expert FFN 是两种完全不同的计算类型。

- attention 更偏显存带宽瓶颈，增加 batch size 对吞吐提升有限
- expert FFN 更像 GEMM，batch 和 token 聚合起来后更容易把算力吃满

这也是很多系统会做 AF 分离的原因：

- attention 侧更适合 `DP` / 小 `TP`
- FFN / experts 侧更适合 `EP`

如果对 MoE 一味增加 TP，会带来两个问题：

1. expert 的局部 GEMM 变小，矩阵更碎，算力利用率下降
2. `all_reduce` 通信占比上升，容易从 compute bound 变成 communication bound

因此对 MoE 来说，TP 往往不是第一优先级，而更像兼容方案或内存兜底方案。

## 更常见的并行形态

### 1. Attention 用 TP，Experts 用 EP

这是目前最典型的做法。

- attention 模块和 dense 模型类似，继续按 head 或 hidden 维做 TP
- routed experts 不再用 TP 拆专家内部矩阵，而是按 expert 维度做 EP
- router 选择专家后，通过 dispatch / combine 把 token 发送到对应 expert 所在设备

这样可以避免把每个 expert 的 FFN 切得太碎，同时又能充分利用多卡上的专家参数容量。

### 2. DP + EP

在线服务里也经常看到 `DP + EP` 组合：

- `DP` 负责副本扩展和吞吐
- `EP` 负责承载 experts

这种方式的优点是：

- 相比大 TP，更容易获得更高吞吐
- attention 不需要承担过多 TP 通信
- experts 可以更自然地按卡分布

它的代价也很明确：

- EP 组内耦合更强，一张卡掉了，整个 expert group 都会受影响
- EP 扩展规模受专家数约束
- token dispatch / combine 的通信路径设计很关键

### 3. 小 TP + 大 EP

这是很常见的工程折中。

例如：

- `TP=1, EP=8`
- `TP=2, EP=4`

这样的思路通常比 `TP=8, EP=1` 更像“MoE 最优解”。

## 推理侧真正的瓶颈

MoE 推理的瓶颈往往不只在 GEMM，而在下面几项：

### 1. Token dispatch / combine

MoE 的核心额外开销是：

- token 被 router 选路
- token 发送到 expert 所在设备
- expert 计算结束后再回收

如果 dispatch / combine 做得差，推理瓶颈会直接落在通信而不是计算上。

因此现在主流系统都会用专门的 EP 通信实现，例如 DeepEP。

### 2. Expert load imbalance

训练时专家分布看起来均衡，不代表线上请求分布也均衡。

真实服务中常见的问题是：

- 少数 experts 被频繁命中
- 对应设备显著更忙
- 形成热点卡

所以 EP 方案通常必须带 expert load balancing，或者至少具备观测与重平衡能力。

### 3. Decode 和 Prefill 的优化目标不同

- prefill 更偏大吞吐
- decode 更偏低延迟

很多 MoE 系统在 prefill 和 decode 会使用不同的通信策略、不同的 overlap 策略，甚至不同的 kernel 路径。

## 工程实践建议

### 1. 单机优先把 EP 放在 NVLink 域内

如果是单机 8 卡，并且卡间有 NVLink，那么应优先在单机内部做 EP。

原因很简单：

- expert routing 的 token 交换非常依赖互联带宽
- 单机 NVLink 通常比跨机 IB/RDMA 更稳、更低延迟

### 2. 跨机时优先节点内 EP，节点间 DP

多机部署时，一般优先：

- 节点内做 EP
- 节点间做 DP

只有当单节点放不下 expert 参数，或者确实需要更大 expert 组时，才进一步做跨节点 EP。

### 3. TP 不要无脑拉满

对 MoE 推理来说，TP 过大常见副作用是：

- expert GEMM 太碎
- `all_reduce` 增加
- decode 延迟变差

所以更推荐把 TP 控制在一个较小值，再用 EP 去承接大部分 MoE 参数。

### 4. 量化优先考虑 experts

MoE 的参数大头通常在 experts。

如果目标是：

- 降显存
- 提高吞吐
- 降低 expert 计算成本

那么最值得优先量化的通常是 expert FFN，而不是先去动 attention。

## 一个常见判断标准

如果你在调 MoE 推理并行策略，可以重点观察这几件事：

1. expert token 分布是否明显偏斜
2. dispatch / combine 时间占总时延的比例
3. expert GEMM 是否已经碎成很多小矩阵
4. TP 继续增大后 decode latency 是降了还是升了

如果出现这些现象：

- TP 越大 expert 算得越碎
- A2A / route 占比越来越高
- 少数卡明显比其它卡更忙

那一般就说明应该往“更强 EP、更低 TP、更好的负载均衡”去调，而不是继续加 TP。

## 一个实用的经验法则

如果是 MoE 在线推理，常见的起步策略是：

- 模型能放下：先试 `TP=1, EP=机器内全部 GPU`
- attention / KV 压力较大：再试 `TP=2, EP=剩余划分`
- 没有成熟 EP 栈时：才退回纯 TP

所以纯 TP 更像“能跑”的方案，而不是“最好”的方案。

## 补充理解

你原来的几个判断基本是成立的，可以保留成下面这几个记忆点：

1. attention 和 FFN 属于不同类型算子，attention 更偏带宽瓶颈，FFN 更偏算力瓶颈，所以 AF 分离是合理的。
2. MoE 部署里更常见的是 `DP + EP` 或 `小 TP + EP`，而不是大 TP。
3. TP 的主要问题不是不能用，而是随着卡数增大，通信容易把收益吃掉。
4. `DP + EP` 也不是没有代价，EP 组内强耦合、受专家数约束、通信路径复杂，这些都是真问题。

## 参考资料

- vLLM Expert Parallel Deployment
  - https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- Megatron Core MoE
  - https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/moe.html
- SGLang Expert Parallelism
  - https://docs.sglang.io/advanced_features/expert_parallelism.html
- DeepEP
  - https://github.com/deepseek-ai/DeepEP
