# Attention 部分用 DP 还是 TP 更好

如果问题限定在 **MoE 推理场景下的 attention 部分**，一个比较稳定的工程结论是：

**能不用大 TP，就尽量不用大 TP。**

更常见也更优的路线通常是：

- attention 用 `DP` 或很小的 `TP`
- experts 用 `EP`

也就是常说的 `DP attention + EP experts`，这通常比“attention 也做大 TP”更符合推理系统的真实瓶颈结构。

## 核心结论

### 1. 从延迟和系统效率看，attention 往往更适合 DP 或小 TP

attention 做 TP 的主要问题是，每层都会引入额外同步。常见实现里，QKV、输出投影或其他线性层在切分后，都会在某些位置触发：

- `all_reduce`
- `all_gather`
- 或 `reduce_scatter + all_gather`

这意味着 attention 不再是本地完成的算子，而变成了“每过若干子层就要跨卡同步一次”的算子。

在 decode 场景里，这个问题尤其明显：

- batch 小
- 每步只生成 1 个 token
- 单层本地计算很小
- 但 `all_reduce` 的固定时延和同步开销不会同比缩小

所以 profiler 里经常会看到 attention TP 路径中的 `all_reduce` 占比很高。

### 2. 从显存和可运行性看，attention 有时又必须使用 TP

如果 attention 权重、激活或 KV cache 单卡放不下，那么 TP 不是“更优选择”，而是“必要选择”。

因此 attention 的并行选择，本质上不是单纯追求某个理论最优，而是在下面几件事之间做折中：

- 单卡显存是否放得下
- decode latency 是否敏感
- prefill 吞吐是否更重要
- GPU 互联是否足够强
- 是否已经存在 experts 的 EP 通信

## 为什么 attention 用 TP 经常不划算

### 1. 每层都会引入同步点

TP 的本质，是把一个线性层按 hidden 维或输出维切到多张卡上，每张卡先做局部 matmul，再通过 collective 拼回完整结果。

这带来一个根本代价：

**attention 的前向路径会被频繁切成“局部计算 + 全局同步”的交替结构。**

层数一多，同步次数就会非常可观。

### 2. TP 越大，局部 GEMM 越碎

TP 增大后，每张卡负责的矩阵更小，结果是：

- 单卡 GEMM 利用率下降
- kernel launch 占比上升
- 算得更快了，但通信次数并没有减少

所以经常出现一种看起来反直觉的现象：

- compute 时间下降了
- 但总 latency 没明显下降
- `all_reduce` 占比反而上升

这不是 profiler 出错，而是 TP 的典型表现。

### 3. attention 本身就更偏带宽瓶颈

和 FFN 相比，attention 往往更依赖：

- HBM 访问
- KV cache 读写
- softmax 前后的数据搬运

它并不是一个“越切越容易把算力吃满”的模块。对 attention 来说，很多时候问题不是 FLOPs 不够，而是带宽和同步在拖后腿。

因此 attention 做大 TP，经常不像 dense GEMM 那样自然获益。

### 4. decode 是 TP 最差的工作点之一

decode 时每步算量很小，但同步点仍然保留，这会导致：

- 通信固定时延难以摊薄
- `all_reduce` 在 trace 中异常突出
- 多卡之间更容易互相等待

所以如果系统是在线服务或 low-latency serving，attention 做大 TP 往往不是最优点。

## 为什么很多系统更偏向 DP attention

### 1. attention 可以本地算完

如果采用 DP attention，那么每张卡都有 attention 的完整权重副本，本地就可以完成 attention 前向，不需要在每层做 TP 同步。

这会带来几个直接收益：

- decode latency 更稳
- profiler 里的 `all_reduce` 明显减少
- 执行路径更简单
- 更容易扩 replica 做吞吐

### 2. attention 的通信压力可以让位给 experts 的 EP

在 MoE 系统里，experts 本身就已经需要做 dispatch / combine，也就是专家并行通信。

如果 attention 也做大 TP，那么系统里会同时出现两类重通信：

- attention 的 `all_reduce`
- experts 的 `all_to_all`

这两种通信叠加后，整体 trace 往往会变得很难看。

相反，如果 attention 用 DP 或小 TP，那么通信预算可以更多留给 experts 的 EP 路径，这通常更符合 MoE 推理的真实收益结构。

### 3. 更适合在线推理

在线服务更看重：

- 小 batch 下的稳定时延
- decode token latency
- 系统复杂度与可维护性

在这些维度上，`DP attention + EP experts` 通常更优于 `大 TP attention + EP experts`。

## attention 什么时候应该用 TP

下面几种情况，attention 使用 TP 是合理甚至必要的。

### 1. 单卡放不下

这是最现实的理由。如果 attention 权重或 KV cache 单卡放不下，就必须做 TP 或其他切分。

### 2. 超长上下文场景

如果是长上下文 prefill 或长序列处理，单卡显存和带宽压力都很大，这时小 TP 可能有明显价值。

### 3. batch 足够大、互联足够强

如果是：

- NVLink 很强
- batch 较大
- 以 prefill 为主
- 系统对 TP overlap 做得比较成熟

那么 TP 的通信成本可以被更大的计算量摊薄，收益会比 decode 场景更明显。

### 4. 系统已经围绕 TP 深度优化

如果一个推理栈已经对下面几件事做了深度优化：

- kernel fusion
- 通信计算重叠
- 拓扑感知 placement
- 减少同步点

那么 attention 用小 TP 可以是合理折中。

但即便如此，很多系统的甜点区也依然是：

- `TP=1`
- 或 `TP=2`

而不是更大的 TP。

## attention 什么时候更适合 DP

下面这些情况，通常更推荐 DP attention：

- 在线 serving，尤其以 decode 为主；
- batch 不大；
- 模型已经有 experts 的 EP 通信；
- 单卡显存还能放下 attention 和 KV cache；
- 目标是整体吞吐和稳定延迟，而不是单请求极限加速。

## 一个非常实用的工程判断

如果是 MoE inference，可以用下面这个经验法则作为起点：

- 显存够：`attention = DP / TP1`，`experts = EP`
- 显存有压力：`attention = TP2`
- 再放不下：再考虑更大的 TP，或者结合别的并行方式

也就是说，attention 最常见的甜点区不是大 TP，而是：

- `TP=1`
- 或 `TP=2`

## 为什么 profiling 里 all reduce 会很多

如果你在 profiling 数据中看到 `all_reduce` 占据大量时间，这通常不是偶然，而是 attention TP 的典型副作用。常见原因包括：

### 1. batch 太小，通信固定开销压过计算

decode 时单层 matmul 很小，但 `all_reduce` 的固定时延还在，于是通信占比迅速变高。

### 2. TP 开得太大，局部 GEMM 太碎

单卡算量下降了，但每层同步次数没有减少，于是看起来像是“算得更少，等得更多”。

### 3. profiler 里的 all reduce 往往还包含等待

很多时候，trace 中的 `all_reduce` 不只是纯粹的数据交换，还包括：

- 等待其他 rank 到齐
- stream 依赖造成的阻塞
- 无法 overlap 的空转时间

因此 `all_reduce` 很大，不一定完全是链路带宽差，也可能是慢 rank 或同步点太多。

### 4. 拓扑不理想

如果 GPU 之间不是全 NVLink，或者跨 socket、跨 PCIe switch，那么 `all_reduce` 时间会进一步放大。

## 总结

如果只看 MoE 推理里的 attention 部分，一个比较稳妥的结论是：

- 从延迟和系统效率看，attention 通常更适合 `DP` 或很小的 `TP`
- 从显存和单请求可运行性看，attention 有时必须使用 `TP`
- 因此工程上最常见的较优方案，通常不是“大 TP attention”，而是：
  - `DP attention + EP experts`
  - 或 `小 TP attention + EP experts`

## 参考延伸

- [moe.md](./moe.md)
- [moe-ep-key-techniques.md](./moe-ep-key-techniques.md)
