# TP 中为什么 AllReduce 很慢，以及可以怎么优化

在 Tensor Parallelism（TP）里，`all_reduce` 往往是最显眼的通信瓶颈。它慢，通常不是因为通信库“不够快”，而是因为它天然同时吃了两类成本：

- 延迟项：所有 rank 必须同步到齐，还要走 collective 协议；
- 带宽项：要在多卡之间真的搬运一整份激活量级的数据。

可以粗略理解为：

`all_reduce 时间 ≈ 协议/同步延迟 + 数据传输时间 + 等待最慢 rank 的时间`

所以 profiler 里看到 `all_reduce` 很大，通常不是单一原因，而是几类问题叠加的结果。

## 核心结论

- `all_reduce` 慢，不只是因为“数据传得多”，更因为它是强同步点。
- 在 TP 中，层数越多、TP 越大、batch 越小，`all_reduce` 越容易成为主导瓶颈。
- decode 场景比 prefill 更容易被 `all_reduce` 拖慢。
- 最有效的优化通常不是先调 NCCL，而是先减少 `all_reduce` 的次数和等待时间。

## 为什么 AllReduce 会慢

### 1. 它是强同步

`all_reduce` 不是“谁先到谁先继续”，而是所有 rank 都必须参与。只要有一张卡前面的计算、访存或调度慢一点，其他卡都会在 `all_reduce` 这里等它。

因此 profiler 里看到的 `all_reduce` 时间，往往不只是纯链路传输时间，还包括：

- 等待最慢 rank 到齐；
- stream 依赖造成的阻塞；
- 前后 kernel 无法 overlap 时的空转。

### 2. TP 把每层都变成同步点

在 Transformer 的 TP 实现里，很多层都会在特定位置触发 collective，例如：

- attention 的输出投影之后；
- MLP 的下投影之后；
- 某些需要恢复完整 hidden state 的中间位置。

这意味着 TP 的代价不是“偶尔通信一次”，而是“几乎每层都要同步”。层数一多，累计成本就会非常可观。

### 3. decode 场景下本地计算太小

decode 时通常是：

- batch 很小；
- 每步只生成 1 个 token；
- 单层本地 matmul 不大。

这时 `all_reduce` 的固定开销和同步等待很难被摊薄，于是会出现：

- 单层计算时间不大；
- `all_reduce` 占比异常突出；
- TP 增大后 latency 不降反升。

这其实是 TP 在 decode 场景里的典型表现。

### 4. TP 越大，局部 GEMM 越碎

TP 增大后，每张卡只算更小的一块矩阵，于是：

- 单卡 GEMM 利用率下降；
- kernel launch 占比上升；
- compute 变小了，但 collective 次数没有减少。

结果就是一种很常见的现象：

- 计算时间下降；
- 总时延没有明显下降；
- `all_reduce` 占比反而升高。

### 5. 拓扑不理想

如果参与 TP 的 GPU 不是处在理想互联结构里，例如：

- 跨 PCIe switch；
- 跨 NUMA；
- 跨 socket；
- 跨机走 RDMA；

那么 `all_reduce` 的时间会明显被放大。

同样是 `TP=2`，不同 GPU 组合的时延可能差很多。

### 6. 通信和计算没有真正重叠

理论上，一些 collective 可以和部分本地计算 overlap。但如果：

- stream 依赖设计不合理；
- kernel 排队过于串行；
- buffer 生命周期限制了调度；
- 存在 host-side wait 或隐式同步；

那么 `all_reduce` 就会以纯阻塞形式暴露在 trace 里。

### 7. collective 算法和运行时配置不合适

不同消息大小、不同拓扑下，ring、tree、collnet 等算法表现差异很大。如果运行时选到的策略与硬件结构不匹配，也会让 `all_reduce` 表现变差。

## 有哪些可优化的地方

### 1. 第一优先级：减少 AllReduce 的出现频率

这是通常最有效的优化方向。

- 不要无脑增大 TP；
- MoE 场景优先考虑 `attention 用 DP 或小 TP，experts 用 EP`；
- 如果单卡放得下，优先用 `TP=1`；
- 能让 shard 状态跨多层保留，就不要每层都恢复完整张量。

这类优化通常比单纯调通信库更值钱，因为它是在减少问题本身，而不是微调症状。

### 2. 第二优先级：减少等待最慢 rank 的时间

很多时候，`all_reduce` 大并不是因为链路真慢，而是因为某个 rank 总慢一步。

应该重点检查：

- `all_reduce` 前是否某个 rank 的 GEMM、KV 读写或其他 kernel 更晚结束；
- 是否某几张卡同时在跑别的通信或拷贝；
- 是否存在 rank skew，导致 collective 变成“大家等最慢者”。

如果是这个问题，优化重点就不是换 collective，而是消除 rank 不齐。

### 3. 第三优先级：控制 TP 粒度

对 attention 来说，TP 通常不宜开得过大。

一个非常常见的经验是：

- `TP=1` 往往最好；
- 显存有压力时，`TP=2` 是常见折中；
- 再往上加，通信收益比常常开始恶化。

如果 `TP=2` 相比 `TP=1` 已经没有明显收益，继续加大 TP 往往只会让 `all_reduce` 更难看。

### 4. 第四优先级：拓扑放置优化

尽量把 TP group 放在高速互联域内，例如：

- 同一 NVLink 域；
- 尽量避免跨 socket；
- 尽量避免跨 PCIe root complex；
- 如果必须跨机，尽量避免让高频 `all_reduce` 走最慢路径。

这类优化经常非常有效，而且往往比改内核更容易落地。

### 5. 第五优先级：做真正有效的 overlap

如果要做通信计算重叠，重点不是“把 collective 变异步”，而是确认依赖关系真的允许：

- 哪些本地计算可以提前开始；
- 哪些通信可以后移；
- 哪些同步可以拆开；
- 是否有 host-side wait、隐式 barrier 或 stream 串行化。

没有正确的依赖切分，所谓 overlap 往往只是表面异步，真实时间并没有缩短。

### 6. 第六优先级：尽量延迟恢复完整张量

优化的关键通常不是机械地把 `all_reduce` 改成 `reduce_scatter + all_gather`，而是：

- 看 `reduce_scatter` 后的分片结果是否能被后续直接消费；
- 如果能，就先不 `all_gather`；
- 让 shard layout 在更多层中保留下去。

真正值钱的是“减少完整张量恢复的频率”，而不是“把一个 collective 拆成两个”。

### 7. 第七优先级：调 NCCL 和通信参数

这通常是次一级优化，但在某些环境里依然有效：

- 检查是否真的走到了 NVLink；
- 比较不同 `NCCL_ALGO`、`NCCL_PROTO` 的效果；
- 检查是否存在 P2P 被禁用、NUMA 绑核不对、GPU/NIC 亲和性不合理；
- 多机场景下检查 RDMA / IB 配置是否正确。

如果是单机，优先先看拓扑和 rank mapping；如果是多机，再重点看 RDMA 配置。

## 一个实用的排查顺序

当 profiling 里看到 `all_reduce` 很大时，可以按下面顺序排查：

1. 先比较 `TP=1` 和 `TP=2` 的总时延变化；
2. 再看 `all_reduce` 前是否存在明显 rank skew；
3. 再看不同 GPU 组合下时延是否有明显变化；
4. 再区分 prefill 和 decode，通常 decode 更容易暴露同步问题；
5. 最后再去调 collective 算法和运行时参数。

这个顺序的原因很简单：绝大多数情况下，结构性问题比通信库参数更重要。

## 总结

`all_reduce` 慢，核心不只是“搬数据慢”，而是：

**TP 把每层都变成了强同步点，而 decode、小 batch、不理想拓扑和慢 rank 会把这个问题进一步放大。**

因此最有效的优化通常不是先调库，而是：

- 少做 `all_reduce`；
- 把 TP 控制在合理范围内；
- 选对 GPU 拓扑；
- 避免 rank 不齐；
- 只在必要时恢复完整张量。

一句话概括：

**优化 `all_reduce`，首先要减少它作为系统主路径同步点的存在感，而不是只盯着它的带宽数字。**

## 参考延伸

- [attention-dp-vs-tp.md](./attention-dp-vs-tp.md)
- [moe.md](./moe.md)
- [moe-ep-key-techniques.md](./moe-ep-key-techniques.md)
