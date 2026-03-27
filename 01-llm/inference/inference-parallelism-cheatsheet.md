# 推理并行速查表

这篇文档是 [inference-parallelism-and-communication.md](./inference-parallelism-and-communication.md) 的速查版，目标是把最常用的信息压缩成一页：

- 并行方式切什么
- 主通信算子是什么
- 典型瓶颈是什么
- 适用场景是什么

## 1. 并行方式速查

| 并行方式 | 切分对象 | 主通信算子 | 典型瓶颈 | 适用场景 | 不适合场景 |
| --- | --- | --- | --- | --- | --- |
| `DP` | 请求 / batch / 副本 | 主路径通常无；少量控制同步 | 权重复制、KV cache 复制 | 在线 serving、吞吐扩展、模型能放下 | 单请求必须跨多卡加速 |
| `TP` | 层内张量 / hidden | `all_reduce`、`all_gather`、`reduce_scatter` | 每层强同步、decode 下 `all_reduce` 占比高 | dense 模型、单机高速互联、单卡放不下 | decode 为主、MoE 已有 EP、MLA/GQA 下 kv-head 很少 |
| `PP` | 层深 / stage | `send/recv` | pipeline bubble、时延变长 | 多机、互联一般、GPU 数不规则 | 极致低时延交互式 |
| `EP` | experts / token-to-expert | `all_to_all` / `all_to_allv` | dispatch/combine、负载均衡、拓扑 | MoE 模型、专家参数大、吞吐优化 | 没有高效 A2A 栈、负载极不均 |
| `CP` | sequence 长度 | `all_gather` 或 ring `send/recv` | 长序列 KV 交换 | 长上下文 prefill、TTFT 优化 | 短上下文、小 batch decode |
| `DCP` | decode 阶段 KV token 维 | backend-specific KV 交换 | KV cache 交换、额外 attention 通信 | MLA/GQA、kv-head 少、decode KV 压力大 | 短上下文、KV duplication 不明显 |
| `SP` | sequence 上部分激活 | `all_gather`、`reduce_scatter` | 激活重分布 | TP 辅助优化、减 activation memory | 单独作为主并行方式意义不大 |

## 2. 通信算子速查

| 通信算子 | 语义 | 常见出现位置 | 粗略开销感受 | 真正痛点 |
| --- | --- | --- | --- | --- |
| `send/recv` | 点对点发送/接收 | PP、ring attention、KV 交换 | 低 | 链路带宽、流水调度 |
| `broadcast` | 一份数据发给所有 rank | 初始化、控制信息、后备路径 | 低 | root 扇出 |
| `reduce` | 所有 rank 规约到一个 root | 统计、控制路径 | 低到中 | root 汇聚 |
| `all_gather` | 所有 shard 拼成完整张量 | 恢复 hidden、恢复 logits、full-KV 聚合 | 中 | 完整张量恢复、带宽占用 |
| `reduce_scatter` | 先规约再分片 | TP/SP、维持 shard layout | 中 | 规约 + 分片同步 |
| `all_reduce` | 规约后所有 rank 都拿完整结果 | TP 主路径 | 中到高 | 强同步、慢 rank、每层重复 |
| `all_to_all` | 每个 rank 向所有 rank 交换不同数据块 | EP / MoE dispatch/combine | 高 | many-to-many、拓扑敏感、pack/unpack |
| `all_to_allv` | 变长 many-to-many 交换 | 动态 MoE 路由 | 很高 | 变长消息、负载不均、动态调度 |

## 3. 不同通信算子的粗略比较

注意，这只是工程上的经验排序，不是严格定理。实际开销还取决于：

- 消息大小
- 拓扑
- rank 数量
- 是否 overlap
- 是否有 pack / unpack / permutation
- 是否存在慢 rank

### 3.1 小消息更偏时延时

粗略感受通常是：

`send/recv < broadcast < all_gather ≈ reduce_scatter < all_reduce < all_to_all`

这里：

- `all_reduce` 会因为强同步更容易难看；
- `all_to_all` 会因为 many-to-many 和调度更难看。

### 3.2 大消息更偏带宽时

粗略感受通常是：

`send/recv < all_gather ≈ reduce_scatter < all_reduce < all_to_all`

这里：

- `all_reduce` 大致等价于 `reduce_scatter + all_gather`
- `all_to_all` 在规则场景下未必理论最差，但 MoE 里通常还叠加重排与不均衡，所以体感最难优化。

## 4. 每种并行方式最怕什么

| 并行方式 | 最怕的问题 |
| --- | --- |
| `DP` | 权重和 KV cache 复制过多，单卡容量不够 |
| `TP` | 每层 `all_reduce`，decode 时同步压过计算 |
| `PP` | stage 切分不均，pipeline bubble 大 |
| `EP` | token dispatch/combine、热点专家、跨域拓扑 |
| `CP` | full-KV 聚合太贵，或 ring 通信轮次太多 |
| `DCP` | KV 虽然切开了，但 attention 交换代价过高 |
| `SP` | 只增加了激活通信，却没有明显内存收益 |

## 5. 一眼判断应该优先看哪种并行

### 5.1 模型能放下，优先 `DP`

原因：

- 主路径通信最少
- 时延最稳
- 吞吐扩展最自然

### 5.2 模型放不下，但单机互联很好，先看 `TP`

前提：

- dense 模型
- prefill 占比较高
- 不准备把 TP 开很大

### 5.3 多机或互联一般，先认真考虑 `PP`

原因：

- `send/recv` 往往比跨机大 `all_reduce` 更好管理

### 5.4 MoE 模型优先把 `EP` 纳入默认方案

经验路线通常是：

- attention：`DP` 或小 `TP`
- experts：`EP`

### 5.5 MLA / GQA / 少 kv-head，必须检查 `DCP`

如果：

- `tp_size > kv_heads`

就要高度怀疑：

- KV duplication 已经在浪费显存和带宽

## 6. 面向 decode 的实用判断

### 如果目标是低 latency

优先级通常是：

1. 少同步
2. 小 TP
3. 减少 `all_reduce`
4. 优化单步 KV cache / attention
5. 避免慢 rank

对应更偏好的并行路线通常是：

- `DP`
- `小 TP`
- 必要时 `DCP`

### 如果目标是高 throughput

优先级通常是：

1. 提高并发
2. 优化 KV cache 利用率
3. MoE 上加强 `EP`
4. 做 overlap
5. 量化

对应更偏好的并行路线通常是：

- `DP`
- `EP`
- `CP / DCP`

## 7. 最常见的几种组合

| 组合 | 常见用途 | 典型问题 |
| --- | --- | --- |
| `DP` | 最简单在线部署 | 权重/KV 复制 |
| `TP + DP` | dense 模型常规扩展 | `all_reduce` 每层同步 |
| `PP + DP` | 多机大模型 | pipeline bubble |
| `TP + EP` | MoE 兼顾容量与性能 | `all_reduce` + `all_to_all` 双重通信 |
| `DP + EP` | MoE 在线吞吐常见最优点 | expert 对齐、A2A 和负载均衡 |
| `TP + DCP` | kv-head 少但仍需 TP | attention 交换复杂 |
| `DP + EP + DCP` | MLA/GQA MoE 的更现代路线 | 调度和实现复杂度高 |

## 8. 最后记忆点

如果只记 5 句话：

1. `DP` 最省主路径通信，但最吃副本显存。
2. `TP` 的问题不是不能用，而是它把每层都变成同步问题。
3. `PP` 的主路径通信更简单，跨节点时常比大 TP 更稳。
4. `EP` 的核心不是“把专家分开”本身，而是能不能把 `all_to_all` 做好。
5. MLA / GQA / 少 kv-head 模型下，不要只看 `TP`，一定要把 `DCP` 一起考虑。

## 参考延伸

- [inference-parallelism-and-communication.md](./inference-parallelism-and-communication.md)
- [decode-throughput-vs-latency.md](./decode-throughput-vs-latency.md)
- [attention-dp-vs-tp.md](./attention-dp-vs-tp.md)
- [tp-allreduce-bottleneck.md](./tp-allreduce-bottleneck.md)
- [vllm-moe-parallelism-notes.md](./vllm-moe-parallelism-notes.md)
