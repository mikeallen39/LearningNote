# vLLM 中 MoE 模型的并行部署笔记

这篇笔记是在一篇知乎文章的基础上扩展而来，主线问题是：

**部署 DeepSeek / Qwen3-A3B 这类 MoE 模型时，TP、DP、PP、EP 到底应该怎么组合，为什么这样组合，以及 MLA / MQA / GQA 会把这个问题变得更复杂。**

相比原文，这里额外补充了：

- vLLM 官方文档中的并行语义；
- `DP attention` 在 vLLM 里的真实含义；
- 开启或不开启 EP 时，MoE expert 的通信行为差异；
- MLA / GQA 模型下 KV cache duplication 的根本原因；
- `DCP`（Decode Context Parallel）为什么是 MLA 模型部署时必须关注的一条线；
- 单机与多机下更实用的选型建议。

## 一句话结论

如果是 MoE 推理，特别是带 MLA / MQA / GQA 特征的模型，工程上通常不应该再把 attention 和 experts 都一股脑塞进大 TP，而更应该分开看：

- **attention**：优先 `DP` 或很小的 `TP`
- **experts**：优先 `EP`
- **长上下文 decode / MLA KV cache 压力大时**：进一步考虑 `DCP`

也就是说，真正实用的组合往往不是“全模型大 TP”，而是：

- `DP attention + EP experts`
- 或 `小 TP attention + EP experts`
- MLA / GQA 下再叠加 `DCP`

## 1. 几种并行方式先分别在切什么

### 1.1 TP：切层内权重

TP（Tensor Parallelism）是在层内切矩阵，常见做法是按输入维或输出维切开。

优点：

- 单个模型副本可以跨多卡放下；
- 单请求延迟在某些场景下会下降；
- dense 模型是最常见的第一选择。

缺点：

- 每层都会插入 `all_reduce` / `all_gather` / `reduce_scatter`；
- decode 场景下每步的有效 token batch 远小于 prefill，通信占比很容易压过计算；
- TP 越大，局部 GEMM 越碎；
- 对 MLA / MQA 模型，KV cache 常常不能优雅地按 TP 扩展。

### 1.2 DP：复制模型副本

DP（Data Parallelism）是在多个 rank 上复制模型权重，每个 rank 处理不同请求或不同 batch 分片。

优点：

- 实现直观；
- 最适合扩吞吐；
- 每个 DP rank 都有独立 KV cache，更容易利用 prefix caching。

缺点：

- 权重复制多份，吃显存；
- 对单请求 latency 帮助不大；
- 在 vLLM 的 MoE 场景下，DP rank 之间未必完全独立。

### 1.3 PP：按层切模型

PP（Pipeline Parallelism）是把模型按层切到不同 GPU / 节点上。

优点：

- 适合单节点放不下模型时继续扩展；
- 对不规则 GPU 切分更友好；
- vLLM 官方明确提到：如果单机 GPU 间没有 NVLink，或者 GPU 数不能均匀切 TP，可以考虑 PP。

缺点：

- 单请求路径更长；
- 调度更复杂；
- 对在线低时延服务通常不如单节点小 TP / DP 自然。

### 1.4 EP：切 experts，不切 attention

EP（Expert Parallelism）是 MoE 特有的并行方式。它不是又一种“全模型并行”，而是：

**只改变 MoE expert 的放置方式和通信方式。**

这点非常重要。EP 一般只作用于 routed experts，不会自动改变 attention 的并行策略。

## 2. 不开 EP 和开 EP，到底差在哪

这是理解 vLLM MoE 部署最关键的一步。

### 2.1 不开 EP：MoE expert 默认还是走 TP 语义

vLLM 官方文档明确写到：

- 对 MoE 模型，如果不启用 `--enable-expert-parallel`
- expert 层默认会形成大小为 `DP × TP` 的 tensor parallel group

也就是说：

- attention 还是按你设置的 TP / DP 跑；
- expert 也没有“真正按 expert 分布到不同卡上”；
- experts 更接近“像 dense 层一样被 TP 切分”。

这种情况下，主通信更接近：

- `all_reduce`
- `all_gather`
- `reduce_scatter`

而不是 MoE 里最典型的 token dispatch `all_to_all`。

### 2.2 开 EP：expert 层切换成 expert parallel 语义

开启 `--enable-expert-parallel` 后：

- attention 层仍按 TP / DP 规则运行；
- 只有 expert 层切换成 EP；
- experts 被分散到不同 GPU 上；
- token 需要被路由到对应 expert 所在的 rank。

这时 expert 层的主通信才真正变成：

- dispatch / combine
- 本质上是 `all_to_all`

所以“开 EP”不是一键换一种并行模式，而是：

**把 MoE 的 expert 部分从 TP 语义切换成 EP 语义。**

## 3. 为什么文章会强调 DP attention

文章里讲的 `DP attention`，如果按直觉理解成“多个完全独立副本”，其实不够准确。

在 vLLM 官方语义里：

- DP rank 是独立 core engine；
- 每个 DP rank 有独立 KV cache；
- 但在 MoE + DP 场景里，这些 rank 又不是完全独立的，因为 expert 层需要同步。

官方文档甚至明确说：

- 当任意一个 DP rank 上还有请求在跑时；
- 其他空闲 rank 也要做 dummy forward；
- 这样才能保证 expert 层 across all ranks 在每次 forward 时保持对齐。

所以这里的 `DP attention` 更准确地说，是：

**attention 权重和 attention KV cache 以 DP 方式分布，而 expert 层在更大的 `DP × TP` 或 `EP` 组上协同工作。**

这和传统“完全无通信的多副本 DP”不是一回事。

## 4. 为什么 MLA / MQA / GQA 会改变最优并行策略

这是原文最值钱的地方，也是最容易被忽略的地方。

### 4.1 TP 对 attention 的一个隐含前提：KV heads 足够多

普通 multi-head attention 下，KV cache 可以比较自然地按 head 维切分。

但在：

- MLA
- MQA
- GQA（尤其 kv head 很少时）

这个前提就变弱了。

因为这类结构的共同点是：

**KV heads 比 query heads 少得多。**

于是当你继续增大 TP 时，会发生一件很不划算的事：

- 线性层虽然被切开了；
- 但 KV cache 不能按同样方式继续均匀扩展；
- 最终会出现 KV cache duplication。

### 4.2 vLLM 官方对这个问题说得很直接

在 vLLM 的 Context Parallel 文档里，官方明确给出：

- decode 阶段，先沿 `H`（kv-head 数）维切 KV cache，这就是普通 TP；
- 但当 `tp_size > H` 时，KV cache 会出现 `tp_size / H` 倍复制；
- 这就是 TP 在 MLA / 少 kv-head 架构上的核心浪费。

官方给的例子非常直接：

- DeepSeek-R1 开启 MLA 时，`kv-head = 1`
- 如果单机用 `-tp 8`
- 会造成 `8x KV cache duplication`

这几乎就是为什么文章会强调 `DP attention` 的根本原因：

**attention 如果继续大 TP，可能并没有真正把 KV cache 摊开，反而是在复制它。**

## 5. 这时为什么 DP attention 往往比大 TP 更合理

如果 attention 走 DP：

- 每个 DP rank 都是一套独立 attention 副本；
- 每个 rank 管自己的请求和 KV cache；
- 不需要为了层内 attention 计算每层做 TP 同步；
- 更重要的是，不会因为 `tp_size > kv_heads` 而无意义复制 KV cache。

这在 online serving 里很关键，因为实际瓶颈通常不是“单层 attention FLOPs 不够快”，而是：

- KV cache 放不下；
- KV cache 带宽压力大；
- decode token latency 太高；
- TP 引入了太多 `all_reduce`。

因此对 MLA / MQA 模型，attention 常常更适合：

- `DP`
- 或很小的 `TP`

而不是一路把 TP 拉大。

## 6. 但只说 DP attention 还不够，还要看 DCP

这是原文没有完全展开、但现在更值得关注的一点。

### 6.1 DCP 是什么

vLLM 官方把它叫 `Decode Context Parallel`。

它解决的问题不是“权重放不下”，而是：

**decode 阶段 KV cache 太大，而且 plain TP 会因为 kv-head 太少而复制 KV cache。**

做法是：

- TP 先沿 kv-head 维切 KV cache；
- DCP 再沿 token / context 长度维继续切 KV cache；
- 从而减少 KV duplication。

### 6.2 DCP 为什么对 MLA 特别重要

官方给出的案例：

- DeepSeek-R1 开 MLA 时，`kv-head = 1`
- `-tp 8` 会导致 `8x` KV duplication
- 可以再加 `-dcp 8`，把 duplication 降掉

对 Qwen3-235B-A22B，官方也给了类似例子：

- `kv-head = 4`
- `-tp 8` 会有 `2x` duplication
- 可以加 `-dcp 2` 去掉 duplication

所以从今天的 vLLM 视角看，文章里的“DP attention”判断可以再向前推进一步：

**MLA / GQA 模型的 attention 并行，不应该只在 DP 和 TP 二选一，还要把 DCP 一起纳入设计。**

## 7. TP、DP、PP、EP、DCP 的通信代价分别是什么

这部分可以把文章里的直觉系统化。

| 并行方式 | 切分对象 | 主通信 | 典型瓶颈 |
| --- | --- | --- | --- |
| TP | 层内权重 / hidden 维 | `all_reduce` / `all_gather` / `reduce_scatter` | 每层同步，decode latency 敏感 |
| DP | 模型副本 / 请求 | 理想情况下近似无主路径模型并行通信；MoE 下仍有协调与 expert 同步 | 权重复制、KV cache 分散、单请求加速有限 |
| PP | 层 | stage 间 `send/recv` | pipeline bubble、时延变长 |
| EP | experts | token dispatch / combine，通常是 `all_to_all` | token 重排、A2A、负载均衡 |
| DCP | decode KV cache 的 token 维 | attention 阶段额外通信 | 降 duplication，但增加通信复杂度 |

这里面最重要的结构性区别是：

- TP 的通信是“每层同步”
- EP 的通信是“路由 token 到专家”
- DCP 的通信是“为了把 decode KV cache 真正分开”

三者不是一类问题，不能混着看。

## 8. 为什么 MoE 部署经常更偏向 EP，而不是继续加 TP

从 vLLM 官方文档和 DeepSeek 公开 profile 数据看，有一个越来越清楚的趋势：

- MoE 在线部署经常会把 TP 控制得很小；
- 然后用更强的 EP 来承载 expert；
- 再用负载均衡与 overlap 去解决 EP 的通信成本。

DeepSeek 公开的 `profile-data` 中：

- prefill：`EP32, TP1`
- decode：`EP128, TP1`

这当然不是所有系统的唯一正确答案，但它非常能说明一个现实：

**大规模 MoE 在线推理的重点常常不是把 TP 拉大，而是把 expert 侧通信与计算组织好。**

## 9. vLLM 官方文档对 EP 的几个关键提醒

### 9.1 EP 通常和 DP 一起用更合适

vLLM 官方在 Expert Parallel Deployment 文档里明确写到：

- EP 通常和 DP 配合使用更高效；
- attention 和 expert 可以采用不同并行方式；
- 例如 `TP=1, DP=8, EP=8` 时，attention 权重在 DP rank 上复制，expert 在 8 张卡上分布。

这和文章的主张高度一致。

### 9.2 开了 EP 后，attention 和 expert 会走不同逻辑

官方明确区分：

- `TP=1` 时，attention 权重在所有 DP rank 上复制；
- `TP>1` 时，attention 在每个 DP 组内部再做 TP；
- 开启 EP 后，expert 不再像 dense 模型那样继续走 `TP × DP` 的 TP 语义，而是切换成 EP。

所以更准确的理解是：

**vLLM 允许 attention 和 experts 分别使用不同并行策略。**

这正是 MoE 部署和 dense 模型部署的核心区别之一。

### 9.3 EP 真正难的不是“能不能分开”，而是性能细节

vLLM 官方文档还额外强调了几个重要点：

- `DeepEP` 的 `high_throughput` / `low_latency` 模式各有工作点；
- 可以用 `--enable-dbo` 做 Dual Batch Overlap；
- 可以用 `--enable-eplb` 做 Expert Parallel Load Balancer；
- 可以用冗余专家 `num_redundant_experts` 对热点专家做复制。

这意味着：

**EP 的价值不只是“把专家分散到多卡”，而是把调度、通信、重叠、负载均衡一起做起来。**

## 10. 一套更实用的选型方法

如果你是从工程部署角度出发，而不是从“理论并行名词”出发，可以按下面这个顺序判断。

### 10.1 dense 模型

- 单卡放得下：先单卡；
- 单机多卡、互联好：优先 TP；
- 没有 NVLink 或 GPU 数不规则：考虑 PP；
- 吞吐不够：再扩 DP 副本。

### 10.2 普通 MoE，但不是明显 MLA / 少 kv-head 模型

- experts 优先 EP；
- attention 保持小 TP 或 DP；
- 高吞吐场景下优先 `DP + EP`；
- 单请求延迟特别敏感时，再考虑少量 TP。

### 10.3 MLA / MQA / 少 kv-head 的 MoE 模型

这里建议直接把思路改成：

1. 先问：attention 做大 TP 会不会导致 KV duplication？
2. 再问：attention 是否更适合 DP？
3. 最后问：是否还需要 `DCP` 来进一步降低 KV duplication？

更直白一点：

- 如果 `kv_heads` 很少，大 TP 往往不是第一选择；
- 如果 decode KV cache 压力大，DCP 不是可选项，而是重点优化项；
- experts 侧仍然优先看 EP，而不是让 experts 继续吃大 TP。

## 11. 结合实际模型的经验判断

### 11.1 DeepSeek-R1 / DeepSeek-V3 这类 MLA 系模型

更合理的优先级通常是：

- `attention = DP 或小 TP`
- `experts = EP`
- `decode = 再看 DCP`

而不是上来就把 attention TP 拉满。

### 11.2 Qwen3-30B-A3B / Qwen3-235B-A22B 这类 MoE 模型

如果是 Qwen3-235B-A22B，vLLM 官方已经明确给出：

- `kv-head = 4`
- `tp=8` 时会有 `2x` KV duplication
- 可加 `dcp=2` 去掉 duplication

这说明 Qwen 系 MoE 部署也应该把：

- TP
- DP
- EP
- DCP

放在一起看，而不是只讨论 TP 和 EP。

## 12. 最后的总结

把原文和官方文档合起来看，可以得到一个更完整的判断：

### 12.1 原文最核心的洞见

- 对 MoE，attention 和 experts 应该分开设计并行策略；
- 对 MLA / 少 kv-head 架构，大 TP 往往会让 attention 侧的 KV cache 变得很不划算；
- 所以 `DP attention + EP experts` 是一条非常有现实意义的路线。

### 12.2 官方文档补上的关键事实

- vLLM 里 DP attention 不是完全独立的“传统 DP”，因为 MoE 下 expert 层仍需对齐与同步；
- 不开 EP 时，expert 默认还是 `DP × TP` 大组上的 TP 语义；
- 开 EP 后，expert 才切换成真正的 EP；
- MLA / GQA 模型要同时考虑 `DCP`，否则 KV duplication 仍然会很严重。

### 12.3 更工程化的一句话

**MoE 部署不是“TP、DP、PP、EP 四选一”，而是 attention、experts、KV cache 三个子问题分别选最合适的并行策略。**

对于今天很多带 MLA / 少 kv-head 特征的 MoE 模型，更合理的默认思路通常是：

- attention：`DP` 或小 `TP`
- experts：`EP`
- decode KV cache：必要时加 `DCP`

## 参考资料

- 知乎文章
  - https://zhuanlan.zhihu.com/p/1984584945700738885
- vLLM Data Parallel Deployment
  - https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/
- vLLM Expert Parallel Deployment
  - https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- vLLM Parallelism and Scaling
  - https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
- vLLM Context Parallel Deployment
  - https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/
- DeepSeek profile-data
  - https://github.com/deepseek-ai/profile-data
