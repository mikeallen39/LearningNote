# Ascend Profiling Analysis Notes

这篇文档记录在昇腾环境下，如何对一份推理 profiling 数据做初步分析。内容以 `msprof`、`msprof-analyze`、`MindStudio Insight` 为主。

## 1. 工具关系

先区分 3 个常见工具：

- `msprof`：负责采集和解析 profiling 数据。
- `msprof-analyze`：负责做命令行聚合分析，尤其适合 cluster / 多卡场景。
- `MindStudio Insight`：负责图形化展示 profiling 结果，本质上是把 profiling 数据和分析结果可视化。

可以把三者理解成：

1. `msprof` 产出原始 profiling 数据。
2. `msprof --analyze=on` 或 `msprof-analyze` 进一步生成统计结果。
3. `MindStudio Insight` 用 GUI 方式查看时间线、通信、算子、内存和集群摘要。

如果是第一次接触，建议先用 CSV 和数据库做“静态分析”，再用 `MindStudio Insight` 看时间线和重叠关系。

## 2. 一份 profiling 目录通常长什么样

以脱敏后的目录为例：

```text
profiling_case_x/
├── PROF_xxx_rank0/
│   ├── host/
│   │   ├── sample.json
│   │   └── sqlite/
│   │       └── hccl.db
│   ├── device_0/
│   │   └── sqlite/
│   └── mindstudio_profiler_output/
│       ├── api_statistic_xxx.csv
│       ├── communication_statistic_xxx.csv
│       ├── op_statistic_xxx.csv
│       ├── op_summary_xxx.csv
│       └── task_time_xxx.csv
├── PROF_xxx_rank1/
├── PROF_xxx_rank2/
└── PROF_xxx_rank3/
```

最常用的是下面这些文件：

- `sample.json`
- `task_time.csv`
- `communication_statistic.csv`
- `op_statistic.csv`
- `op_summary.csv`
- `api_statistic.csv`
- `host/sqlite/hccl.db`

## 3. 新手推荐分析顺序

不要一上来盯某个 kernel。更稳妥的顺序是：

1. 先看任务背景。
2. 再看大类瓶颈是算、传还是等。
3. 再看最重的通信类型。
4. 再看最重的计算算子。
5. 最后下钻到通信链路和时间线。

对应文件分别是：

1. `sample.json`
2. `task_time.csv`
3. `communication_statistic.csv`
4. `op_statistic.csv`
5. `hccl.db` + `MindStudio Insight Timeline`

## 4. 字段速查

### 4.1 sample.json

作用：看这次 profiling 到底采了什么任务。

重点字段：

- `app_parameters`：启动参数，最值得先看。
- `profLevel`：采集粒度。
- `ai_core_metrics`：采集的 AI Core 指标。
- `devices`：当前目录对应的 device / rank。

你通常能从这里先确认：

- 是训练还是推理。
- 是单卡还是多卡。
- 大概的输入规模。
- 是否采了足够的信息。

### 4.2 task_time.csv

作用：先做粗粒度分账。

重点字段：

- `kernel_name`
- `kernel_type`
- `stream_id`
- `task_time(us)`

常见 `kernel_type` 的含义：

- `AI_CORE`：主要计算任务。
- `MEMCPY_ASYNC`：搬运和通信相关任务。
- `REDUCE_ASYNC_V2`：归约通信。
- `EVENT_WAIT`：等待。

怎么看：

- `AI_CORE` 占大头，通常偏计算瓶颈。
- `MEMCPY_ASYNC` / `REDUCE_ASYNC_V2` 占大头，通常偏通信瓶颈。
- `EVENT_WAIT` 很大，通常说明同步等待或流水不顺。

注意：

- 这里的时间相加通常不等于真实 wall time。
- 原因是 task 之间可能重叠执行，异步任务也会重复计时。
- 它更适合回答“谁最重”，不适合直接回答“程序实际跑了多少秒”。

### 4.3 communication_statistic.csv

作用：只看通信内部是谁最重。

重点字段：

- `OP Type`
- `Count`
- `Total Time(us)`
- `Avg Time(us)`
- `Ratio(%)`

怎么看：

- 先看 `Total Time(us)` 最大的是谁。
- 再看 `Count` 和 `Avg Time(us)`，判断是调用太频繁还是单次太慢。

注意：

- 这里的 `Ratio(%)` 是通信内部占比。
- 它不是整份程序时间的全局占比。

### 4.4 op_statistic.csv

作用：看单卡的计算热点。

重点字段：

- `OP Type`
- `Core Type`
- `Count`
- `Total Time(us)`
- `Avg Time(us)`
- `Ratio(%)`

怎么看：

- 按 `Total Time(us)` 排序，先找最重算子。
- 如果某个算子 `Count` 很高且 `Avg Time(us)` 也不低，通常优化收益最大。
- 如果 4 张卡的 top op 几乎一致，说明没有明显慢卡。

### 4.5 op_summary.csv

作用：看更细的单个 task 明细。

重点字段：

- `Op Name`
- `OP Type`
- `Task Duration(us)`
- `Task Wait Time(us)`
- `Input Shapes`
- `Output Shapes`
- `Block Dim`

怎么看：

- 用来解释“为什么这个 op 慢”。
- 常见原因包括 shape 太大、前后等待长、`TransData` 太多、Block 利用不理想。

### 4.6 api_statistic.csv

作用：看 Host / CANN 层 API 的时间分布。

重点字段：

- `Level`
- `API Name`
- `Time(us)`
- `Count`
- `Avg(us)`

怎么看：

- 如果 `communication` 类 API 很重，说明 host 侧也能明显看到通信压力。
- 如果很多 `Setup` 很重，说明初始化、图构建、tiling 也可能是问题。

### 4.7 hccl.db

作用：进一步确认通信算法和链路。

常看的表：

- `HCCLOP`
- `HCCLTask`

常看字段：

- `HCCLOP.op_type`
- `HCCLOP.alg_type`
- `HCCLTask.transport_type`
- `HCCLTask.link_type`
- `HCCLTask.local_rank`
- `HCCLTask.remote_rank`

怎么看：

- `op_type` 告诉你是 `allReduce`、`allGather` 还是别的。
- `alg_type` 告诉你算法，例如 ring。
- `link_type` 告诉你链路，例如 `PCIE`。
- `local_rank` / `remote_rank` 可以帮助你判断慢链路在哪两个 rank 之间。

## 5. 一份案例怎么读

下面给出一个例子：

- 场景：某个 4 卡推理任务。
- 并行方式：张量并行。
- 现象：吞吐不理想。

### 第一步：看 task_time.csv

先按 `kernel_type` 汇总，得到类似结论：

- `AI_CORE` 约 84 秒
- `MEMCPY_ASYNC` 约 345 到 350 秒
- `REDUCE_ASYNC_V2` 约 341 到 346 秒
- 4 张卡数值非常接近

这一步可以先下两个判断：

1. 主要瓶颈不是单卡算子，而是通信相关任务。
2. 没有明显慢卡，因为各卡分布很一致。

### 第二步：看 communication_statistic.csv

继续看通信内部的构成：

- `allReduce` 调用次数非常多。
- `allReduce` 总时间远高于 `allGather`。
- `allReduce` 在通信内部占比接近全部。

这一步说明：

- 通信瓶颈主要集中在 `allReduce`。
- 如果并行方式是 TP，那么怀疑方向应该放在 TP 同步开销，而不是先怀疑某个算子写坏了。

### 第三步：看 op_statistic.csv

再看计算热点：

- top op 是 attention、matmul、norm、moe routing 这一类典型 LLM 算子。
- 4 张卡的 top op 排名和时长几乎一致。

这一步说明：

- 单卡热点正常。
- 不是某一张卡计算特别慢导致其他卡等待。

### 第四步：看 hccl.db

最后看通信数据库：

- `op_type` 主要是 `allReduce`
- `alg_type` 是 ring 类算法
- `link_type` 里可以看到 `PCIE`

这一步能把结论从“通信很多”进一步收敛到：

- 这是一个典型的多卡同步成本过高案例。
- 如果机器有更快的卡间互联，但实际落到了 `PCIE`，那么这通常是重要排查点。

### 最终结论

对这个案例，更合理的结论是：

- 不是单卡 compute bound。
- 是多卡 `allReduce` 主导。
- 优化优先级应放在减少 TP 通信、提高通信重叠、检查 rank 拓扑和链路上。

## 6. 怎么判断自己看到的是哪类瓶颈

### 6.1 计算瓶颈

常见表现：

- `task_time.csv` 中 `AI_CORE` 明显最大。
- `op_statistic.csv` 中少数几个算子占比很高。
- 不同卡之间计算分布接近，但通信占比不高。

常见动作：

- 优化热点算子。
- 调整 shape / batch。
- 减少无意义的 `TransData`。

### 6.2 通信瓶颈

常见表现：

- `MEMCPY_ASYNC`、`REDUCE_ASYNC_V2` 很大。
- `communication_statistic.csv` 中某个 collective 特别重。
- `hccl.db` 可以看到大量跨卡任务。

常见动作：

- 降低 TP / 同步频率。
- 检查拓扑是否合理。
- 检查是否走了预期链路。
- 提高通信与计算重叠。

### 6.3 等待瓶颈

常见表现：

- `EVENT_WAIT` 较大。
- `Task Wait Time(us)` 较大。
- `MindStudio Insight Timeline` 里能明显看到空洞。

常见动作：

- 检查流水是否断开。
- 检查 host 端 dispatch 是否太慢。
- 检查是否有单卡拖慢全局同步。

## 7. MindStudio Insight 适合在什么时候用

如果 CSV 已经足够说明问题，命令行分析通常更快。

如果你还想看下面这些问题，`MindStudio Insight` 更合适：

- 某段时间线上到底是算、传还是空转。
- 计算和通信是否发生了重叠。
- 哪个 rank 在哪一段最慢。
- 某个 op 的前后依赖关系。

可以把它理解成：

- CSV / sqlite：适合快速定位瓶颈类型。
- `MindStudio Insight`：适合做时间线确认和可视化说明。


资料：
1. https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000068.html