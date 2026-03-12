# 分布式训练加速

LLM 训练加速主要从三个方面优化：
1. **分布式并行训练**：数据并行、模型并行、流水线并行
2. **算法模型架构**：如 MoE
3. **内存和计算优化**：梯度检查点、混合精度、算子融合等

---

## 并行策略

### 数据并行 (DDP)
每个 GPU 保存完整模型副本，梯度通过 All-Reduce 同步。

### 模型并行
- **张量并行 (TP)**：切分模型层内的张量
- **流水线并行 (PP)**：切分模型层，不同 GPU 处理不同层

### FSDP vs DDP
- **DDP**：每个 GPU 保存完整模型参数
- **FSDP**：参数分片，每个 GPU 只保存部分参数

---

## ZeRO 优化

ZeRO (Zero Redundancy Optimizer) 通过分片消除冗余：

| 级别 | 分片内容 | 内存节省 |
|------|----------|----------|
| ZeRO-1 | Optimizer States | ~4x |
| ZeRO-2 | + Gradients | ~8x |
| ZeRO-3 | + Parameters | ~N倍 (N=GPU数) |

### ZeRO-1
每个 rank 只更新优化器分片部分，更新后通过 All-Gather 同步参数。
- **适合**：Adam 优化器 + FP16 混合精度
- **不适合**：SGD（参数少，通信成本高）

### ZeRO-2
Backward 时 gradients 通过 Reduce 到对应 rank，取代 All-Reduce，减少通信开销。

### ZeRO-3
参数也分片，需要时通过 All-Gather 获取。

---

## 技术演进

| 技术 | 说明 |
|------|------|
| Parameter Server | 李沐提出，中心化参数服务器 |
| Ring All-Reduce | 去中心化，环形通信 |
| GPipe | Google 针对 PP 的优化 |
| PipeDream | Microsoft 针对 PP 的优化 |

---

## 参考

- [DeepSpeed ZeRO 论文](https://arxiv.org/abs/1910.02054)
