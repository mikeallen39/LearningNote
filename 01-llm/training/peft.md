# PEFT 高效参数微调

PEFT（Parameter-Efficient Fine-Tuning）通过只更新少量参数来实现模型微调，大幅降低训练成本。

## 分类

| 类型 | 说明 | 代表方法 |
|------|------|----------|
| 加性微调 | 添加可学习模块 | Adapter、Prefix Tuning |
| 选择性微调 | 只更新部分参数 | BitFit |
| 重参数化微调 | 低秩表示训练，推理时合并 | LoRA |
| 混合微调 | 结合多种方法优势 | - |

---

## LoRA

核心思想：在关键层添加低秩矩阵，而非改变整个模型结构。

```
W' = W + BA  (W: d×d, B: d×r, A: r×d, r << d)
```

**初始化**：矩阵 A 初始化为随机正态分布，矩阵 B 初始化为零矩阵，确保训练初始不改变原始输出。

**训练经验**：
- 建议在所有层应用 LoRA，而不仅仅是 K/V 矩阵
- α 值通常设为 rank 的 2 倍

### LoRA 变体

| 方法 | 核心改进 |
|------|----------|
| **LoRA+** | 为 A/B 设置不同学习率（B 设为 A 的 16 倍），加速收敛 |
| **VeRA** | 共享随机权重，只训练缩放向量，参数减少 97% |
| **AdaLoRA** | 根据层重要性动态调整秩 r |
| **DoRA** | 分解为幅度向量 + 方向向量，只微调方向，收敛更快 |
| **LISA** | 根据层重要性分配微调资源 |

## QLoRA

在 LoRA 基础上引入量化技术，在 48GB 显存上微调 65B 模型：
1. **NF4 数据类型**：4 位标准浮点数
2. **双重量化**：对参数和量化常数分别量化
3. **分页优化器**：显存过高时用内存替代

---

## Adapter Tuning

在 Transformer Block 中插入小型 MLP 模块（降维→升维），冻结预训练参数，只训练 Adapter。

```
Transformer Block → Adapter (降维) → Adapter (升维) → Output
```

## Prefix Tuning

为 LM 添加可训练的任务特定前缀（连续可微 token），相当于可训练的 prompt engineering。

## Prompt Tuning

与 Prefix Tuning 的区别：
- Prompt Tuning 只在输入层添加可训练 token
- 模型规模超过 100 亿参数时，效果可媲美全量微调

## P-Tuning

优化输入的 prompt，本质上是在优化词嵌入向量。

## BitFit

最简单的 PEFT 方法：只更新 bias 参数或部分 bias 参数。

---

## 参考

- [大模型微调方法总结](https://zhuanlan.zhihu.com/p/636481171)
- [大模型训练之微调篇](https://zhuanlan.zhihu.com/p/625896377)
