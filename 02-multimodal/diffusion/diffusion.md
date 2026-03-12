# Diffusion Model 笔记

---

## 基础概念

扩散模型通过逐步添加噪声到数据，然后学习逆转这个过程来生成数据：
- **正向过程**：逐渐向数据添加噪声，直到完全变成噪声
- **逆向过程**：从噪声逐步恢复原始数据

DDPM 采样速度很慢，后续工作主要围绕加速采样展开。

### 频域视角

Diffusion 在频域的行为近似于自回归模型：
- 向图像添加噪声 ≈ 低通滤波（噪声掩盖高频部分）
- 前向过程的不同时间步对应频率分解
- Denoising process 实际上是不断预测高频部分

---

## VAE 与自编码器

### AutoEncoder
通过压缩数据并尝试重构来捕捉关键特征。Encoder 类似 PCA 降维。

### Denoising AutoEncoder
在输入中引入噪声，训练网络恢复原始数据，学习更鲁棒的表示。

### Sparse Autoencoder
在隐藏层应用稀疏约束，防止过拟合。

### VAE (Variational Autoencoder)
将输入映射到概率分布而非固定向量，不仅重构数据，还能生成新数据。核心是学习输入数据的概率分布特性。

---

## DDIM：加速采样

DDIM 对 DDPM 的采样过程做了加速，训练过程完全相同。

**核心思想**：将 DDPM 的马尔可夫链泛化到非马尔可夫过程，对应确定性的生成路径，大大加快采样速度。

**特点**：
- 采样过程是确定性的，$x_0$ 仅由 $x_T$ 决定
- 可以直接在 latent space 进行插值控制
- 具有类似 Neural ODE 和 Normalizing Flows 的性质

**Implicit 含义**：$x_0$ 由 $x_T$ 唯一确定，samples 由 latent variables 决定。

参考：[DDIM 简明讲解与 PyTorch 实现](https://zhouyifan.net/2023/07/07/20230702-DDIM/)

---

## Classifier-Free Guidance (CFG)

**问题**：如何在生成多样性和质量之间平衡？

**做法**：同时训练条件和无条件去噪模型（同一网络），训练时随机将 condition 置为 empty token。

**采样方式**：条件分数估计和无条件分数估计的线性加权和

$$\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, c) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

**优势**：简洁易实现，无需重新训练分类器。

---

## DiT：Diffusion Transformer

将 Transformer 用于扩散模型的噪声预测网络。

### 条件注入方式
1. 直接拼接到序列
2. 单独序列做 Cross Attention
3. Adaptive Layer Normalization（外部 embedding 生成 LN 参数）
4. adaLN + 缩放因子调节 MHA 输出

### 变体

| 模型 | 特点 |
|------|------|
| **U-ViT** | 借鉴 U-Net 残差思路，每层输出通过 skip connection 加到更深层 |
| **MDT** | 引入 Mask Latent Modeling，加速收敛，强行学习 patch 之间关系 |
| **Diffit** | U-Net + Transformer 结合，增加 downsample/upsample 实现层次建模，采用 Time-dependent Self-Attention |

---

## sCM：一致性模型

OpenAI 提出的简化连续时间一致性模型，只需 **2 步采样** 即可获得与前沿扩散模型相当的效果，带来约 **50 倍采样加速**。

**核心**：一致性模型目标是一步将噪声直接转换为无噪声样本，从预训练扩散模型中蒸馏知识。

---

## Flow Matching

Flow Matching 训练目标是**预测速度**，利用这个速度将样本从一个状态逐步"移动"到目标状态，就像分布从一个状态流动（flow）到另一个状态。

推理阶段：从标准正态分布采样 $X_0$，用模型估计 $\frac{dX_t}{dt}$，基于常微分方程求解器（如一阶欧拉法）计算 $X_1$。

**优势**：自然满足扩散过程终点信噪比为零，比传统扩散损失更稳定。

---

## 参考

- [Lilian Weng: Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Flow Matching: The Next Frontier in Generative AI](https://medium.com/@rsiddhant73/flow-matching-the-next-frontier-in-generative-ai-7cf02ebbe859)
