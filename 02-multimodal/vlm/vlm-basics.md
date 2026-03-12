# VLM 多模态大模型基础

---

## 架构组成

多模态大模型通常包含 5 个部分：
1. **模态编码器** (Vision Encoder)
2. **输入投影器** (Input Projector)
3. **语言模型 Backbone** (LLM)
4. **输出投影器** (Output Projector)
5. **模态生成器** (Modality Generator)

**训练技巧**：可以只优化 output projector，其他部分保持冻结，以减小开销。

---

## 视觉编码器

### ViT (Vision Transformer)
将图片分成小 patch，每个 patch 作为序列元素输入 Transformer。

### Swin Transformer
将自注意力限制在局部窗口，同时允许跨窗口连接，提高效率。

### 常用编码器
| 编码器 | 特点 |
|--------|------|
| **NFNet-F6** | 无归一化 ResNet，增强数据上 SOTA |
| **ViT** | Transformer 处理图像 patch |
| **CLIP-ViT** | 图文对比学习优化 |
| **EVA-CLIP** | 大规模 CLIP 训练稳定化 |

---

## CLIP

OpenAI 2021 年推出的对比学习模型，利用文本信息训练 zero-shot 视觉模型，迁移能力强。

**训练方式**：
- 第一阶段：大规模文本数据训练 Transformer
- 第二阶段：图文对数据训练整个模型，学习图文匹配

**关键**：成对的文本图像视为正样本，其他为负样本。

---

## BLIP-2

增强多模态模型的图像理解与推理能力。

**架构**：ViT + Q-Former + LLM
- 有什么（ViT）
- 问的是什么（Q-Former, LLM）
- 找答案（LLM）

**训练**：使用轻量级 Q-Former 对齐图像和文本空间，分两阶段预训练。

---

## Stable Diffusion

**特点**：在潜在空间（latent space）扩散，而非高维图像空间。

**架构**：
- 使用 VAE 压缩图像到 latent space
- CLIP ViT-L/14 作为文本编码器
- 860M 参数 UNet + 123M 文本编码器

**训练策略**：先在 256×256 预训练，再在 512×512 微调。

---

## 语音语言模型

**核心思想**：将连续语音离散化成 token，并入 LLM 词表。

**语音 vs 文本**：
- 文本：离散、序列短、信息密度高
- 语音：连续、序列长、信息密度低（可压缩）

**语音离散化**：
- 语义建模：wav2vec 2.0、HuBERT、w2v-BERT（MLM 方法）
- 离散+压缩：RVQ-VAE，包含层次化语义和声学信息

---

## 评估指标

### FID (Fréchet Inception Distance)
衡量生成图像与真实图像的相似度。通过 Inception 模型提取特征向量，计算两者距离。**值越小越好**。

---

## 视频生成流程

1. **图像生成低分辨率视频**（如 600×600×32 帧）
   - 运动模块：生成帧间运动效果
   - 扩散模型：图像特征提取和合成
   - 参考图像特征提取器：提取关键特征

2. **视频超分辨率**（600→1048）
   - 继续使用运动模块、扩散模型增强质量

3. **帧插值**（32→94 帧）
   - 平滑帧间过渡，生成高帧率视频

---

## 训练经验

- 固定参数时，语言模型主干质量对最终 VLM 性能影响大于视觉模型
- 单模态模块冻结时，交叉注意结构更优；解冻训练后，完全自回归架构更佳
- 大 batch size 时，BF16 比 FP16 更稳定

---

## 参考

- [HuggingFace 多模态实验](https://mp.weixin.qq.com/s/JnXU8wuyGyWgf7jjMtnFuw)
- [多模态 LLM SOTA 模型](https://mp.weixin.qq.com/s/Bvp0gBOkxHGH-XXNdrvOqg)
- [GPT-4o 语音技术解读](https://mp.weixin.qq.com/s/RKSrystS53HN4C0POr6PYQ)
