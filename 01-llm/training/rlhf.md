# RLHF 强化学习人类反馈

> "让 AI 用随机路径去尝试新任务，如果效果超出预期，就更新权重让其多记住成功的实践。" —— Ilya

---

## 三阶段流程

### 1. SFT Phase
基于预训练模型，在高质量下游任务数据上训练，得到 $\pi^{SFT}$。

### 2. Reward Modelling Phase
训练奖励模型学习人类偏好。

### 3. RL Fine-tuning Phase
优化目标：
$$\max_{\pi_\theta} \mathbb{E}[r_\phi(x, y)] - \beta \mathbb{D}_{KL}[\pi_\theta \parallel \pi_{\text{ref}}]$$

- 第一项：对齐人类偏好（最大化奖励）
- 第二项：KL 散度约束，避免模型过度优化失去多样性

奖励函数构建：
$$r(x,y) = r_\phi(x,y) - \beta(\log \pi_\theta - \log \pi_{\text{ref}})$$

---

## 核心算法

### PPO
信任区域优化算法，理论上保证策略学习的性能单调性。

### DPO
直接使用偏好进行策略优化，省去训练 reward model 步骤。

### GRPO / RLOO
去掉 critic model，节省显存：
- 从每个 prompt 采样 N 个回答
- 优势估计 = 当前样本 reward - 其他样本平均 reward
- GRPO 额外进行标准化处理

**缺点**：标准化可能放大微小差距；去掉 critic 可能降低训练效率。

---

## Process Reward Model (PRM)

传统 reward model 只奖励最终结果，PRM 对推理的每一步打分，更接近人类学习方式。

---

## o1 与 Self-Play

o1 的核心技术：
- **STaR (Self-Taught Reasoner)**：自我训练推理能力
- **Self-Play**：LLM 同时扮演 agent 和 reward model
- **MCTS**：结合策略和价值评估最优行动

训练大模型快速找到通向正确答案的 CoT 路径。

---

## 实践经验

- 训练 reward model 约需 **50k** 标记好的偏好数据
- PPO 训练中 critic model 可能比 actor model 收敛更快
- Trust region optimization 保证策略学习单调性

---

## 参考

- [HuggingFace RLHF 简介](https://huggingface.co/blog/rlhf)
- [HuggingFace PPO 教程](https://huggingface.co/blog/deep-rl-ppo)
- [大模型偏好对齐-DPO](https://mp.weixin.qq.com/s/-wUwacSz7D8E0sRdDqB2Pg)
