主要贡献：Cambrian-10M和Cambrian-7M instruction tuning数据集。

Cambrian项目中对data engine做了详尽的介绍。



在该项目中，作者对数据占比做了实验，以下是作者选出的最优的数据占比分布。

![data_ratio](D:\myProject\llm-cs-learning\MLLM\images\data_ratio.png)

此外作者还提到了"Answer Machine" Phenomenon（具体来说就是模型倾向于回答更短的答案并且表现得像问答机器人一样而不像人类），作者说可以通过在训练期间添加system prompt来缓解该问题。



作者在文中着重要强调的另外一点就是当前的MLLMs大多是为了刷榜，虽然在benchmark上有很好的表现但是缺乏实用能力。因此引出了未来建立**更加科学、多元的评估手段**的必要性。

作者另外在discussion中强调的一点为了不改变原意我进行了摘取，如下：

> One promising direction for post-training alignment is through reinforcement learning rather than supervised fine-tuning. 



# 训练

作者强调了两阶段训练的重要性：

- **Visual Connector Training**: 训练一个空间视觉聚合器 (SVA)，将冻结的预训练视觉编码器连接到冻结的LLM上。
- **Instruction Tuning**: 联合训练visual connector和LLM

































