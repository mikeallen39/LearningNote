1. gelu激活函数



> https://zhuanlan.zhihu.com/p/654897296

**预训练模型要如何改写？**

**模型的训练(train)和验证(validate)具体逻辑是怎么样的？**

为什么qkv中只对q和v进行低秩适配？



```python
optimizer_step(_lm_loss/(args.grad_acc), optimizer, model, scheduler, args,is_update=is_update)
```

在模型训练的过程中，**我们有时不会一个step更新一次梯度，而是累积若干次step（即代码中的grac_acc值）再做一次梯度更新**，这样做的原因是：

- 减少模型的更新频率，一定程度上加速训练速度
- 节省显存。



# LoRA论文

## Introduction

首先提出了该方法诞生的背景，即LLM的全量微调由于参数量很大存在许多困难和限制因素。

然后介绍了现有的方法的局限性，例如只微调部分参数/额外学习一个新的adapter module，这些方法不仅会引入推理延迟，而且性能往往低于fine-tuning的base model。

接下来提出了该idea的来源，受到了Li et al. (2018a); Aghajanyan et al. (2020) 的启发，并且做出假设、提出method。

再接下来总结advantages。

最后简要介绍论文中的术语和约定。

## PROBLEM STATEMENT