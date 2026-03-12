**flash attention相比于之前的工作聚焦于减少FLOPS，本工作主要是为了降低MAC（memory access cost）**



**矩阵的乘法可以拆分成多个分块小矩阵的乘法再组合起来。**

相对于原始attention的3-pass算法，online-softmax是2-pass算法，而FlashAttention是1-pass算法。

[flashattention手稿](./flashattn.pdf)

what is safe softmax? ：就是在实现过程中先把每一个数值减去其中的最大值再计算exp值，这样可以避免数值溢出的问题。



Flash Attention 的动机是尽可能避免大尺寸的注意力权重矩阵在 HBM 和 SRAM 之间的换入换出。具体方法包含两个部分：**tiling** 和 **recomputation**。

tiling 的基本思路：不直接对整个输入序列计算注意力，而是将其分为多个较小的块，逐个对这些块进行计算，增量式地进行 softmax 的规约。规约过程中只需要更新某些中间变量，**不需要计算整个注意力权重矩阵**。

recomputation 的基本思路：基于 tiling 技巧，在反向传播过程中不保留整个注意力权重矩阵，而是只保留前向过程中 tiling 的某些中间变量，然后在反向传播过程中重新计算注意力权重矩阵。recomputation 可以看作是**一种基于 tiling 的特殊的 gradient checkpointing**，因此后文主要介绍 tiling，想进一步了解 recomputation 的读者可以翻阅 Flash Attention 原文。

得益于上述技巧，Flash Attention 可以同时做到又快（运算速度快）又省（节省显存）。



Flash Attention 的特点在于尽量减少 GPU 的 HBM 和片上 SRAM 之间的数据交换，从而达到加速运算以及节省显存的目的。

Flash Attention 的核心方法是 tiling 和 recomputation。其中 tiling 递推式地计算 softmax，避免了计算整个注意力权重矩阵，而 recomputation 则基于前向运算中的 tiling 保存的某些中间变量，在反向传播时重新计算注意力权重矩阵。



参考：https://zhuanlan.zhihu.com/p/668888063

https://www.zhihu.com/question/611236756/answer/3132304304