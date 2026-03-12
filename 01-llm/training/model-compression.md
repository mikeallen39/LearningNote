# 模型压缩

---

## 知识蒸馏

将大模型（教师）的知识迁移到小模型（学生）。

### NVIDIA Llama-3.1-Minitron 4B

结合结构权重剪枝和知识蒸馏：
1. **评估重要性**：评估不同 layer/head/channel 的重要性
2. **排序**：按重要性排序
3. **剪枝**：进行模型剪枝
4. **再训练**：通过蒸馏重新训练

**蒸馏范式**：
- Classical Knowledge Distillation
- SDG Fine-tuning

参考：[NVIDIA: How to Prune and Distill Llama-3.1-8B](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/)
