用于深度学习模型在Nvidia GPU上的高性能推理加速。



作者提出Power efficiency and speed of response是两个关键的指标。深度学习一般是两阶段，build和inference

在build阶段，TensorRT 对网络配置进行优化，并生成用于计算深度神经网络前向传播的优化计划。该计划是一个优化的目标代码，可以序列化并存储在内存或磁盘上。



TensorRT-LLM 融合了综合的众多优化（kernel fusion 、paged attention等）以及更多功能，同时提供了用于定义和构建新模型的直观 Python API。

TensorRT-LLM 封装了 TensorRT 的深度学习编译器，并包含最新的优化内核，用于实现 FlashAttention 和用于 LLM 执行的屏蔽多头注意力 (MHA)。

> Highlights of TensorRT-LLM include the following:
>
> - Support for LLMs such as Llama 1 and 2, ChatGLM, Falcon, MPT, Baichuan, and Starcoder
> - In-flight batching and paged attention
> - Multi-GPU multi-node (MGMN) inference
> - NVIDIA Hopper transformer engine with FP8
> - Support for NVIDIA Ampere architecture, NVIDIA Ada Lovelace architecture, and NVIDIA Hopper GPUs
> - Native Windows support (beta)





博客文章

https://developer.nvidia.com/blog/deploying-deep-learning-nvidia-tensorrt/

https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/