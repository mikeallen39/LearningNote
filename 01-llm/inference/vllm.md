专用于LLM inference and serving，using an intelligent batching mechanism and efficient memory management.



作者认为LLM serving的速度主要受限于内存大小。输入序列中的每一个token都需要生成key和value向量，并且需要保存在GPU内存中（这就是KV Cache）来生成new tokens。

作者指着KV Cache对显存的占用是巨大的（一个输入序列在LLAMA-13B中的占用显存可能高达1.7GB）。与此同时，KV Cache的大小是动态的（随着输入序列变化而变化），因此需要高效的KV Cache管理。作者指出当时的现有系统由于**碎片**和**过度预留**而浪费了 **60% - 80%** 的内存。



PagedAttention是受到了操作系统中虚拟内存和分页的启发而设计的。

# 核心做法

PagedAttention 允许在不连续的内存空间中存储连续的键和值。具体来说，PagedAttention 将每个序列的 KV 缓存划分为块，每个块包含固定数量token的key和value向量。在注意力计算过程中，PagedAttention 内核有效地识别并获取这些block包含的向量。

根据和操作系统设计的对比，我们可以把block想象成操作系统中的page，把token想象成字节，把sequences想象成进程（processes）。

sequences的连续逻辑块（logical block）通过一个map映射到非连续的物理块（physical block）。

vllm还有一个优点，就是在并行采样（parallel sampling）的过程中可以share memory。具体做法就是将不同的logical block映射到相同的physical block上。

> To ensure safe sharing, PagedAttention keeps track of the reference counts of the physical blocks and implements the *Copy-on-Write* mechanism.





博客文章[vllm](https://blog.vllm.ai/2023/06/20/vllm.html)

