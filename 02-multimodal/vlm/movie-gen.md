[TOC]



# 摘要解读

Movie Gen 是Meta 团队近期放出来的一项工作，目前应该是只开源了tech report，期待后续开源相关代码和model weight。

关于movie gen，作者在摘要中提到

> We present Movie Gen, a cast of foundation models that generates high-quality, 1080p HD videos with different aspect ratios and synchronized audio.

非常简明扼要地表明了movie gen模型的地位，即可以生成高质量的1080p高清视频、支持不同的宽高比并能同步生成音频的foundation models。

除了text-to-video之外，meta团队通过一系列fine-tuning的操作使模型还具备了以下功能：

- 基于指令的视频编辑
- 基于用户形象（user image）的个性化视频生成（即image-to-video）

作者提到movie gen在**文生视频**、**视频个性化**、**视频编辑**、**视频到语音的生成**、**文本到语音**的生成几个任务上都达到了SOTA水平。

需要注意的是Movie Gen是一系列模型，有不同的大小尺寸。作者提到最大的视频生成模型的参数量大小是30B，最大上下文长度为73K的video tokens，对应于能够生成16秒的16fps视频；最大的语音生成模型的参数量大小是13B，能够生成48k Hz的语音。

最后，笔者再次强调Movie Gen是一系列生成模型，支持生成**图片、视频和语音**。

# intro解读

作者在intro中提到了scaling law的重要性（数据、算力、模型大小），同时指出了是使用**flow matching**进行训练。

> We find that scaling the training data, compute, and model parameters of a simple Transformer-based (Vaswani et al., 2017) model trained with Flow Matching (Lipman et al., 2023) yields high quality generative models for video or audio.

下图是movie gen应用的示例。

- 根据文本提示生成视频
- 支持在提供的参考图像中生成与角色一致的视频
- 支持根据用户提供的指令进行精确的视频编辑
- 为给定的视频生成与视频同步的音频

![image-20241021112546623](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241021112546623.png)

作者提到movie gen video在text-to-video任务上超越了之前的sota模型，包括runway gen3、lumalabs、openai sora等等；movie gen audio也超越了之前的sota模型，包括pikalabs、elevenlabs等等。

最后，团队提出了movie gen video bench、movie gen edit bench和movie gen audio bench用于未来的相关评估，完善了评估体系。在本篇tech report中，团队针对**模型架构**、**训练**、**推理**和**实验设置**都做了详尽的介绍。



# 图片视频生成解读

作者将图片视为单帧的视频，从而将text-to-image和text-to-video任务进行了统一，训练一个joint foundation model用于相关任务。之所以对两个任务进行统一处理，作者提到**图文对数据集的数量更多并且包含更加广泛的各种概念和风格**，能够使模型具备更好的性能。

在image and video joint training上，团队采用了多阶段训练的方法来提高训练效率。

1. 首先在256px的图像上训练
2. 然后在低分辨率的图像和视频上联合训练
3. 在高分辨率的图像和视频上联合训练
4. 在高质量的视频上进一步进行fine-tuning
5. 通过post-training来添加personalization和editing的能力

为了提高训练和推理的效率，作者团队训练了一个TAE（temporal autoencoder）来将原始的视频和图片压缩到一个**时空压缩**的latent space中。对于text prompt，作者团队使用预训练好的text encoder来获得文本嵌入。在训练目标上，作者团队使用flow matching training objective。

![image-20241021164228804](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241021164228804.png)

对于transformer的backbone architecture，作者团队使用了和Llama3一样的架构。

尽管作者提到能够生成1080p的视频，但是实际上原始模型只能生成768p的视频，后续通过spatial upsampler来进一步提高分辨率。

## TAE

作者提到TAE是基于vae的一种改进，将原始的pixel input映射到连续值的latent space（a continuous-valued latent  $X$）。在本文中，作者使用的压缩率都是8（在temporal、height、width维度）。

![image-20241021165640530](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241021165640530.png)

笔者在这里直接放上了论文中关于模型架构的相关描述，这是为了让读者能够更清晰地进行理解。

以下是笔者的理解：

在每次**2D空间卷积**之后，都会应用一个**1D时间卷积**。这意味着模型不仅在图像的空间维度（即高度和宽度）上进行处理，还在时间维度（即帧序列）上进行卷积操作。

**对称复制填充**：所有的时间卷积操作都使用了**对称复制填充**，即在输入的边界处复制边界值，以确保输入和输出的尺寸匹配。

**时间降采样**：通过步长为2的卷积操作进行**时间降采样**。这意味着在时间维度上，输出帧的数量将减少一半。

**时间上采样**：通过最近邻插值进行**上采样**，然后再通过卷积来细化。这是为了在时间维度上恢复更多的细节，使得帧序列的分辨率恢复到原始大小。



> 这里详细介绍一下时间降采样和上采样：
>
> 1. **时间降采样 (Temporal Downsampling)** 通过步长为2的卷积（strided convolution）
>
> - **降采样的目标**：在时间维度上减少帧数或分辨率，使得网络能够处理更长的时间序列或视频，减少计算量，同时保留重要的时间信息。
>
> - 步长卷积：卷积核在时间维度上以步长为2进行操作，这意味着每次卷积操作后跳过一个时间点。即，网络并不会对每一个时间帧进行卷积，而是跳过部分帧，从而减少输出的时间帧数。
>
>   - **卷积核**：用于提取时间上的特征，如动态变化、时间序列模式等。
>   - **步长为2**：意味着网络只保留每隔一个时间帧的数据，使得时间维度上的分辨率减半。如果原始时间序列有100帧，经过步长为2的卷积后，输出序列会减少到50帧。
>
> - **对称复制填充（symmetrical replicate padding）**：为了保持输出尺寸的一致性，在进行卷积操作时，边界的填充采用对称复制的方式，即将边界的值复制，使卷积操作不会导致边缘信息丢失。
>
> 2. **时间上采样 (Temporal Upsampling)** 通过最近邻插值 (Nearest-neighbour interpolation) 和卷积
>
> - **上采样的目标**：在时间维度上增加帧数，使网络能够恢复到更高的时间分辨率，同时通过卷积进一步调整生成的帧，确保时间序列的平滑性和一致性。
>
> - 最近邻插值
>
>   ：最近邻插值是一种简单的插值方法，通过复制最近的帧来生成新的时间帧。例如，如果一个序列的帧数是50，通过最近邻插值可以扩展到100帧，这些新增的帧是基于最近的原始帧复制生成的。
>
>   - **优点**：计算量低，易于实现。
>   - **缺点**：生成的帧可能存在突兀的过渡，因为它只是简单的复制，没有考虑数据的平滑性或连续性。
>
> - **卷积调整**：在插值生成的时间帧后，使用卷积操作来对这些新增帧进行调整，使得时间序列更加平滑，减少插值造成的不连续性。卷积可以帮助提取上下文信息，使得插值后的帧在时间上保持一致性，同时通过卷积的权重学习改善时间上的细节特征。
>
> **综合说明**
>
> 在这个过程中，时间维度上的降采样通过步长卷积使时间帧数减少，简化数据并减少计算负担；而上采样通过最近邻插值生成更多的时间帧，再通过卷积调整，使得生成的视频或时间序列在较高时间分辨率下仍然保持合理的连续性。这种技术常用于视频生成或处理任务，尤其是需要同时处理空间和时间维度的模型。

与SD3类似，作者团队在文中提到了提高latent space的通道数在重建和生成方面都能够提高模型性能，作者团队采用的通道数为16。

对于TAE中空间相关的参数，团队提出使用一个预训练的image autoencoder进行初始化，原文描述如下：

> We initialize the spatial parameters in the TAE using a pre-trained image autoencoder, and then add the temporal parameters to inflate the model as described above. 

在训练方面，作者提到是使用图像和视频进行联合训练，采用1：3的图像：视频的比例。

> we jointly train the TAE on both images and videos, in a ratio of 1 batch of images to 3 batches of videos. 

在training objective方面，作者提到[High-resolution image synthesis with latent diffusion models.](https://arxiv.org/abs/2112.10752) 这篇论文中提出的训练目标会导致生成的视频中有斑点伪影的存在，如下图所示。

![image-20241021224137714](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241021224137714.png)



作者在进一步研究之后发现模型在某些空间位置生成了**高范数**的潜在编码，这些编码在解码回图像时会导致图像中出现**斑点现象**，即在像素空间中产生异常的亮点或噪点。

研究团队推测，模型在某些空间位置生成的高范数潜在编码点是一种**捷径学习**的表现。模型通过在这些高响应位置存储全局的关键信息，避免了全面学习整个输入数据的复杂结构。换句话说，模型在这些点上找到了快速处理任务的“捷径”，而不是通过深入理解整个数据来完成任务。

作者指出尽管有研究表明使用group norm能够解决斑点问题，但为了不改变模型架构，作者选择向目标loss中添加一个项，以惩罚模型编码出远离平均值的latent values。

![image-20241021225521295](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241021225521295.png)

该项中的r是一个惩罚因子，决定了模型对潜在变量的宽容度。

### temporal tillling

这个主要是针对长时间高分辨率视频做的推理优化，通过将原始视频和latent tensor进行沿着时间维度的切块操作，然后对每一块进行编解码，最后将解码的结果进行拼接。

注意，不同的块之间是可以选择有重叠的（optional），在拼接的时候会使用混合加权来使得得到的视频更加平滑。

最后，作者提到了在他们的工作中使用了块大小（tile size）为32 raw frames or 4 latent frames，encoder中没有使用overlap，decoder中使用了16 raw frames or 2 latent frames的overlap。

按照如下公式进行融合

![image-20241022105858858](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241022105858858.png)

### Training Objective

在训练的目标函数上作者团队是使用flow matching的框架进行训练。

何为flow？为什么叫flow matching ？阅读完接下来这段话笔者相信你一定会豁然开朗。

![image-20241022111500884](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241022111500884.png)

flow matching训练的目标是**预测速度**，并利用这个速度将样本从一个状态逐步“移动”到目标状态，就像一个分布从一个状态流动（flow）到另一个状态。

![image-20241022112327915](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241022112327915.png)

如上图所示，训练的目标loss的构建其实相当简单。在时间步的采样上，作者是从logit-normal分布中进行采样。以下是logit-normal分布的定义

![image-20241023085850014](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241023085850014.png)

在推理（inference）阶段，我们首先从一个标准正态分布中进行采样$\mathbf{X_0} \sim \mathcal{N}(0, 1)$ ，然后使用模型来估算出$\frac{dX_t}{dt}$，已知前两者之后基于常微分方程求解器计算出$\mathbf{X_1}$。作者提到在常微分方程求解器中有很多配置的东西会影响到估计$\mathbf{X_1}$所花的时间和其准确率。于是，作者使用了一种非常基础的常微分方程求解方法——**一阶欧拉法**，并且根据模型的特点，将时间离散为$N$个步长来逐步进行计算。关于常微分方程求解相关的知识笔者在这里不做过多展开，希望感兴趣的读者可以自行学习。

此外，作者还提到了SNR（signal-to-noise ratio）的问题。作者说在**视频生成**任务中，选择合适的**扩散噪声调度**（噪声随时间的引入方式 noise scheduler）非常重要。传统的扩散模型不保证扩散过程的终点信噪比为零，必须进行修改，而作者的方法（Flow Matching）自然满足了这个条件。（这个地方为什么flow matching能够满足该条件笔者尚且也还不清楚）实验表明，Flow Matching 在应对不同的噪声调度时表现更稳定，且优于传统的扩散损失，因此作者选择了这种方法，理由是它既简单又高效。

### backbone architecture

由于原始输入通过TAE之后得到的latent code shape为$T \times C \times H \times W$，但transformer接收的输入是一维tokens，因此需要先对latent code使用3D卷积进行处理然后展开成一维序列。3D卷积核的形状为$k_t \times k_h \times k_w$，并使用与核大小相同的步长（stride）。

作者使用的是因式分解的可学习位置嵌入，并解释道这样能够用于在 Transformer 中处理不同尺寸、不同纵横比、以及不同视频长度的输入。特别需要提到一点：与传统方法只在第一层添加位置嵌入不同，作者将**因式分解的位置嵌入**（时间、空间等维度）添加到每一层的输入中。这样做能够有效减少生成视频或时间序列数据时的**失真和变形伪影**，特别是在**时间维度**上，模型能够更好地保持帧与帧之间的连贯性。

![image-20241023094233603](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241023094233603.png)

关于Transformer架构上，团队采用了和Llama3 几乎相同的架构，使用了RMSNorm和SwiGLU，但作者也提到了3点改进以适用于flow matching的视频生成：

- **为了结合基于文本提示嵌入 $P$ 的文本条件（text conditioning）**，我们在每个 Transformer 块的自注意力模块和前馈网络（FFN）之间添加了一个交叉注意力模块。我们利用了多个不同的文本编码器，因为它们具有互补的优势，并将它们的嵌入简单地连接成一个序列来构建$P$。
- **我们添加了自适应层归一化模块（adaptive layer norm blocks）**，用于将时间步长$t$引入 Transformer 中，这与之前DiT的工作一致。
- **我们使用了完全的双向注意力**，而不是语言模型中使用的因果注意力。

之所以要使用和Llama 3几乎完全一样的架构，作者提到这是为了更好地保证模型能够scaling。

![image-20241023095019620](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241023095019620.png)



在text encoder方面，作者团队使用了UL2、ByT5、Long-prompt MetaCLIP 这三种encoder的融合。作者将三个文本编码器的文本嵌入进行拼接，但在此之前，分别为每个嵌入添加单独的线性投影层和层归一化（LayerNorm）层，将它们投影到相同的 6144 维空间，并对嵌入进行归一化。

作者接下来提到了3种不同的text encoder的特性：

- UL2是在大量的text-only data上训练的，因此在其embedding中有很强的推理能力
- Long-prompt MetaCLIP提供prompt-level的与视觉信息相对齐的embedding
- ByT5 编码器（以字符为单位的编码器）仅用于对视觉文本进行编码，也就是说，它用于处理那些文本提示中特别要求在输出中生成具体字符or字符串的部分。

对于帧率FPS（frames per second）的控制，作者在训练中直接将其转换为文字输入，如“FPS-16”。



为了生成1080p的视频，团队使用了1个独立的spatial upsampler来将768p的视频转为1080p视频。作者提到这是为了降低生成模型的计算量。

作者将该任务视为一个video-to-video的生成任务。

原本分辨率为768像素的视频首先使用双线性插值（bilinear interpolation）方法进行上采样，将分辨率提升到HD（高清）。双线性插值是一种常用的图像缩放方法，通过插值算法计算新像素点的颜色值，使得图像在放大时尽量保持平滑和自然。采样后的高清图像接着会输入到一个图像编码器（VAE）中转换为latent representation，接着首先通过加入噪声来扩展潜在表示，接着通过训练好Transformer模型进行去噪，最后利用图像解码器生成高清的、上采样后的视频。

在具体的实现细节中，论文中提到spacial upsampler的模型架构是基于一个7B的在1024分辨率上训练的text-to-video模型进行初始化的。与TAE中的分块技巧相似，在上采样中也采用了滑动窗口的方法，窗口大小为14 frames，overlap为4 frames。但文中提到sliding window的方法会导致生成的视频在窗口边界会出现明显的不一致的情况；为了解决该问题，文中提到团队使用了MultiDiffusion方法，这是一种无需训练的优化方法可以确保由一组公共约束限制的不同生成过程之间的一致性。**特别地**，文中提到在视频去噪或生成的过程中，模型通过对**重叠帧的潜变量**进行**加权平均**，从而在去噪的每一步中促进帧与帧之间的信息交换。这种方法有助于提高视频输出的**时间一致性**，确保相邻帧之间的平滑过渡，使生成的视频更加自然流畅。

### Model Scaling and Training Efficiency

简单来说，在该部分中，文中介绍了

- 训练Movie Gen的硬件和底层设施细节
- 与SOTA LLMs的训练设置相比较
- 介绍并行训练的各种方法

在硬件设施方法，meta使用了多达**6,144个H100 GPU**，每个GPU配备**80GB HBM3高速显存**，并在Meta的**Grand Teton AI服务器平台**上进行训练。每个服务器有8个GPU通过**NVSwitch**连接，不同服务器之间通过高速**400Gbps RoCE RDMA网络**进行连接。训练任务通过Meta的全球训练调度器**MAST**来高效管理和调度。（说实话这段话笔者也不是特别了解）

与常见的causal attention不同，movie gen movie中使用了双向注意力。因果掩码（causal masking）的使用可以在计算注意力机制时带来大约2倍的速度提升，同时还能够减少峰值内存的需求。此外作者提到由于MOVIE GEN VIDEO是非自回归模型，因此并没有使用GQA的设计而是将其留给未来的工作进行探索（挖坑等后人填了）。

在模型训练方面，与大语言模型训练类似，模型训练根据context length分为了多个阶段。（这里需要提到，768分辨率的16秒16fps的视频的tokens数量大约为73K左右）

![image-20241025093334908](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241025093334908.png)

上图为作者关于训练阶段的详细介绍。

#### 分布式训练

作者提到通过**3D并行化**（3D parallelism）来支持大规模模型的训练，采用了以下并行化方式的结合：

**全分片数据并行化（Fully Sharded Data Parallelism）**：通过将模型参数在多个设备之间分片来减少内存需求。

**张量并行化（Tensor Parallelism）**：将一个张量操作分解到多个设备上，减少每个设备的计算负担。TP方法将神经网络中**线性层的权重**在多个GPU之间分片（“shard”），即沿着**列或行**来分配权重，使得每个GPU只处理一部分权重。（这就是矩阵乘法的优势，能够通过分块矩阵进行运算）TP的优点是每个参与分片的GPU的计算量和生成的中间激活值都降低到了原来的1/tp，然而缺点就是带来了不同GPU之间的通信开销（前向传播和反向传播中all reduce的开销）

原文描述如下，但笔者不太理解为什么前向传播是行并行而反向传播是列并行：

> The cost of performing such a sharding is the addition of all-reduce communication overheads in both
> the forward (row-parallel) and backward (column-parallel) passes.

**序列并行化（Sequence Parallelism）**：将序列长度分割到不同设备上，以便更高效地处理长序列输入。SP基于张量并行化（TP），使得在**序列维度上**对输入进行分片成为可能，特别适用于那些可被复制的层，并且在这些层中，每个序列元素可以独立处理。（笔者暂时不太理解这一部分）

**上下文并行化（Context Parallelism）**：上下文并行化（Context-Parallelism, CP）在序列维度上实现部分分片，专门用于序列依赖的softmax注意力操作。CP利用了一个关键点：对于任意一对源序列（context）和目标序列（query），softmax注意力只对源序列（context）有序列依赖，而不是目标序列（query）。因此，在自注意力的情况下（即输入的源序列和目标序列相同），CP可以只对K和V投影进行全收集（all-gather）来完成注意力计算，而不需要对Q、K、V都进行全收集。在反向传播时，也只需对K和V的梯度进行分散归约（reduce-scatter）。

在上下文并行化（Context-Parallelism, CP）中，由于**查询向量（Q）和键、值向量（K和V）**在行为上的不同，对CP性能的影响不仅取决于上下文长度，还取决于上下文维度的大小。

具体来说，这种不同的行为使得CP的性能和开销在不同模型中有所差异。例如：Movie Gen Video和LLaMa3的CP性能表现和开销就有所不同。这是因为LLaMa3等大型语言模型使用分组查询注意力（Grouped Query Attention, GQA），这会生成更小的K和V张量，以便在各GPU之间通信。

**FSDP**：将**模型参数**、**优化器**和**梯度**在所有数据并行的GPU上进行**分片**。在每一步训练中，FSDP会**同步地收集和分发**这些参数和梯度：

- **模型参数分片**：模型的参数被分散到多个GPU上，每个GPU只存储一部分参数，减小内存需求。
- **优化器和梯度分片**：在训练过程中，优化器和梯度的计算也在不同的GPU上分片进行，避免单个GPU内存超载。
- **同步收集和分发**：在每一步训练中，FSDP会将分散的参数和梯度收集到一起进行更新，然后将更新后的参数分发回各个GPU上。

接下来，作者进一步介绍了并行训练的实现技巧以提高训练效率。作者团队建立了一个**分析框架**，用于建模计算和通信时间。这一框架帮助他们识别出哪些**重复的激活值**需要在GPU之间通信，从而设计出高度优化的并行训练解决方案。该方案是使用**PyTorch**编写，并通过**CUDAGraphs**进行编译，达到了**强大的激活内存扩展**，并将通信时间最小化。



## 预训练

### 预训练数据

前文中也有所提到预训练的数据集包含了100M的视频-文字对和1B的图片-文字对。在图文对方面使用了和emu论文中相同的处理方法。

在视频数据方面，初始数据包含4s~2min的各种视频，通过一定的处理，最终变成4s ~ 16s的clip-prompt pairs。

![image-20241025164758387](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241025164758387.png)

数据处理的流程如上，包含3个过滤阶段：

- 视觉过滤：作者团队使用6种过滤器来过滤低质量的视频，具体来说移除width或者height小于720的视频；通过筛选宽高比将视频的比例控制为60%横屏视频（landscape）和40%竖屏视频（portrait）的组合；使用视频OCR模型移除带有过多文字的视频；使用FFmpeg 检测视频中的场景切换（即场景边界），并以此为依据将视频分割成长度在 4 到 16 秒之间的短片段；训练简单的视觉模型，通过基于帧级别的视觉美学、视觉质量、大边框和视觉效果来获得用于过滤的预测信号；删除视频开始的前几秒钟，因为视频开始通常包含不稳定的摄像机运动或过渡效果。
- 运动过滤：自动过滤掉low motion videos。使用**内部静态视频检测模型**去除无运动的视频。接着，我们根据 **VMAF 运动得分和运动矢量**识别具有“合理”运动的视频。为了去除存在频繁抖动的摄像机运动的视频，我们使用了 PySceneDetect 库中的**场景边界检测**。最后，去除带有特殊运动效果的视频，比如幻灯片视频。
- 内容过滤：为确保预训练集的多样性，我们使用**复制检测嵌入空间**中的相似性去除预训练集中的重复剪辑。作者还通过重新采样以减少主要概念的出现频率来创建我们的训练集。我们对来自视频-文本联合嵌入模型的语义嵌入进行聚类，以识别细粒度的概念簇。接着，我们合并重复簇，并按照簇大小的平方根倒数对每个合并簇中的剪辑进行采样。
- 字幕生成：作者使用 **LLaMa3-Video** 模型为视频剪辑创建准确且详细的文本提示。团队对 8B 和 70B 两种变体模型进行微调，以用于视频字幕生成任务，并使用这些模型为整个训练集的视频剪辑生成字幕。我们的训练集中 70% 使用 8B 模型生成的字幕，30% 使用 70B 模型生成的字幕。为了实现电影级摄像机运动控制，我们训练了一个**摄像机运动分类器**（camera motion classifier），该分类器可以预测 16 种摄像机运动类型之一，例如，缩小、向左平移等。作者提到将高置信度的摄像机运动预测结果放到先前生成的text caption之前。在推理时，这允许用户为视频生成指定显式的摄像机控制。
- **多阶段数据筛选**：我们根据逐步严格的视觉、运动和内容标准，筛选出满足不同阶段预训练需求的三个数据子集。首先，我们筛选了视频宽度和高度至少为 720 像素的片段视频，用于低分辨率训练。接下来，我们过滤了这一数据集，以提供宽度和高度至少为 768 像素的视频，用于高分辨率训练。最后，我们增添了新的视频以扩充高分辨率训练集。我们的高分辨率训练集中有 80% 是横向视频，20% 是纵向视频，其中至少 60% 含有人物。在筛选过程中，我们建立了一个包含 600 个与人类相关的动词和表情的分类体系，并使用这一分类体系进行零样本文本到视频的检索，以选择含有人物的视频。在内容重新采样过程中，我们保持了这些人物视频的频率。
- **基于时长和尺寸的分桶**：为适应多样化的视频长度和纵横比，我们根据纵横比和时长对训练数据进行分桶。每个桶中的视频都具有相同的潜在形状，这使得训练数据的批处理更为方便。对于图像和视频数据集，我们设定了五个纵横比桶。因此，我们的模型能够生成不同纵横比的图像和视频，例如，1024 × 576 的横向比例和 576 × 1024 的纵向比例。我们定义了五个时长桶（4s – 16s），并根据视频长度调整潜在帧的数量。如前文所述，作者通过在文本字幕中添加 FPS 标记来引入 FPS 控制，从而允许我们以不同的帧率（16 – 32 FPS）采样视频。

### 训练

作者在本节中介绍了训练MOVIE GEN VIDEO 30B模型的细节。作者提到为了training efficiency 和model scalability，团队与emu相同采用了多阶段训练，主要包含以下3步：

- step1：先用text-to-image任务训练，之后用text-to-image和text-to-video的联合训练
- step2：从低分辨率256px数据逐渐到高分辨率768px数据
- step3：在计算和时间限制下，使用改进的数据集和优化的训练配方进行持续训练

作者提到保留了一个模型在训练期间从未见到的视频数据集用于在训练期间监视模型在该验证集上的表现。

在训练方面，作者提到使用预训练text2image模型来初始化模型相比直接训练视频生成模型能够带来更好的收敛效果和视频质量；因此作者采用了一个text-to-image的**warm-up 训练**。

在joint training阶段，作者提出将空间位置嵌入（PE）层加倍，以适应不同的纵横比，新增时间位置嵌入层，以支持最多32个潜在帧，并通过2倍扩展从T2I模型初始化空间位置嵌入层。对于768px视频数据的训练，作者将空间位置嵌入层扩展3倍。

### 微调

作者提到在微调阶段团队训练了多个模型并在最后通过模型平均的方法得到最终模型。

笔者在这里暂时跳过作者关于如何获取用于fine-tuning视频数据的描述。

### 推理

#### 提示重写

作者提到团队使用Llama 3进行提示重写。

- 作者使用标准化的信息架构来重新表述提示
- 通过用更容易理解和直接的术语替换复杂的词汇来改进重写的提示
- 对运动细节的过度详细描述可能导致在生成的视频中引入伪影

本文中使用了teacher-student的蒸馏方法，先使用Llama 3 70B构建一个提示重写模型，然后收集human-in-the-loop数据，最后使用这些数据来微调Llama 3 8B模型获得最终的prompt rewrite model，这样的小型模型有利于降低整个系统的延时。

#### 提高推理效率

作者提到使用具有专为模型定制的时间调度的Euler采样器。

> The linear-quadratic t-schedule.

对于该部分笔者尚且并不了解，暂时跳过。



## 评估

> This suggests that the Flow Matching validation loss can serve as a useful proxy for human evaluations during model development.



### 消融实验

![image-20241026091751577](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241026091751577.png)



![image-20241026092302159](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241026092302159.png)

上图为Lllama3-like 和DiT模型的不同之处。



## 文生图post-training



# 基于user image的video personalization

PT2V 模型架构图如下

![image-20241026105140618](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20241026105140618.png)

作者团队使用从长提示MetaCLIP初始化的可训练视觉编码器对参考图像进行编码，并与text embedding进行连接。

作者提到直接在长视频上训练模型是低效的，并且往往导致对个性化模型的身份注入缓慢。



# Instruction-Guided Precise Video Editing

两个主要假设：



同样地，训练方法包括几个训练阶段，旨在逐渐减少train与test之间的差异。

在第一阶段，我们使用多任务目标来训练文本到视频模型。该**目标在两种任务之间交替进行**：图像编辑（我们将其视为单帧的视频编辑）和视频生成。（笔者认为这种设计是非常有意思的）

在第二阶段，我们引入了两个新的合成任务，这些任务更接近于多帧视频编辑，并在这些任务上对模型进行微调。































