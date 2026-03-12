DeepEP 是由 DeepSeek 团队开源的、专为 **混合专家模型（Mixture-of-Experts, MoE）** 和 **专家并行（Expert Parallelism, EP）** 设计的高性能通信库。


---

### 1. 核心定位与背景
DeepEP 旨在解决 MoE 模型在训练和推理过程中面临的通信瓶颈。在 MoE 架构中，数据需要根据路由策略（Gating）分发到不同的专家（Expert）节点进行处理，这涉及大量的 **All-to-All** 通信操。

*   **主要功能**：提供高吞吐、低延迟的 GPU 内核，专门用于 MoE 的 **Dispatch**（分发）和 **Combine**（合并）操作。
*   **精度支持**：原生支持低精度运算，包括 **FP8**（用于分发）和 **BF16**（用于合并），以匹配 DeepSeek-V3/R1 的训练设定。
*   **算法对齐**：其内核优化逻辑对齐了 DeepSeek-V3 论文中提出的 **Group-Limited Gating** 算法，特别针对非对称带宽域（如从 NVLink 域转发到 RDMA 域）进行了优化 。
*   **适用场景**：
    *   **训练/预填充（Prefilling）**：使用高吞吐内核，利用 NVLink 和 RDMA 混合传输。
    *   **推理解码（Decoding）**：使用纯 RDMA 的低延迟内核，最小化单 token 生成的延迟。

---

### 2. 性能表现 (基于 H800 + CX7 InfiniBand)
DeepEP 在 NVIDIA H800 GPU（NVLink 带宽 ~160 GB/s）和 ConnectX-7 InfiniBand 网卡（RDMA 带宽 ~50 GB/s）上进行了测试。

#### A. 常规内核（高吞吐模式）
适用于训练和推理的预填充阶段，结合 NVLink（节点内）和 RDMA（节点间）。
*   **节点内 (Intranode, EP=8)**：瓶颈在 NVLink，分发/合并带宽分别达到 **153/158 GB/s**。
*   **节点间 (Internode)**：瓶颈在 RDMA。
    *   EP=16 时，带宽约 **43 GB/s**。
    *   EP=32 时，带宽提升至 **57-58 GB/s**（接近网卡物理极限）。
    *   EP=64 时，带宽维持在 **50-51 GB/s**。

#### B. 低延迟内核（纯 RDMA 模式）
适用于对延迟敏感的推理解码阶段（Batch size=128 tokens）。
*   **延迟表现**：
    *   EP=8 时，分发延迟仅 **77 μs**，合并延迟 **114 μs**。
    *   即使扩展到 EP=256，分发延迟也仅为 **194 μs**。
*   **带宽效率**：在小包传输下仍能跑出极高的有效带宽（如 EP=8 时 RDMA 带宽达 98 GB/s，远超单卡物理限制，利用了聚合效应）。

---

### 3. 核心技术架构

#### A. 两种工作模式
1.  **正常模式 (Normal Kernels)**：
    *   **机制**：智能调度 NVLink 和 RDMA。节点内通信走 NVLink，跨节点通信走 RDMA。
    *   **特点**：支持 SM（流多处理器）数量控制，允许用户调整用于通信的 GPU 计算资源比例，以平衡通信与计算。
    *   **流程**：CPU 需等待 GPU 信号以确定接收到的 Token 数量（存在隐式同步），因此不完全兼容 CUDA Graph，除非指定 `num_worst_tokens`。

2.  **低延迟模式 (Low-Latency Kernels)**：
    *   **机制**：完全基于 **RDMA**，绕过复杂的片上网络调度，专为微秒级响应设计。
    *   **特点**：不支持 SM 数量控制（固定优化路径），但完美支持 **CUDA Graph**，适合推理引擎的重放执行。
    *   **QP 配置**：为了最佳性能，建议 Queue Pair (QP) 的数量等于本地专家的数量 [[3]]。

#### B. 通信 - 计算重叠 (Communication-Computation Overlap)
这是 DeepEP 的一大亮点。
*   **Hook 机制**：引入了一种基于 **Hook** 的重叠方法。传统的重叠往往需要占用 SM 资源来轮询或管理通信，而 DeepEP 的 Hook 机制让 RDMA 网络流量在后台进行，**不占用任何 SM 资源**。
*   **应用场景**：在双微批次（Two-micro-batch）重叠中，当 GPU 在进行 Attention 或 MoE 计算时，下一批数据的 RDMA 传输已在后台通过 Hook 触发，实现了极致的流水线并行。

#### C. 未定义行为 PTX 指令 (Undefined-behavior PTX)
*   **激进优化**：为了极致性能，DeepEP 使用了一条未定义行为的 PTX 指令 `ld.global.nc.L1::no_allocate.L2::256B` 来读取易失性数据。
*   **原理**：通常 `.nc` (non-coherent) 用于只读缓存，但在 Hopper 架构上，配合 `.L1::no_allocate` 修饰符，可以确保数据不污染 L1 缓存且保持正确性，同时获得比手动展开的 volatile 读取更快的速度。
*   **兼容性**：如果在其他平台出现问题，可通过环境变量 `DISABLE_AGGRESSIVE_PTX_INSTRS=1` 禁用此特性。

---

### 4. 安装与依赖

#### 硬件与软件要求
*   **GPU**：Ampere (SM80, 如 A100/H800) 或 Hopper (SM90, 如 H100/B200)。A100 仅支持节点内功能。
*   **CUDA**：SM80 需 CUDA 11.0+，SM90 需 CUDA 12.3+。
*   **网络**：节点内需 NVLink，节点间需 RDMA (InfiniBand 或 RoCE)。
*   **关键依赖**：**NVSHMEM**。DeepEP 强依赖 NVIDIA 的 NVSHMEM 库进行显存直接访问和通信，需单独安装并打补丁。

#### 安装步骤
1.  **安装 NVSHMEM**：参考官方指南安装，并应用 DeepEP 提供的补丁以支持 IBGDA (InfiniBand GPUDirect Async)。
2.  **编译 DeepEP**：
    ```bash
    NVSHMEM_DIR=/path/to/nvshmem python setup.py install
    ```
3.  **环境变量**：
    *   `NVSHMEM_DIR`：必需，指向 NVSHMEM 安装路径。
    *   `TORCH_CUDA_ARCH_LIST`：指定目标架构，如 `"9.0"`。
    *   `DISABLE_SM90_FEATURES`：在非 SM90 设备或旧版 CUDA 上设为 1。

---

### 5. 网络配置建议

DeepEP 主要在 InfiniBand (IB) 网络上经过充分测试，理论上兼容 RoCE。

*   **流量隔离 (Traffic Isolation)**：
    *   利用 IB 的 **虚拟通道 (Virtual Lanes, VL)** 隔离不同类型的流量（正常内核、低延迟内核、其他业务），防止相互干扰。
    *   通过设置 `NVSHMEM_IB_SL` 环境变量控制服务等级 (Service Level) [[3]]。
*   **自适应路由 (Adaptive Routing)**：
    *   **重负载**：建议开启，可消除路由冲突导致的拥塞，但会略微增加延迟。
    *   **轻负载**：建议使用静态路由以降低延迟 [[3]]。
*   **拥塞控制**：在生产环境中通常**禁用**，因为未观察到显著拥塞 [[3]]。

---

### 6. 代码接口与使用示例

DeepEP 提供了 Python 接口，核心类为 `Buffer` 和 `EventOverlap`。

#### A. 训练/预填充场景 (Normal Mode)
主要流程包括：计算布局 (`get_dispatch_layout`) -> 执行分发 (`dispatch`) -> 专家计算 -> 执行合并 (`combine`)。
*   **异步执行**：支持 `async_finish=True`，返回 `EventOverlap` 对象用于同步。
*   **反向传播**：
    *   Dispatch 的反向实际上是 Combine 操作。
    *   Combine 的反向实际上是 Dispatch 操作 [[3]]。

#### B. 推理解码场景 (Low-Latency Mode)
针对单 Token 或小 Batch 优化。
*   **初始化**：需指定 `low_latency_mode=True` 和 `num_qps_per_rank`。
*   **Hook 用法**：
    ```python
    # 返回 hook 函数，不立即接收数据
    recv_hidden_states, ..., hook = buffer.low_latency_dispatch(..., return_recv_hook=True)
    
    # 在需要时调用 hook() 触发实际的数据接收，实现与计算的无 SM 重叠
    hook() 
    ```
    这种机制允许在 GPU 计算当前层时，后台静默传输下一层所需数据 [[3]]。

---

### 7. 路线图与社区生态

#### 未来规划 (Roadmap)
*   **TMA Copy**：计划用 Tensor Memory Accelerator (TMA) 指令替代传统的 Load/Store，进一步降低 SM 占用（目前节点内已支持）[[3]]。
*   **SM-Free**：致力于完全移除通信内核中的 SM 占用。
*   **移除未定义指令**：长期目标是移除激进的 PTX 指令以提高通用性 [[3]]。

#### 实验性分支与社区 Fork
*   **Zero-copy**：由腾讯开发，移除 PyTorch Tensor 与通信缓冲区的拷贝，显著降低 SM 占用 [[3]]。
*   **Hybrid-EP**：支持 TMA 指令、PCIe 环境（无 NVLink）及 NVFP4 数据类型 [[3]]。
*   **Mori-EP**：由 ROCm/MORI 支持，使得 DeepEP 能在 **AMD GPU** 上运行 [[3]]。
*   **DeepXTrace**：由蚂蚁集团开发，用于诊断和定位慢速 Rank 的工具 [[3]]。

### 总结
DeepEP 是目前针对 MoE 模型最先进（SOTA）的通信库之一。它通过**软硬协同设计**（利用 NVLink+RDMA 异构带宽、激进的 PTX 指令、NVSHMEM 显存直连）解决了大规模专家并行的通信难题。特别是其**无 SM 占用的通信重叠机制**和**微秒级低延迟内核**，为千亿参数 MoE 模型的高效训练和实时推理提供了关键的基础设施支持 [[41]]。