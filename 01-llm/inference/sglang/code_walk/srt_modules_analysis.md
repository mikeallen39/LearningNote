# SGLang SRT 模块架构分析

> **路径**: `/mnt/42_store/zxz/sglang/python/sglang/srt`
>
> **SRT (SGLang Runtime)** 是 SGLang 的核心运行时模块，包含了推理引擎的所有关键组件。

---

## 目录

1. [概述](#概述)
2. [目录结构总览](#目录结构总览)
3. [各模块详细说明](#各模块详细说明)
4. [模块依赖关系](#模块依赖关系)

---

## 概述

SGLang SRT 是一个高性能的大语言模型推理运行时，主要特点包括：

- **高效的 KV Cache 管理**: 支持 Radix Cache、Prefix Cache 等多种缓存策略
- **灵活的调度策略**: 支持连续批处理、投机解码等
- **多后端支持**: 支持 FlashInfer、FlashAttention、Triton 等多种推理后端
- **分布式推理**: 支持张量并行、流水线并行、专家并行
- **多模态支持**: 支持 VLM（视觉语言模型）推理
- **LoRA 支持**: 支持动态加载和卸载 LoRA 适配器

---

## 目录结构总览

```
srt/
├── batch_invariant_ops/    # 批处理不变操作优化
├── batch_overlap/          # 批处理重叠调度
├── checkpoint_engine/      # 检查点引擎
├── compilation/            # 模型编译优化
├── configs/                # 模型配置定义
├── connector/              # 远程连接器
├── constrained/            # 结构化输出约束
├── debug_utils/            # 调试工具
├── disaggregation/         # 分离式推理
├── distributed/            # 分布式通信
├── dllm/                   # 扩散语言模型支持
├── elastic_ep/             # 弹性专家并行
├── entrypoints/            # 服务入口点
├── eplb/                   # 专家并行负载均衡
├── function_call/          # 函数调用解析
├── grpc/                   # gRPC 服务
├── hardware_backend/       # 硬件后端抽象
├── layers/                 # 神经网络层实现
├── lora/                   # LoRA 适配器管理
├── managers/               # 核心管理器
├── mem_cache/              # 内存缓存管理
├── metrics/                # 指标收集
├── model_executor/         # 模型执行器
├── model_loader/           # 模型加载器
├── models/                 # 模型实现
├── multimodal/             # 多模态处理
├── multiplex/              # 多路复用调度
├── parser/                 # 对话模板解析
├── sampling/               # 采样策略
├── speculative/            # 投机解码
├── tokenizer/              # 分词器
├── tracing/                # 链路追踪
├── utils/                  # 工具函数
└── weight_sync/            # 权重同步
```

---

## 各模块详细说明

### 1. `batch_invariant_ops/` - 批处理不变操作优化

**功能**: 提供批处理不变（batch-invariant）操作的优化实现。

**核心组件**:
- `batch_invariant_ops.py`: 主要实现文件
- 支持 RMS Norm、MatMul、LogSoftmax 等操作的批处理不变模式

**关键函数**:
| 函数名 | 说明 |
|--------|------|
| `set_batch_invariant_mode` | 设置批处理不变模式 |
| `matmul_persistent` | 持久化矩阵乘法 |
| `rms_norm_batch_invariant` | 批处理不变的 RMS 归一化 |

**应用场景**: 当批处理中的某些操作不依赖于批处理大小时，可以使用这些优化来提高效率。

---

### 2. `batch_overlap/` - 批处理重叠调度

**功能**: 实现批处理操作的重叠执行，提高 GPU 利用率。

**核心文件**:
- `operations.py`: 操作定义和执行框架
- `operations_strategy.py`: 操作策略
- `single_batch_overlap.py`: 单批次重叠
- `two_batch_overlap.py`: 双批次重叠

**关键概念**:
- `YieldOperation`: 标记可以暂停的操作点
- `ExecutionOperation`: 实际执行的操作
- `_StageExecutor`: 阶段执行器，支持操作的分阶段执行

**调度流程**:
```
execute_overlapped_operations
  └── _StageExecutor (batch A)  ─┐
  └── _StageExecutor (batch B)  ─┴─> 交替执行
```

---

### 3. `checkpoint_engine/` - 检查点引擎

**功能**: 提供模型权重的检查点更新功能。

**核心组件**:
- `checkpoint_engine_worker.py`: 检查点引擎工作器
- `update.py`: 权重更新逻辑

**应用场景**:
- 在线模型权重更新
- 从远程存储加载检查点

---

### 4. `compilation/` - 模型编译优化

**功能**: 提供基于 PyTorch Dynamo 的模型编译优化能力。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `compile.py` | 编译入口，支持 `torch.compile` 集成 |
| `backend.py` | SGLang 自定义编译后端 |
| `compiler_interface.py` | 编译器接口 |
| `cuda_piecewise_backend.py` | CUDA 分段编译后端 |
| `inductor_pass.py` | Inductor 优化 Pass |

**关键类**:
```python
class IntermediateTensors:
    """用于流水线阶段间传递中间张量"""
    tensors: dict[str, torch.Tensor]
    finished_sending: Optional[set[str]]
    finished_recving: Optional[set[str]]
```

**编译流程**:
```
install_torch_compiled(module)
  └── torch.compile(module, backend=SGLangBackend)
       └── 图优化 → CUDA Kernel 生成
```

---

### 5. `configs/` - 模型配置定义

**功能**: 定义各种模型的自定义配置，扩展 HuggingFace Transformers 的配置。

**支持的模型配置**:
| 配置类 | 模型 |
|--------|------|
| `AfmoeConfig` | AFMoE 模型 |
| `BailingHybridConfig` | 百灵混合模型 |
| `ChatGLMConfig` | ChatGLM 系列 |
| `DbrxConfig` | Databricks DBRX |
| `DeepseekVL2Config` | DeepSeek-VL2 |
| `ExaoneConfig` | Exaone 模型 |
| `FalconH1Config` | Falcon H1 |
| `GraniteMoeHybridConfig` | Granite MoE 混合 |
| `KimiVLConfig` | Kimi 视觉语言模型 |
| `Qwen3_5Config` | Qwen3.5 系列 |
| `Step3VLConfig` | Step3 视觉语言模型 |
| ... | 更多模型 |

---

### 6. `connector/` - 远程连接器

**功能**: 提供与远程存储和计算资源的连接能力。

**支持的连接器类型**:
| 连接器 | 说明 |
|--------|------|
| `RedisConnector` | Redis KV 存储 |
| `S3Connector` | S3 对象存储 |
| `RemoteInstanceConnector` | 远程实例连接 |

**架构设计**:
```
BaseConnector (抽象基类)
├── BaseKVConnector    # KV 存储连接器
├── BaseFileConnector  # 文件系统连接器
└── RemoteInstanceConnector  # 远程实例
```

**应用场景**:
- 分布式推理中的 KV Cache 传输
- 模型权重的远程加载
- 分离式推理架构

---

### 7. `constrained/` - 结构化输出约束

**功能**: 实现基于语法的结构化输出（JSON、正则表达式、EBNF 等）。

**核心组件**:
| 文件 | 说明 |
|------|------|
| `base_grammar_backend.py` | 语法后端基类 |
| `xgrammar_backend.py` | XGrammar 后端 |
| `outlines_backend.py` | Outlines 后端 |
| `llguidance_backend.py` | LLGuidance 后端 |
| `grammar_manager.py` | 语法管理器 |

**支持的结构化输出类型**:
- **JSON Schema**: 强制输出符合 JSON Schema
- **Regex**: 正则表达式约束
- **EBNF**: 扩展巴科斯范式语法
- **Structural Tag**: 结构化标签

**工作流程**:
```
用户请求 (JSON Schema)
  └── GrammarBackend.dispatch_json(schema)
       └── 编译语法 → 生成约束
            └── 推理时应用 vocab_mask 约束
```

---

### 8. `debug_utils/` - 调试工具

**功能**: 提供调试和分析工具。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `dumper.py` | 张量转储工具 |
| `dump_loader.py` | 转储数据加载 |
| `dump_comparator.py` | 转储比较工具 |
| `text_comparator.py` | 文本比较工具 |
| `model_truncator.py` | 模型截断工具 |
| `schedule_simulator/` | 调度模拟器 |

**应用场景**:
- 推理结果对比调试
- 性能瓶颈分析
- 调度策略模拟

---

### 9. `disaggregation/` - 分离式推理

**功能**: 实现预填充（Prefill）和解码（Decode）分离的分布式推理架构。

**架构模式**:
```
┌─────────────────┐    KV Cache    ┌─────────────────┐
│  Prefill Server │ ────────────→  │  Decode Server  │
│  (计算密集型)    │               │  (内存密集型)    │
└─────────────────┘                └─────────────────┘
```

**核心文件**:
| 文件 | 说明 |
|------|------|
| `prefill.py` | Prefill 服务器实现 |
| `decode.py` | Decode 服务器实现 |
| `encode_server.py` | 编码服务器 |
| `encode_receiver.py` | 编码接收器 |
| `kv_events.py` | KV 事件处理 |

**子目录**:
- `base/`: 基础抽象类
- `common/`: 公共组件
- `mooncake/`: Mooncake 传输后端
- `nixl/`: NVIDIA NIXL 传输后端
- `mori/`: Mori 传输后端
- `ascend/`: 华为 Ascend 后端

**请求生命周期**:
```
1. Bootstrap Queue → 初始化发送器和握手
2. Waiting Queue → Prefill Adder 弹出请求执行前向
3. Inflight Queue → 等待 KV Cache 传输完成
```

---

### 10. `distributed/` - 分布式通信

**功能**: 管理分布式推理的通信状态和操作。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `parallel_state.py` | 并行状态管理（核心） |
| `communication_op.py` | 通信操作 |
| `naive_distributed.py` | 朴素分布式实现 |
| `device_communicators/` | 设备通信器 |

**支持的并行模式**:
- **TP (Tensor Parallel)**: 张量并行
- **PP (Pipeline Parallel)**: 流水线并行
- **DP (Data Parallel)**: 数据并行
- **EP (Expert Parallel)**: 专家并行

**关键函数**:
```python
init_distributed_environment()  # 初始化分布式环境
initialize_model_parallel()     # 初始化模型并行组
get_tp_group()                  # 获取张量并行组
get_pp_group()                  # 获取流水线并行组
```

**使用流程**:
```
init_distributed_environment()
  └── initialize_model_parallel()
       └── 业务代码
            └── destroy_model_parallel()
                 └── destroy_distributed_environment()
```

---

### 11. `dllm/` - 扩散语言模型支持

**功能**: 支持扩散式语言模型（如 LLaDA、SDAR）的推理。

**核心文件**:
- `config.py`: DLLM 配置
- `algorithm/`: 扩散算法实现
- `mixin/`: 混入类

**支持的模型**:
| 模型架构 | block_size | mask_id |
|----------|------------|---------|
| LLaDA2MoeModelLM | 32 | 156895 |
| SDARForCausalLM | 4 | 151669 |
| SDARMoeForCausalLM | 4 | 151669 |

---

### 12. `elastic_ep/` - 弹性专家并行

**功能**: 实现弹性专家并行，支持动态调整活跃专家数量。

**核心类**:
```python
@dataclass
class ElasticEPState:
    active_ranks: Optional[torch.Tensor]      # 当前活跃的 rank
    last_active_ranks: Optional[torch.Tensor] # 上次活跃的 rank
    active_ranks_cpu: Optional[torch.Tensor]  # CPU 副本
```

**应用场景**:
- 专家模型故障恢复
- 动态负载均衡
- 弹性伸缩

---

### 13. `entrypoints/` - 服务入口点

**功能**: 提供推理引擎的各种入口点和 API。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `engine.py` | Python API 入口（主要） |
| `http_server.py` | HTTP/REST API 服务器 |
| `grpc_server.py` | gRPC API 服务器 |
| `http_server_engine.py` | HTTP 服务器引擎 |

**子目录**:
- `openai/`: OpenAI 兼容 API
- `ollama/`: Ollama 兼容 API

**Engine 核心功能**:
```python
class Engine:
    async def async_generate()      # 异步生成
    async def async_batch_generate() # 批量异步生成
    def generate()                   # 同步生成
    async def update_weights()       # 更新权重
```

---

### 14. `eplb/` - 专家并行负载均衡

**功能**: 实现专家模型并行的负载均衡和动态重平衡。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `eplb_manager.py` | EPLB 管理器 |
| `expert_distribution.py` | 专家分布记录 |
| `expert_location.py` | 专家位置管理 |
| `expert_location_updater.py` | 专家位置更新器 |

**子目录**:
- `eplb_algorithms/`: 负载均衡算法
- `eplb_simulator/`: 模拟器

**工作原理**:
```
1. 记录专家访问分布 (ExpertDistributionRecorder)
2. 定期触发重平衡 (EPLBManager.rebalance)
3. 计算最优专家位置
4. 执行专家迁移
```

---

### 15. `function_call/` - 函数调用解析

**功能**: 解析和处理模型的函数调用（Tool Calling）输出。

**支持的解析器**:
| 解析器 | 模型 |
|--------|------|
| `DeepSeekV3Detector` | DeepSeek V3 |
| `DeepSeekV31Detector` | DeepSeek V3.1 |
| `Glm4MoeDetector` | GLM-4-MoE |
| `Llama32Detector` | Llama 3.2 |
| `Qwen25Detector` | Qwen2.5 |
| `MistralDetector` | Mistral |
| `HermesDetector` | Hermes 格式 |
| ... | 更多模型 |

**核心类**:
```python
class FunctionCallParser:
    def parse_streaming_increment(text) -> (normal_text, calls)
    def parse_non_streaming(text) -> calls
```

---

### 16. `grpc/` - gRPC 服务

**功能**: 提供 gRPC 协议的推理服务。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `grpc_request_manager.py` | gRPC 请求管理器 |
| `health_servicer.py` | 健康检查服务 |
| `scheduler_launcher.py` | 调度器启动器 |

**特点**:
- 支持流式响应
- 内置健康检查
- 与 HTTP 服务共享调度器

---

### 17. `hardware_backend/` - 硬件后端抽象

**功能**: 提供不同硬件平台的抽象层。

**子目录**:
- `npu/`: 华为 NPU (昇腾) 后端

**支持的硬件**:
- NVIDIA GPU (CUDA)
- AMD GPU (ROCm/HIP)
- 华为 NPU (昇腾)
- Intel CPU (AMX)
- 其他 XPU 设备

---

### 18. `layers/` - 神经网络层实现

**功能**: 实现各种优化的神经网络层。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `activation.py` | 激活函数 |
| `linear.py` | 线性层（含量化） |
| `layernorm.py` | LayerNorm/RMSNorm |
| `logits_processor.py` | Logits 处理器 |
| `communicator.py` | 分布式通信封装 |
| `dp_attention.py` | 数据并行注意力 |

**子目录**:

#### `attention/` - 注意力机制
| 子目录/文件 | 说明 |
|------------|------|
| `flashinfer_backend.py` | FlashInfer 后端 |
| `flashattention_backend.py` | FlashAttention 后端 |
| `triton_backend.py` | Triton 后端 |
| `cutlass_mla_backend.py` | CUTLASS MLA 后端 |
| `nsa_backend.py` | NSA (Native Sparse Attention) |
| `mamba/` | Mamba 状态空间模型 |
| `triton_ops/` | Triton 优化算子 |

#### `moe/` - 混合专家
| 子目录/文件 | 说明 |
|------------|------|
| `fused_moe_triton/` | Triton 融合 MoE |
| `cutlass_moe.py` | CUTLASS MoE |
| `router.py` | 专家路由 |
| `topk.py` | Top-K 选择 |
| `token_dispatcher/` | Token 分发器 |

---

### 19. `lora/` - LoRA 适配器管理

**功能**: 支持动态加载、卸载和运行 LoRA 适配器。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `lora_manager.py` | LoRA 管理器（核心） |
| `lora.py` | LoRA 适配器定义 |
| `lora_config.py` | LoRA 配置 |
| `mem_pool.py` | LoRA 内存池 |
| `layers.py` | 支持 LoRA 的层 |

**子目录**:
- `backend/`: LoRA 后端实现
- `torch_ops/`: PyTorch 操作
- `triton_ops/`: Triton 优化操作

**关键特性**:
- **S-LoRA**: 支持数千个并发 LoRA 适配器
- **Punica**: 多租户 LoRA 服务
- **动态加载/卸载**: 运行时管理适配器

---

### 20. `managers/` - 核心管理器

**功能**: 包含推理引擎的核心调度和管理组件。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `schedule_batch.py` | 批处理调度数据结构（核心） |
| `schedule_policy.py` | 调度策略 |
| `io_struct.py` | 请求/响应数据结构 |
| `detokenizer_manager.py` | Detokenizer 管理 |
| `data_parallel_controller.py` | 数据并行控制器 |
| `cache_controller.py` | 缓存控制器 |

**数据流**:
```
ScheduleBatch → ModelWorkerBatch → ForwardBatch

- ScheduleBatch: 调度器管理，高层调度数据（CPU）
- ModelWorkerBatch: 模型工作者，前向传播相关数据
- ForwardBatch: 模型运行器，底层张量数据（GPU）
```

---

### 21. `mem_cache/` - 内存缓存管理

**功能**: 管理 KV Cache 的内存分配、缓存和回收。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `memory_pool.py` | 内存池（核心） |
| `radix_cache.py` | Radix Cache 实现 |
| `hiradix_cache.py` | 分层 Radix Cache |
| `allocator.py` | 内存分配器 |
| `common.py` | 公共组件 |

**子目录**:
- `cpp_radix_tree/`: C++ 实现的 Radix 树
- `sparsity/`: 稀疏性支持

**两级内存池架构**:
```
ReqToTokenPool: 请求 → Token 位置映射
TokenToKVPool:  Token → KV Cache 数据
```

**支持的缓存类型**:
- **Radix Cache**: 基于 Radix 树的前缀缓存
- **HiRadix Cache**: 分层 Radix 缓存
- **Mamba Radix Cache**: 状态空间模型缓存

---

### 22. `metrics/` - 指标收集

**功能**: 收集和暴露 Prometheus 指标。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `collector.py` | 指标收集器 |
| `func_timer.py` | 函数计时器 |
| `cpu_monitor.py` | CPU 监控 |

**关键指标**:
- 请求延迟
- Token 生成速度
- GPU 内存使用
- 队列长度
- 批处理大小

---

### 23. `model_executor/` - 模型执行器

**功能**: 执行模型前向传播，管理 CUDA Graph。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `model_runner.py` | 模型运行器（核心） |
| `forward_batch_info.py` | 前向批处理信息 |
| `cuda_graph_runner.py` | CUDA Graph 运行器 |
| `piecewise_cuda_graph_runner.py` | 分段 CUDA Graph |
| `cpu_graph_runner.py` | CPU Graph 运行器 |

**ForwardBatch 结构**:
```python
@dataclass
class ForwardBatch:
    forward_mode: ForwardMode  # EXTEND/DECODE/...
    batch_size: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    # ... 更多字段
```

---

### 24. `model_loader/` - 模型加载器

**功能**: 加载模型权重，支持多种加载方式。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `loader.py` | 模型加载器（核心） |
| `weight_utils.py` | 权重处理工具 |
| `ci_weight_validation.py` | CI 权重验证 |
| `remote_instance_weight_loader_utils.py` | 远程权重加载 |

**支持的加载格式**:
- HuggingFace Safetensors
- PyTorch .bin
- GGUF
- 远程存储 (S3, Redis)

---

### 25. `models/` - 模型实现

**功能**: 各种模型的实现代码。

**支持的模型**（部分）:
| 文件 | 模型 |
|------|------|
| `deepseek.py` | DeepSeek 系列 |
| `baichuan.py` | 百川 |
| `chatglm.py` | ChatGLM |
| `bert.py` | BERT |
| `clip.py` | CLIP |
| `afmoe.py` | AFMoE |
| `bailing_moe.py` | 百灵 MoE |
| ... | 更多模型 |

**子目录**:
- `deepseek_common/`: DeepSeek 公共组件

---

### 26. `multimodal/` - 多模态处理

**功能**: 处理多模态（图像、视频等）输入。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `mm_utils.py` | 多模态工具函数 |
| `vit_cuda_graph_runner.py` | ViT CUDA Graph |
| `internvl_vit_cuda_graph_runner.py` | InternVL ViT |

**子目录**:
- `processors/`: 多模态处理器
- `evs/`: EVS 处理

**支持的多模态模型**:
- LLaVA 系列
- InternVL
- DeepSeek-VL
- Qwen-VL

---

### 27. `multiplex/` - 多路复用调度

**功能**: 实现 Prefill-Decode 多路复用调度。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `multiplexing_mixin.py` | 多路复用混入类 |
| `pdmux_context.py` | PD-Mux 上下文 |

**工作原理**:
- 将 GPU 流处理器划分为多个流
- Prefill 和 Decode 可以并发执行
- 动态调整资源分配

---

### 28. `parser/` - 对话模板解析

**功能**: 管理和解析对话模板。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `conversation.py` | 对话模板定义（核心） |
| `harmony_parser.py` | Harmony 解析器 |
| `reasoning_parser.py` | 推理内容解析器 |
| `jinja_template_utils.py` | Jinja 模板工具 |

**支持的模板样式**:
- LLAMA2/LLAMA3/LLAMA4
- CHATGLM
- CHATML
- Mistral
- ... 更多

---

### 29. `sampling/` - 采样策略

**功能**: 实现各种采样策略和参数。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `sampling_batch_info.py` | 批处理采样信息 |
| `sampling_params.py` | 采样参数定义 |
| `custom_logit_processor.py` | 自定义 Logit 处理器 |

**子目录**:
- `penaltylib/`: 惩罚库（频率、重复等）

**支持的采样方法**:
- Temperature 采样
- Top-p (Nucleus) 采样
- Top-k 采样
- Min-p 采样
- Beam Search
- 自定义 Logit 处理

---

### 30. `speculative/` - 投机解码

**功能**: 实现投机解码（Speculative Decoding）加速推理。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `eagle_worker.py` | EAGLE 投机解码 |
| `eagle_worker_v2.py` | EAGLE V2 |
| `ngram_worker.py` | N-gram 投机解码 |
| `eagle_info.py` | EAGLE 信息结构 |
| `draft_utils.py` | Draft 模型工具 |

**子目录**:
- `cpp_ngram/`: C++ N-gram 实现

**支持的投机方法**:
- **EAGLE**: 基于草稿模型的投机解码
- **N-gram**: 基于 N-gram 的简单推测
- **Multi-layer EAGLE**: 多层 EAGLE

---

### 31. `tokenizer/` - 分词器

**功能**: 提供分词器支持。

**核心文件**:
- `tiktoken_tokenizer.py`: Tiktoken 分词器封装

**支持**:
- HuggingFace Tokenizers
- Tiktoken
- 自定义分词器

---

### 32. `tracing/` - 链路追踪

**功能**: 提供 OpenTelemetry 集成的分布式追踪。

**核心文件**:
- `trace.py`: 追踪实现

**支持**:
- OTLP HTTP/gRPC 导出
- Trace Context 传播
- Span 属性记录

---

### 33. `utils/` - 工具函数

**功能**: 提供各种通用工具函数。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `common.py` | 通用工具函数（核心，136KB） |
| `hf_transformers_utils.py` | HuggingFace 工具 |
| `auth.py` | 认证工具 |
| `bench_utils.py` | 基准测试工具 |
| `nvtx_pytorch_hooks.py` | NVTX 性能分析钩子 |

---

### 34. `weight_sync/` - 权重同步

**功能**: 支持分布式环境下的权重同步和更新。

**核心文件**:
| 文件 | 说明 |
|------|------|
| `utils.py` | 权重同步工具 |
| `tensor_bucket.py` | 张量桶管理 |

**应用场景**:
- 分布式训练与推理同步
- 在线权重更新
- 多副本模型同步

---

## 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                      entrypoints/                            │
│  (engine.py, http_server.py, grpc_server.py)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                       managers/                              │
│  (schedule_batch.py, schedule_policy.py, io_struct.py)     │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│ model_executor│ │   mem_cache   │ │  distributed  │
│               │ │               │ │               │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                        layers/                               │
│  (attention/, moe/, linear.py, activation.py)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                       models/                                │
│  (具体模型实现: deepseek.py, baichuan.py, ...)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心数据流

```
用户请求
    │
    ▼
┌─────────────┐
│ entrypoints │ (HTTP/gRPC/Python API)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Tokenizer   │ (tokenization)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Scheduler   │ (managers/schedule_batch.py)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ModelRunner  │ (model_executor/model_runner.py)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Layers    │ (layers/attention/, layers/moe/)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Sampling   │ (sampling/)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Detokenizer │ (managers/detokenizer_manager.py)
└──────┬──────┘
       │
       ▼
响应输出
```

---