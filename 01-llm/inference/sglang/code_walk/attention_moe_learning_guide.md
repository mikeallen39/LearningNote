# Attention 优化与 MoE 并行推理学习指南

> 基于 SGLang 代码库的学习路径建议

---

## 一、Attention 优化学习路径

### 1.1 整体架构理解

首先理解 SGLang 中 Attention 的抽象层次：

```
┌─────────────────────────────────────────────────────────────┐
│                    RadixAttention                           │
│               (layers/radix_attention.py)                   │
│                    高层注意力接口                             │
└─────────────────────────┬───────────────────────────────────┘
                          │ 调用
┌─────────────────────────▼───────────────────────────────────┐
│                  AttentionBackend                           │
│           (layers/attention/base_attn_backend.py)           │
│                    抽象基类                                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
     ┌────────────────────┼────────────────────┐
     │                    │                    │
┌────▼────┐        ┌─────▼─────┐       ┌──────▼──────┐
│FlashInfer│        │  Triton   │       │ FlashAttention│
│ Backend  │        │  Backend  │       │   Backend    │
└──────────┘        └───────────┘       └──────────────┘
```

### 1.2 推荐学习顺序

#### 第一步：理解高层接口（1-2小时）

**文件**: `layers/radix_attention.py`

```python
class RadixAttention(nn.Module):
    """Attention 层的统一入口"""

    def __init__(self, num_heads, head_dim, scaling, num_kv_heads, ...):
        # 关键参数：
        # - num_heads: Q 的头数
        # - num_kv_heads: KV 的头数（支持 GQA）
        # - layer_id: 层 ID，用于定位 KV Cache
        # - attn_type: DECODER / ENCODER_ONLY
```

**关键概念**:
- **GQA (Grouped Query Attention)**: `num_kv_heads < num_heads`
- **AttentionType**: 区分解码器注意力和编码器注意力

---

#### 第二步：理解后端抽象（1-2小时）

**文件**: `layers/attention/base_attn_backend.py`

```python
class AttentionBackend(ABC):
    """所有 Attention 后端的基类"""

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """初始化前向传播所需的元数据"""

    @abstractmethod
    def forward(self, q, k, v, ...):
        """执行实际的 Attention 计算"""
```

**核心方法**:
| 方法 | 作用 |
|------|------|
| `init_forward_metadata` | 准备 attention 计算所需的元数据 |
| `forward` | 执行 Q * K^T * V 计算 |
| `init_cuda_graph_state` | CUDA Graph 相关状态初始化 |

---

#### 第三步：深入 FlashInfer 后端（重点，3-4小时）

**文件**: `layers/attention/flashinfer_backend.py`

这是**生产环境推荐**的后端，性能最优。

**关键数据结构**:
```python
@dataclass
class DecodeMetadata:
    """Decode 阶段的元数据"""
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]

@dataclass
class PrefillMetadata:
    """Prefill 阶段的元数据"""
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool
```

**核心流程**:
```python
class FlashInferAttnBackend(AttentionBackend):
    def forward(self, q, k, v, layer: RadixAttention, forward_batch):
        if forward_batch.forward_mode.is_extend():
            # Prefill/Extend 路径
            return self._forward_extend(q, k, v, layer, forward_batch)
        else:
            # Decode 路径
            return self._forward_decode(q, k, v, layer, forward_batch)
```

**学习要点**:
1. 区分 **Extend** 和 **Decode** 两种模式
2. 理解 **Paged KV Cache** 的索引方式
3. 学习如何处理变长序列

---

#### 第四步：学习 Triton 后端（可选，2-3小时）

**文件**: `layers/attention/triton_backend.py`

**适用场景**:
- 需要自定义 attention 变体
- 学习 Triton 编程
- 研究/实验目的

**目录**: `layers/attention/triton_ops/`

---

#### 第五步：高级主题

**稀疏注意力 (NSA)**:
- `layers/attention/nsa/` - Native Sparse Attention
- `layers/attention/nsa_backend.py`

**多头潜在注意力 (MLA)**:
- `layers/attention/cutlass_mla_backend.py`
- `layers/attention/flashmla_backend.py`
- `layers/attention/flashinfer_mla_backend.py`

---

### 1.3 Attention 学习检查清单

- [ ] 理解 `RadixAttention` 如何被模型调用
- [ ] 理解 `AttentionBackend` 的抽象接口
- [ ] 跟踪一次完整的 Prefill 路径
- [ ] 跟踪一次完整的 Decode 路径
- [ ] 理解 KV Cache 的分页管理
- [ ] 理解 GQA 的实现方式

---

## 二、MoE 并行推理学习路径

### 2.1 MoE 计算流程

```
Input Token (hidden_dim)
        │
        ▼
┌─────────────────┐
│     Router      │  计算每个 token 应该路由到哪些专家
│  (router.py)    │  输出: topk_ids, topk_weights
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Token Dispatcher│  将 token 分发到对应专家
│ (token_dispatcher/)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Expert Compute  │  并行执行专家计算
│ (fused_moe/)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Combine       │  按权重聚合专家输出
└─────────────────┘
         │
         ▼
    Output Token
```

### 2.2 推荐学习顺序

#### 第一步：理解 MoE 层结构（1-2小时）

**目录**: `layers/moe/`

```
layers/moe/
├── router.py              # 路由器：决定 token 去哪个专家
├── topk.py                # Top-K 选择
├── token_dispatcher/      # Token 分发器
├── fused_moe_triton/      # Triton 优化的 MoE kernel
├── cutlass_moe.py         # CUTLASS MoE 实现
└── ep_moe/                # 专家并行实现
```

---

#### 第二步：学习 Router 实现（1-2小时）

**文件**: `layers/moe/router.py`

```python
@triton.jit
def fused_moe_router_cudacore_kernel(...):
    """Triton 实现的高效路由 kernel"""

    # 1. 加载路由权重
    w_router = tl.load(moe_router_weight_ptr + ...)

    # 2. 计算路由 logits: x @ W^T
    logits = tl.sum((w_router * x), axis=-1)

    # 3. 可选的 softcap
    if moe_softcapping != 0:
        logits = softcap(logits, moe_softcapping)

    # 4. Top-K 选择
    top1 = tl.argmax(logits, axis=0)

    # 5. 计算权重 (softmax)
    weights = softmax(logits)
```

**关键点**:
- 路由是逐 token 独立计算的
- 支持 Top-1 和 Top-K 路由
- 支持 Softcap 归一化

---

#### 第三步：理解 Token Dispatcher（2-3小时）

**目录**: `layers/moe/token_dispatcher/`

**文件结构**:
```
token_dispatcher/
├── base.py        # 抽象基类
├── standard.py    # 标准（单 GPU）分发器
├── deepep.py      # DeepEP 分发器（高性能）
├── flashinfer.py  # FlashInfer 分发器
├── moriep.py      # MoRIEP 分发器
└── mooncake.py    # Mooncake 分发器
```

**核心接口** (`base.py`):
```python
class BaseTokenDispatcher(ABC):
    def dispatch(self, hidden_states, topk_ids, ...):
        """将 token 分发到专家"""

    def combine(self, expert_output, ...):
        """聚合专家输出"""
```

**关键概念**:
- **Token Permutation**: 重排 token 使同一专家的 token 连续
- **Expert Parallel**: 不同专家分布在不同 GPU 上
- **All-to-All Communication**: 跨 GPU 的 token 交换

---

#### 第四步：深入 Fused MoE Kernel（3-4小时）

**目录**: `layers/moe/fused_moe_triton/`

**关键文件**:
| 文件 | 说明 |
|------|------|
| `fused_moe.py` | 核心 fused MoE kernel |
| `layer.py` | MoE 层定义 |
| `fused_moe_triton_config.py` | 自动调优配置 |

**入口函数**:
```python
def fused_experts(
    hidden_states,      # (num_tokens, hidden_dim)
    w1,                 # Gate 权重 (num_experts, intermediate_dim, hidden_dim)
    w2,                 # Up 权重 (num_experts, hidden_dim, intermediate_dim)
    topk_weights,       # (num_tokens, topk)
    topk_ids,           # (num_tokens, topk)
    ...
):
    """
    执行: output = (x @ w1) * silu(x @ w3) @ w2
    按 topk_weights 加权求和
    """
```

**优化要点**:
1. **Block-wise Matrix Multiplication**: 分块计算适应不同 expert load
2. **Shared Memory Optimization**: 利用 GPU shared memory
3. **Load Balancing**: 处理专家负载不均衡

---

#### 第五步：理解专家并行（EP）（3-4小时）

**目录**: `layers/moe/ep_moe/`

**相关文件**:
- `distributed/parallel_state.py` - 并行状态管理
- `eplb/` - 专家并行负载均衡

**EP 核心概念**:

```
假设 4 个 GPU，8 个专家：

GPU 0: Expert 0, 1
GPU 1: Expert 2, 3
GPU 2: Expert 4, 5
GPU 3: Expert 6, 7

Token 路由流程:
1. 本地路由计算 topk_ids
2. All-to-All 将 token 发送到对应专家所在 GPU
3. 执行专家计算
4. All-to-All 将结果发回原 GPU
5. 加权聚合
```

**关键通信操作**:
```python
# distributed/parallel_state.py
def expert_model_parallel_all_reduce(tensor):
    """EP 组内的 All-Reduce"""

def all_to_all(tensor, group):
    """All-to-All 通信，用于 token 交换"""
```

---

#### 第六步：学习动态负载均衡

**目录**: `eplb/`

**关键文件**:
| 文件 | 说明 |
|------|------|
| `expert_distribution.py` | 记录专家访问分布 |
| `expert_location.py` | 管理专家位置 |
| `eplb_manager.py` | 负载均衡管理器 |

**工作原理**:
```python
class EPLBManager:
    def rebalance(self):
        """定期重新平衡专家位置"""

        # 1. 收集专家访问统计
        distribution = self._get_expert_distribution()

        # 2. 计算最优专家布局
        new_location = self._compute_optimal_location(distribution)

        # 3. 迁移专家权重
        self._migrate_experts(new_location)
```

---

### 2.3 MoE 学习检查清单

- [ ] 理解 Router 如何计算 topk_ids 和 topk_weights
- [ ] 理解 Token Dispatcher 的 dispatch/combine 流程
- [ ] 跟踪一个 token 的完整 MoE 路径
- [ ] 理解 Fused MoE kernel 的优化点
- [ ] 理解专家并行（EP）的通信模式
- [ ] 理解 All-to-All 在 EP 中的作用

---

## 三、代码阅读路线图

### 3.1 Attention 优化代码路线

```
第一天: 基础架构
├── layers/radix_attention.py (100 行)
└── layers/attention/base_attn_backend.py (80 行)

第二天: FlashInfer 后端
├── layers/attention/flashinfer_backend.py (重点)
│   - DecodeMetadata, PrefillMetadata
│   - forward_extend(), forward_decode()
└── 对比阅读 flashinfer 库文档

第三天: KV Cache 管理
├── mem_cache/memory_pool.py
│   - ReqToTokenPool
│   - TokenToKVPool
└── mem_cache/radix_cache.py (Radix Tree 实现)

第四天: 高级主题
├── layers/attention/triton_backend.py
└── layers/attention/triton_ops/ (Triton kernel)
```

### 3.2 MoE 并行代码路线

```
第一天: MoE 基础
├── layers/moe/router.py (Triton router kernel)
└── layers/moe/topk.py (Top-K 选择)

第二天: Fused MoE Kernel
├── layers/moe/fused_moe_triton/fused_moe.py (核心)
└── layers/moe/fused_moe_triton/layer.py (层定义)

第三天: Token Dispatcher
├── layers/moe/token_dispatcher/base.py (接口)
├── layers/moe/token_dispatcher/standard.py (单 GPU)
└── layers/moe/token_dispatcher/deepep.py (高性能)

第四天: 专家并行
├── distributed/parallel_state.py (并行状态)
├── eplb/expert_distribution.py (分布记录)
└── eplb/eplb_manager.py (负载均衡)
```

---

## 四、调试与实验技巧

### 4.1 单元测试入口

```bash
# Attention 相关测试
python -m pytest tests/test_attention.py -v

# MoE 相关测试
python -m pytest tests/test_moe.py -v
```

### 4.2 关键日志开关

```python
# 在代码中添加调试日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 或使用环境变量
export SGLANG_LOG_LEVEL=DEBUG
```

### 4.3 性能分析

```python
# 使用 NVTX 标记
from sglang.srt.utils.nvtx_pytorch_hooks import register_nvtx_hooks

# 使用 PyTorch Profiler
with torch.profiler.profile() as p:
    model.forward(...)
print(p.key_averages())
```

---

## 五、推荐学习资源

### 论文

| 论文 | 主题 |
|------|------|
| FlashAttention | 高效注意力计算 |
| FlashAttention-2 | 更快的注意力实现 |
| FlashInfer | 可变长注意力 |
| Mixtral | MoE 模型架构 |
| Switch Transformer | 专家并行基础 |

### 博客/教程

- FlashInfer 官方文档: https://flashinfer.ai/
- Triton 编程教程: https://triton-lang.org/
- SGLang 官方文档: https://sgl-project.github.io/

---

*最后更新: 2026-03-16*
