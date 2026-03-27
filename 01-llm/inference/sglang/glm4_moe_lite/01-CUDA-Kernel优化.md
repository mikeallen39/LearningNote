# CUDA Kernel 优化

## 1. Router GEMM 优化 (dsv3_router_gemm)

> **CUDA 源码**: `sgl-kernel/csrc/gemm/dsv3_router_gemm_*.cu`
>
> **适配来源**: NVIDIA TensorRT-LLM 的 DeepSeek-V3 低延迟 kernel

### 1.1 问题背景

Router GEMM 计算: `logits = hidden_states @ weight.T`

| 参数 | DeepSeek-V3 | GLM-4 MoE |
|------|-------------|-----------|
| M (tokens) | 1~16 | 1~4 |
| K (hidden_dim) | 7168 | 7168 |
| N (experts) | 256 / 384 | 256 |

**特点**: M 极小，延迟敏感，精度要求高（影响专家选择）

### 1.2 为什么比 cuBLAS 快？

| 因素 | cuBLAS | dsv3_router_gemm |
|------|--------|------------------|
| 批量优化 | 大 M 优先 | 小 M 延迟优先 |
| Kernel 选择 | 运行时选择 | 编译期特化 |
| 循环展开 | 部分 | 完全展开 |
| 归约方式 | 通用树形 | Butterfly 无冲突 |
| 启动开销 | 标准 | PDL 流水线 |

**综合提升**: 小 M 场景 2-3x

### 1.3 核心优化技术

#### (1) 编译期常量展开 + 模板特化

```cpp
// 所有维度编译期确定
template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
__global__ void router_gemm_kernel(...);

// 为每个 M 值单独实例化 (1-16)
template void invoke<bf16, 1, 256, 7168>(...);
template void invoke<bf16, 2, 256, 7168>(...);
// ... 到 16
```

**收益**: 编译器完全展开循环，消除分支，优化寄存器分配

#### (2) 向量化内存访问

```cpp
// 一次加载 8 个 bf16 (128-bit)
uint4 vec = *reinterpret_cast<uint4 const*>(ptr);
constexpr int VPT = 16 / sizeof(bf16);  // = 8
```

#### (3) Warp 级 Butterfly 归约

```cpp
// 5 步完成 32 线程归约，无 shared memory 冲突
float sum = acc;
sum += __shfl_xor_sync(0xffffffff, sum, 16);
sum += __shfl_xor_sync(0xffffffff, sum, 8);
sum += __shfl_xor_sync(0xffffffff, sum, 4);
sum += __shfl_xor_sync(0xffffffff, sum, 2);
sum += __shfl_xor_sync(0xffffffff, sum, 1);
```

| 方法 | 步数 | Shared Memory |
|------|------|---------------|
| 树形归约 | 5 | 需要 |
| Butterfly | 5 | **不需要** |

#### (4) 每专家一个 Block

```cpp
int n_idx = blockIdx.x;  // 每个 block 负责一个专家
// gridDim = num_experts, blockDim = 128
```

**优势**: 完美并行，输出直接写入最终位置，无原子操作

#### (5) Programmatic Dependent Launch (PDL)

```cpp
#if __CUDA_ARCH__ >= 90
  asm volatile("griddepcontrol.wait;");
  // ... kernel 计算 ...
  asm volatile("griddepcontrol.launch_dependents;");
#endif
```

**效果**: Hopper 架构下，当前 kernel 完成前就开始调度下一个

#### (6) FP32 累加精度

```cpp
float acc[kNumTokens] = {};  // FP32 累加器
acc[m_idx] += a * b;         // BF16 * BF16 -> FP32
```

**原因**: Router 输出影响专家选择，精度敏感

### 1.4 触发条件

```python
# glm4_moe_lite.py:176-194
if (
    _is_cuda
    and not self.is_nextn
    and hidden_states.shape[0] < 4           # 小批量
    and hidden_states.shape[1] == 7168       # GLM-4 隐藏维度
    and self.weight.shape[0] == 256          # 256 专家
    and _device_sm >= 90                     # Hopper 架构
):
    logits = dsv3_router_gemm(hidden_states, self.weight)
```

| 条件 | 原因 |
|------|------|
| `shape[0] < 4` | 小批量延迟敏感场景 |
| `shape[1] == 7168` | 编译期特化维度 |
| `weight.shape[0] == 256` | 编译期特化专家数 |
| `_device_sm >= 90` | 需要 Hopper PDL 指令 |

### 1.5 sgl_kernel 接口

```python
from sgl_kernel import dsv3_router_gemm

def dsv3_router_gemm(
    hidden_states: torch.Tensor,  # [M, K] BF16
    router_weights: torch.Tensor, # [N, K] BF16
    out_dtype: torch.dtype = torch.bfloat16,  # 或 float32
) -> torch.Tensor:                # [M, N]
```

### 1.6 代码结构

```
sgl-kernel/csrc/gemm/
├── dsv3_router_gemm_entry.cu      # 入口，参数校验，LoopUnroller
├── dsv3_router_gemm_float_out.cu  # FP32 输出版本
└── dsv3_router_gemm_bf16_out.cu   # BF16 输出版本
```

## 2. 激活函数融合 Kernel (SiluAndMul)

### 2.1 源码位置

`layers/activation.py:63-97`

### 2.2 完整代码

```python
# layers/activation.py:63-97
class SiluAndMul(MultiPlatformOp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # RL 训练时使用原生实现
        if get_global_server_args().rl_on_policy_target is not None:
            self._forward_method = self.forward_native

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch 原生实现"""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """CUDA 融合 kernel"""
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)  # 调用 sgl_kernel
        return out

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        """CPU AMX 优化"""
        if _is_cpu_amx_available:
            out = torch.ops.sgl_kernel.silu_and_mul_cpu(x)
            return out
        else:
            return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        """NPU 优化"""
        out = torch_npu.npu_swiglu(x)
        return out
```

### 2.3 sgl_kernel 接口

```python
# 来自 sgl_kernel 的 Python 绑定
from sgl_kernel import silu_and_mul

# 函数签名
def silu_and_mul(
    input: torch.Tensor,   # [batch, d * 2]
    output: torch.Tensor,  # [batch, d] (预分配)
) -> None:  # 原地写入 output
```

### 2.4 使用位置

```python
# glm4_moe_lite.py:120, 152
self.act_fn = SiluAndMul()
...
x = self.act_fn(gate_up)  # gate_up: [batch, intermediate * 2]
```

## 3. RMSNorm Kernel

### 3.1 源码位置

`layers/layernorm.py:89-299`

### 3.2 核心接口

```python
# layers/layernorm.py:89-149
class RMSNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        cast_x_before_out_mul: bool = False,
        fp32_residual: bool = False,
        has_weight: bool = True,
        weight_dtype: Optional = None,
        override_orig_dtype: Optional = None,
    ) -> None:
        super().__init__()
        self.has_weight = has_weight
        if self.has_weight:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            # 融合 Add + RMSNorm
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out
```

### 3.3 sgl_kernel 接口

```python
# 来自 sgl_kernel
from sgl_kernel import fused_add_rmsnorm, rmsnorm

# RMSNorm
def rmsnorm(
    x: torch.Tensor,      # [batch, hidden_size]
    weight: torch.Tensor, # [hidden_size]
    eps: float,
) -> torch.Tensor:        # [batch, hidden_size]

# Fused Add + RMSNorm (原地修改)
def fused_add_rmsnorm(
    x: torch.Tensor,       # [batch, hidden_size] (输入，将被修改为归一化结果)
    residual: torch.Tensor,# [batch, hidden_size] (残差，将被修改为新残差)
    weight: torch.Tensor,  # [hidden_size]
    eps: float,
) -> None:
```

### 3.4 使用位置

```python
# glm4_moe_lite.py:401-404
self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = RMSNorm(
    config.hidden_size, eps=config.rms_norm_eps
)
```

## 4. Top-K Kernel

> **核心问题**: MoE 场景下的 Top-K 远比普通 Top-K 复杂，需要融合优化

### 4.1 为什么 MoE Top-K 需要优化？

#### 普通视角 vs MoE 实际情况

**你以为的 Top-K**:
```python
# 简单：找一个数组中最大的 K 个元素
values, indices = torch.topk(logits, k=8)  # 一行搞定
```

**MoE 实际的 Grouped Top-K** (以 DeepSeek/GLM 为例):
```python
# grouped_topk_gpu 的实际逻辑 (topk.py:494-544)

# 1. Softmax 归一化
scores = torch.softmax(gating_output, dim=-1)  # [M, 256]

# 2. 分组最大值（256 专家分成 8 组，每组 32 专家）
group_scores = scores.view(M, 8, 32).max(dim=-1).values  # [M, 8]

# 3. 找 Top-K 组
group_idx = torch.topk(group_scores, k=topk_group)  # [M, topk_group]

# 4. 创建组掩码
group_mask = torch.zeros_like(group_scores)
group_mask.scatter_(1, group_idx, 1)

# 5. 扩展掩码到专家维度
score_mask = group_mask.unsqueeze(-1).expand(...).reshape(...)

# 6. 掩码过滤
tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)

# 7. 最终 Top-K
topk_weights, topk_ids = torch.topk(tmp_scores, k=topk)

# 8. 可选：应用 correction_bias
# 9. 可选：应用 scaling_factor
# 10. 可选：renormalize
```

**问题**: 原生实现涉及 **5-10 个独立 kernel**，每个都有启动开销和内存访问

#### Grouped Top-K 算法说明

```
256 专家分成 8 组，每组 32 专家

约束: 每个 token 最多从 topk_group 个组中选择专家
目的: 负载均衡，避免某些组过载

示例: 选 8 个专家，topk_group=4
→ 先选 4 个最"强"的组
→ 再从这 4 组中选 8 个专家
```

#### 原生实现 vs 融合 Kernel 性能对比

| 指标 | 原生 PyTorch | moe_fused_gate |
|------|-------------|----------------|
| Kernel 启动次数 | 5-10 次 | **1 次** |
| 中间张量 | 多个临时张量 | 零临时张量 |
| 内存访问 | 多次读写 | **1 次读 + 1 次写** |
| 延迟 | ~50-100 μs | ~5-10 μs |

### 4.2 源码位置

`layers/moe/topk.py:202-369`

### 4.3 moe_fused_gate 融合 Kernel

**触发条件**:
```python
# topk.py:802-815
if (
    _is_cuda
    and not torch_native
    and correction_bias is not None
    and num_fused_shared_experts == 0
    and experts_per_group <= 32
    and is_power_of_two(num_experts)
):
    topk_weights, topk_ids = moe_fused_gate(...)
```

**一次 Kernel 完成的操作**:
```python
moe_fused_gate(
    gating_output,          # [M, 256] 输入
    correction_bias,        # [256] 可学习偏置
    num_expert_group,       # 8 组
    topk_group,             # 选 4 组
    topk,                   # 选 8 专家
    num_fused_shared_experts,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
) -> (topk_weights, topk_ids)  # [M, 8]
```

### 4.4 TopK 类定义

```python
# layers/moe/topk.py:202-252
class TopK(MultiPlatformOp):
    def __init__(
        self,
        top_k: int,
        *,
        use_grouped_topk: bool = False,      # 是否使用分组 Top-K
        topk_group: Optional[int] = None,    # 选择的组数
        num_expert_group: Optional[int] = None,  # 专家分组数
        renormalize: bool = True,
        correction_bias: Optional[torch.Tensor] = None,  # 修正偏置
        routed_scaling_factor: Optional[float] = None,   # 缩放因子
        ...
    ):
```

### 4.5 sgl_kernel 接口

```python
from sgl_kernel import topk_softmax, moe_fused_gate

# 标准 Top-K + Softmax
def topk_softmax(
    topk_weights: torch.Tensor,  # [M, top_k] 输出
    topk_ids: torch.Tensor,      # [M, top_k] 输出
    gating_output: torch.Tensor, # [M, n_experts] 输入
    renormalize: bool,
) -> None:

# DeepSeek/GLM 专用融合 Top-K (分组 + 偏置 + 缩放)
def moe_fused_gate(
    gating_output: torch.Tensor,   # [M, n_experts]
    correction_bias: torch.Tensor, # [n_experts]
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:  # (weights, ids)
```

### 4.6 使用位置

```python
# glm4_moe_lite.py:251-266
self.topk = TopK(
    top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
    use_grouped_topk=True,
    num_expert_group=config.n_group,      # 如 8 组
    topk_group=config.topk_group,          # 如选 4 组
    correction_bias=self.gate.e_score_correction_bias,
    routed_scaling_factor=self.routed_scaling_factor,
)
```

### 4.7 优化要点总结

| 因素 | 说明 |
|------|------|
| **算法复杂** | 不是简单 Top-K，是分组 Top-K + 偏置 + 缩放 |
| **操作链长** | 原生实现需要 5-10 个 kernel 串联 |
| **内存开销** | 多个中间张量，多次内存读写 |
| **延迟敏感** | 每个 token 都要执行，在关键路径上 |
| **批量小** | Decode 阶段 M 很小，kernel 启动开销占比高 |

## 5. MultiPlatformOp 基类

### 5.1 源码位置

`layers/utils.py`

### 5.2 自动分发机制

```python
class MultiPlatformOp(nn.Module):
    """根据平台自动选择 forward 方法"""

    def forward(self, *args, **kwargs):
        if _is_cuda:
            return self.forward_cuda(*args, **kwargs)
        elif _is_hip:
            return self.forward_hip(*args, **kwargs)
        elif _is_npu:
            return self.forward_npu(*args, **kwargs)
        elif _is_cpu:
            return self.forward_cpu(*args, **kwargs)
        else:
            return self.forward_native(*args, **kwargs)
```

---

## 6. 通用优化原则总结

### 6.1 何时考虑特化 Kernel？

| 条件 | 说明 |
|------|------|
| 问题规模固定/有限 | 可编译期特化，消除运行时开销 |
| 延迟敏感 | Kernel 启动开销占比高 |
| 特殊内存模式 | 通用 kernel 非最优访问 |
| 特定硬件特性 | 如 PDL、Tensor Core、新指令 |

### 6.2 优化优先级

```
1. 内存访问优化（带宽为王）
   └── 向量化加载、合并访问、Shared Memory 复用
2. 计算优化
   └── 循环展开、寄存器复用、指令融合
3. 并行策略
   └── Block/Warp 映射，消除同步
4. 架构特定
   └── PDL、Tensor Core、Warp Shuffle
```

### 6.3 调优 Checklist

- [ ] 内存访问能否向量化？(uint4, float4)
- [ ] 循环边界能否编译期确定？
- [ ] 归约能否用 shuffle 指令？
- [ ] Block 映射是否消除跨块同步？
- [ ] 是否利用了新架构特性？(SM90+)

### 6.4 性能分析方法

```bash
# NVTX 标记
nsys profile -o profile python your_script.py

# Nsight Compute 分析单个 kernel
ncu --set full python your_script.py

# PyTorch Profiler
with torch.profiler.profile() as p:
    model.forward(...)
p.key_averages().table()
```

---

*最后更新: 2026-03-16*
