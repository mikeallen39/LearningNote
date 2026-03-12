# CUDA Kernel 优化

## 1. Router GEMM 优化 (dsv3_router_gemm)

### 1.1 源码位置

`glm4_moe_lite.py:176-194`

### 1.2 完整代码

```python
# glm4_moe_lite.py:160-194
class Glm4MoeLiteGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()
        self.is_nextn = is_nextn
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((config.n_routed_experts), dtype=torch.float32)
        )

    def forward(self, hidden_states, gemm_output_zero_allocator: BumpAllocator = None):
        # H100+ 小批量专用优化 kernel
        if (
            _is_cuda
            and not self.is_nextn
            and hidden_states.shape[0] < 4           # 条件1: 批量 < 4
            and hidden_states.shape[1] == 7168       # 条件2: GLM-4 隐藏维度
            and self.weight.shape[0] == 256          # 条件3: 256 个专家
            and _device_sm >= 90                     # 条件4: SM90+ (H100+)
        ):
            from sgl_kernel import dsv3_router_gemm
            logits = dsv3_router_gemm(hidden_states, self.weight).to(hidden_states.dtype)
        else:
            # 标准矩阵乘法
            logits = F.linear(hidden_states, self.weight, None)
        return logits
```

### 1.3 条件判断详解

| 条件 | 源码 | 原因 |
|------|------|------|
| `_is_cuda` | `utils/common.py:133-134` | 仅 CUDA 支持 |
| `not self.is_nextn` | `glm4_moe_lite.py:168` | NextN 层禁用此优化 |
| `shape[0] < 4` | 运行时 | 小批量时内存带宽瓶颈 |
| `shape[1] == 7168` | 运行时 | GLM-4 专用维度 |
| `weight.shape[0] == 256` | 模型配置 | 针对 256 专家优化 |
| `_device_sm >= 90` | `utils/common.py:263-267` | H100+ 专用指令 |

### 1.4 sgl_kernel 接口

```python
# sgl_kernel 中的 Python 绑定
from sgl_kernel import dsv3_router_gemm

# 函数签名 (推测，来自 sgl-kernel 库)
def dsv3_router_gemm(
    hidden_states: torch.Tensor,  # [batch, hidden_size]
    weight: torch.Tensor,         # [n_experts, hidden_size]
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:                # [batch, n_experts]
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

### 4.1 源码位置

`layers/moe/topk.py:202-369`

### 4.2 TopK 类定义

```python
# layers/moe/topk.py:202-252
class TopK(MultiPlatformOp):
    """
    Parameters:
    --top_k: 选中的专家总数，包括融合的共享专家
    --num_fused_shared_experts: 融合的共享专家数
    --routed_scaling_factor: 路由专家的缩放因子
    """
    def __init__(
        self,
        top_k: int,
        *,
        layer_id: Optional[int] = None,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        renormalize: bool = True,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        quant_config: Optional[QuantizationConfig] = None,
        routed_scaling_factor: Optional[float] = None,
        apply_routed_scaling_factor_on_output: Optional[bool] = False,
        output_format: Optional[TopKOutputFormat] = None,
        fused_shared_experts_scaling_factor: Optional[float] = None,
    ):
```

### 4.3 Forward 实现

```python
# layers/moe/topk.py:272-321
def forward_cuda(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> TopKOutput:
    # 根据后端选择输出格式
    if self.topk_config.output_format is not None:
        output_format = self.topk_config.output_format
    elif get_moe_runner_backend().is_triton_kernels():
        output_format = TopKOutputFormat.TRITON_KERNEL
    elif (
        get_moe_runner_backend().is_flashinfer_trtllm()
        or get_moe_runner_backend().is_flashinfer_mxfp4()
    ):
        output_format = TopKOutputFormat.BYPASSED
    else:
        output_format = TopKOutputFormat.STANDARD

    # 使用 symmetric memory 优化通信
    with use_symmetric_memory(
        get_tp_group(), disabled=not is_allocation_symmetric()
    ):
        topk_output = select_experts(...)
    return topk_output
```

### 4.4 sgl_kernel 接口

```python
# 来自 sgl_kernel
from sgl_kernel import topk_softmax, moe_fused_gate

# Top-K Softmax
def topk_softmax(
    topk_weights: torch.Tensor,  # [batch, top_k] 输出
    topk_ids: torch.Tensor,      # [batch, top_k] 输出
    gating_output: torch.Tensor, # [batch, n_experts] 输入
    renormalize: bool,
) -> None:

# DeepSeek/GLM 专用融合 Top-K
def moe_fused_gate(
    gating_output: torch.Tensor,   # [batch, n_experts]
    correction_bias: torch.Tensor, # [n_experts]
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:  # (weights, ids)
```

### 4.5 使用位置

```python
# glm4_moe_lite.py:251-266
self.topk = TopK(
    top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
    layer_id=self.layer_id,
    renormalize=config.norm_topk_prob,
    use_grouped_topk=True,
    num_expert_group=config.n_group,
    num_fused_shared_experts=self.num_fused_shared_experts,
    topk_group=config.topk_group,
    correction_bias=self.gate.e_score_correction_bias,
    quant_config=quant_config,
    routed_scaling_factor=self.routed_scaling_factor,
    apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
    output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
)
```

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
