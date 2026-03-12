# MoE 专项优化

## 1. 共享专家融合

### 1.1 源码位置

`glm4_moe_lite.py:211-215, 518-545`

### 1.2 配置

```python
# glm4_moe_lite.py:211-215
self.num_fused_shared_experts = (
    0
    if get_global_server_args().disable_shared_experts_fusion
    else config.n_shared_experts
)
```

### 1.3 融合条件判断

```python
# glm4_moe_lite.py:518-545
def determine_num_fused_shared_experts(
    self, architecture: str = "Glm4MoeLiteForCausalLM"
):
    self.num_fused_shared_experts = 0
    if get_global_server_args().disable_shared_experts_fusion:
        return

    disable_reason = None
    if (
        not _is_cuda
        or torch.cuda.get_device_capability("cuda") < (8, 0)
        or self.config.architectures[0] != architecture
        or self.config.n_shared_experts != 1  # 仅支持 1 个共享专家
    ):
        disable_reason = "Only GLM-4.5 or GLM-4.6 on NV-platform with capability >= 80..."
    elif get_moe_expert_parallel_world_size() > 1:
        disable_reason = "GLM-4.5 or GLM-4.6 cannot use shared experts fusion under EP..."

    if disable_reason is not None:
        get_global_server_args().disable_shared_experts_fusion = True
        self.num_fused_shared_experts = 0
        log_info_on_rank0(logger, f"{disable_reason} Shared experts fusion optimization is disabled.")
        return

    self.num_fused_shared_experts = self.config.n_shared_experts
```

### 1.4 权重重命名

```python
# glm4_moe_lite.py:576-595
if self.num_fused_shared_experts > 0:
    assert self.num_fused_shared_experts == 1

    def iter_weights_with_fused_shared_experts(
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        import re
        pattern = re.compile(
            r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(.+)$"
        )
        for name, weight in weights:
            match = pattern.match(name)
            if match:
                layer_id = int(match.group(1))
                suffix = match.group(2)
                # 重命名为最后一个路由专家
                name = f"model.layers.{layer_id}.mlp.experts.{self.config.n_routed_experts}.{suffix}"
            yield name, weight

    weights = iter_weights_with_fused_shared_experts(weights)
```

## 2. Glm4MoeLiteSparseMoeBlock

### 2.1 源码位置

`glm4_moe_lite.py:197-323`

### 2.2 完整初始化代码

```python
# glm4_moe_lite.py:197-323
class Glm4MoeLiteSparseMoeBlock(DeepseekV2MoE):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ):
        nn.Module.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = (
            0
            if get_global_server_args().disable_shared_experts_fusion
            else config.n_shared_experts
        )
        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # Gate
        self.gate = Glm4MoeLiteGate(
            config=config, prefix=add_prefix("gate", prefix), is_nextn=is_nextn
        )

        # Experts
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts
            + self.num_fused_shared_experts
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )

        # TopK
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

        # Shared experts (if not fused)
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)  # EP 场景禁用 TP
                    if get_moe_a2a_backend().is_deepep()
                    or get_moe_a2a_backend().is_mooncake()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else {}
                ),
            )

        self.top_k = config.num_experts_per_tok

        # EP 配置
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
        )
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()
```

## 3. TopK 配置

### 3.1 源码位置

`glm4_moe_lite.py:251-266`

### 3.2 完整配置

```python
self.topk = TopK(
    top_k=config.num_experts_per_tok + self.num_fused_shared_experts,  # 包含融合的共享专家
    layer_id=self.layer_id,
    renormalize=config.norm_topk_prob,
    use_grouped_topk=True,            # 启用分组 Top-K
    num_expert_group=config.n_group,  # 专家分组数
    num_fused_shared_experts=self.num_fused_shared_experts,
    topk_group=config.topk_group,     # 每组选择的专家数
    correction_bias=self.gate.e_score_correction_bias,  # 纠偏偏置
    quant_config=quant_config,
    routed_scaling_factor=self.routed_scaling_factor,
    apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
    output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
)
```

## 4. Grouped Top-K 参数

### 4.1 参数说明

| 参数 | 配置键 | 说明 |
|------|--------|------|
| `top_k` | `num_experts_per_tok` | 每个 token 激活的专家数 |
| `num_expert_group` | `n_group` | 专家分组数 |
| `topk_group` | `topk_group` | 每组选择的专家数 |
| `renormalize` | `norm_topk_prob` | 是否归一化权重 |

### 4.2 选择逻辑

```
1. 计算每组得分 (组内最大值)
2. 选择 topk_group 个组
3. 在选中的组内选择专家
```

## 5. 纠偏偏置 (correction_bias)

### 5.1 源码位置

`glm4_moe_lite.py:172-174, 259`

### 5.2 Gate 中的定义

```python
# glm4_moe_lite.py:172-174
self.e_score_correction_bias = nn.Parameter(
    torch.empty((config.n_routed_experts), dtype=torch.float32)
)
```

### 5.3 TopK 中使用

```python
# glm4_moe_lite.py:259
correction_bias=self.gate.e_score_correction_bias,
```

## 6. 冗余专家支持

### 6.1 配置

```python
# glm4_moe_lite.py:240
num_experts=config.n_routed_experts
    + self.num_fused_shared_experts
    + get_global_server_args().ep_num_redundant_experts,  # 冗余专家数
```

### 6.2 EP 配置

```python
# glm4_moe_lite.py:306-309
self.num_experts = (
    config.n_routed_experts
    + get_global_server_args().ep_num_redundant_experts
)
```
