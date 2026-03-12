# CUDA Graph

## 1. 概述

CUDA Graph 允许预录制一系列 CUDA 操作，减少 kernel 启动开销。

## 2. is_in_piecewise_cuda_graph

### 2.1 源码位置

`compilation/piecewise_context_manager.py`

### 2.2 函数定义

```python
# compilation/piecewise_context_manager.py
def is_in_piecewise_cuda_graph() -> bool:
    """检查当前是否在 Piecewise CUDA Graph 模式"""
    return _IN_PIECEWISE_CUDA_GRAPH
```

## 3. MoE 的 CUDA Graph 支持

### 3.1 FusedMoE 中的使用

```python
# layers/moe/fused_moe_triton/layer.py
def moe_forward_piecewise_cuda_graph_impl(
    hidden_states,
    topk_weights,
    topk_ids,
    router_logits,
    layer_id,
):
    """使用预录制的 CUDA Graph 执行 MoE forward"""
    ...
```

### 3.2 条件检查

```python
# 在 MoE forward 中检查
if is_in_piecewise_cuda_graph():
    return moe_forward_piecewise_cuda_graph_impl(...)
else:
    return self.forward_impl(...)
```

## 4. MLA 权重打包 (CUDA Graph 所需)

### 4.1 源码位置

`glm4_moe_lite.py:798-805`

### 4.2 post_load_weights 调用

```python
# glm4_moe_lite.py:805
self.post_load_weights(is_nextn=is_nextn, weight_names=None)
```

### 4.3 注释说明

```python
# glm4_moe_lite.py:798-804
# DeepseekV2AttentionMLA.forward_* expects post_load_weights() to populate
# per-layer packed weights like `w_kc`/`w_vc` (used during CUDA graph capture).
# GLM-Lite configs may not set `config.mla`, but this model always uses
# DeepseekV2AttentionMLA, so we must run the post-load processing.
```

## 5. qkv_latent_func 预计算

### 5.1 传递给 LayerCommunicator

```python
# glm4_moe_lite.py:414
self.layer_communicator = LayerCommunicator(
    ...
    qkv_latent_func=self.self_attn.prepare_qkv_latent,  # MLA QKV 预计算
)
```

### 5.2 作用

在 CUDA Graph 模式下预计算 QKV Latent，避免重复计算。
