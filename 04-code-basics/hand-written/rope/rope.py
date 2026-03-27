"""
RoPE (Rotary Position Embedding) - PyTorch 主流实现

参考:
- LLaMA: https://github.com/facebookresearch/llama
- HuggingFace Transformers: https://github.com/huggingface/transformers
- 论文: RoFormer: Enhanced Transformer with Rotary Position Embedding

RoPE 核心思想:
    通过旋转矩阵将位置信息注入到 token 表示中，使得模型能够感知相对位置。

数学公式:
    对于位置 m 的 token，其第 i 个维度对 (2i, 2i+1) 会被旋转 θ_i * m 角度
    θ_i = base^(-2i/d)，其中 d 是维度大小

    旋转操作:
    [x'_{2i}]   [cos(mθ_i)  -sin(mθ_i)] [x_{2i}]
    [x'_{2i+1}] = [sin(mθ_i)   cos(mθ_i)] [x_{2i+1}]
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def precompute_freqs_cis(dim: int, end: int, base: float = 10000.0) -> torch.Tensor:
    """
    预计算旋转位置编码的复数形式频率 (LLaMA 风格)

    这种实现将 cos 和 sin 合并为复数形式，更加高效。

    Args:
        dim: 嵌入维度（必须是偶数）
        end: 最大序列长度
        base: 频率基数，默认 10000

    Returns:
        freqs_cis: [end, dim//2] 复数张量
    """
    # 计算频率 θ_i = base^(-2i/d), i = 0, 1, ..., dim//2-1
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # 生成位置索引 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)

    # 外积: [end, dim//2]
    freqs = torch.outer(t, freqs)

    # 转换为复数形式 e^(iθ) = cos(θ) + i*sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    调整频率张量的形状以便广播

    Args:
        freqs_cis: [seq_len, dim//2]
        x: [batch, seq_len, num_heads, dim//2]

    Returns:
        调整后的频率张量 [1, seq_len, 1, dim//2]
    """
    ndim = x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Query 和 Key 应用旋转位置编码 (LLaMA 风格，复数实现)

    这是目前主流的实现方式，使用复数乘法来高效实现旋转。

    Args:
        xq: [batch, seq_len, num_heads, head_dim] Query 张量
        xk: [batch, seq_len, num_heads, head_dim] Key 张量
        freqs_cis: [seq_len, head_dim//2] 预计算的复数频率

    Returns:
        xq_out: 应用 RoPE 后的 Query
        xk_out: 应用 RoPE 后的 Key
    """
    # 将实数张量转换为复数张量
    # [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, num_heads, head_dim//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 广播频率并应用复数乘法
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============= 另一种常见的实现方式 (rotate_half) =============

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将张量的一半维度旋转 90 度

    将 [x1, x2, x3, x4, ...] 转换为 [-x2, x1, -x4, x3, ...]

    这等价于将每对维度 (x_{2i}, x_{2i+1}) 旋转 90 度，即：
    [x'_{2i}]   [0  -1] [x_{2i}]
    [x'_{2i+1}] = [1   0] [x_{2i+1}]

    Args:
        x: [..., dim] 输入张量

    Returns:
        旋转后的张量
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码 (HuggingFace Transformers 风格)

    这是另一种主流实现，直接使用 cos 和 sin 值。
    公式: x' = x * cos(θ) + rotate_half(x) * sin(θ)

    Args:
        q: [batch, num_heads, seq_len, head_dim] Query 张量
        k: [batch, num_heads, seq_len, head_dim] Key 张量
        cos: [seq_len, head_dim] 余弦值
        sin: [seq_len, head_dim] 正弦值
        position_ids: [batch, seq_len] 位置索引（可选）

    Returns:
        q_embed: 应用 RoPE 后的 Query
        k_embed: 应用 RoPE 后的 Key
    """
    # 如果提供了 position_ids，根据位置选择对应的 cos/sin
    if position_ids is not None:
        cos = cos[position_ids]  # [batch, seq_len, head_dim]
        sin = sin[position_ids]  # [batch, seq_len, head_dim]

    # 调整维度以便广播: [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # 应用旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# ============= 完整的 RoPE 模块 =============

class RotaryEmbedding(nn.Module):
    """
    RoPE 位置编码模块 (参考 HuggingFace Transformers 实现)

    支持特性:
    - 自动预计算并缓存 cos/sin 值
    - 支持动态序列长度扩展
    - 支持 fp16/bf16 混合精度

    使用示例:
        rope = RotaryEmbedding(dim=64)
        q, k = rope(q, k, seq_len=128)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            dim: 头维度大小
            max_position_embeddings: 最大位置编码长度
            base: 频率基数
            device: 计算设备
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 计算频率: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算 cos 和 sin
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        预计算并缓存 cos 和 sin 值

        Args:
            seq_len: 序列长度
            device: 计算设备
            dtype: 数据类型
        """
        self.max_seq_len_cached = seq_len

        # 生成位置索引
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 计算位置与频率的外积: [seq_len, dim//2]
        freqs = torch.outer(t, self.inv_freq)

        # 拼接为完整维度: [seq_len, dim]
        # 这样每对维度使用相同的角度
        emb = torch.cat([freqs, freqs], dim=-1)

        # 缓存 cos 和 sin
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：对 Query 和 Key 应用旋转位置编码

        Args:
            q: [batch, num_heads, seq_len, head_dim] Query 张量
            k: [batch, num_heads, seq_len, head_dim] Key 张量
            seq_len: 序列长度（用于动态扩展）
            position_ids: [batch, seq_len] 位置索引

        Returns:
            q_embed: 应用 RoPE 后的 Query
            k_embed: 应用 RoPE 后的 Key
        """
        # 动态扩展序列长度
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=q.device, dtype=q.dtype)

        # 获取缓存的 cos 和 sin
        cos = self.cos_cached[:, :, :q.shape[2], :]
        sin = self.sin_cached[:, :, :q.shape[2], :]

        return apply_rotary_pos_emb(q, k, cos.squeeze(0).squeeze(0), sin.squeeze(0).squeeze(0), position_ids)


# ============= LLaMA 风格的 RoPE 模块 =============

class LlamaRotaryEmbedding(nn.Module):
    """
    LLaMA 风格的 RoPE 实现

    使用复数形式，更加简洁高效。
    这是目前大模型中最主流的实现方式。
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 预计算复数频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建缓存
        self._build_cache(device)

    def _build_cache(self, device: Optional[torch.device] = None):
        """
        构建 freqs_cis 缓存
        """
        t = torch.arange(self.max_position_embeddings, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # e^(iθ) = cos(θ) + i*sin(θ)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [batch, seq_len, num_heads, head_dim] 注意：LLaMA 的维度顺序
            k: [batch, seq_len, num_heads, head_dim]
            position_ids: [batch, seq_len]

        Returns:
            q_embed, k_embed
        """
        # 根据 position_ids 获取对应的频率
        if position_ids is not None:
            freqs_cis = self.freqs_cis[position_ids]
        else:
            freqs_cis = self.freqs_cis[:q.shape[1]]

        return apply_rotary_emb(q, k, freqs_cis)


# ============= 测试代码 =============

def test_rope():
    """测试 RoPE 实现"""
    print("=" * 60)
    print("RoPE (Rotary Position Embedding) 测试")
    print("=" * 60)

    # 参数设置
    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 创建随机 Query 和 Key
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print(f"输入形状: Q={q.shape}, K={k.shape}")

    # 测试 HuggingFace 风格的 RoPE
    print("\n" + "-" * 40)
    print("测试 HuggingFace 风格 RoPE")
    print("-" * 40)

    rope_hf = RotaryEmbedding(dim=head_dim, max_position_embeddings=512, device=device)
    q_hf, k_hf = rope_hf(q, k)

    print(f"输出形状: Q={q_hf.shape}, K={k_hf.shape}")
    print(f"Q 示例值 (第一个 token 前 4 维): {q_hf[0, 0, 0, :4]}")

    # 测试 LLaMA 风格的 RoPE (需要调整维度顺序)
    print("\n" + "-" * 40)
    print("测试 LLaMA 风格 RoPE")
    print("-" * 40)

    # LLaMA 使用 [batch, seq_len, num_heads, head_dim] 的维度顺序
    q_llama = q.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
    k_llama = k.transpose(1, 2)

    rope_llama = LlamaRotaryEmbedding(dim=head_dim, max_position_embeddings=512, device=device)
    q_llama_out, k_llama_out = rope_llama(q_llama, k_llama)

    print(f"输出形状: Q={q_llama_out.shape}, K={k_llama_out.shape}")
    print(f"Q 示例值 (第一个 token 前 4 维): {q_llama_out[0, 0, 0, :4]}")

    # 验证两种实现结果一致
    print("\n" + "-" * 40)
    print("验证两种实现的一致性")
    print("-" * 40)

    # 将 LLaMA 输出转回 HuggingFace 维度顺序
    q_llama_out_hf = q_llama_out.transpose(1, 2)

    diff = torch.abs(q_hf - q_llama_out_hf).max().item()
    print(f"两种实现的最大差异: {diff:.6e}")
    print(f"结果一致性: {'通过' if diff < 1e-5 else '失败'}")

    # 验证相对位置特性
    print("\n" + "-" * 40)
    print("验证 RoPE 的相对位置特性")
    print("-" * 40)

    # 创建相同的 token 在不同位置
    token = torch.randn(1, 1, 1, head_dim, device=device)
    tokens = token.expand(1, 1, 4, head_dim).clone()  # 复制 4 份

    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=10, device=device)

    # 获取不同位置的编码
    positions = torch.arange(4).unsqueeze(0)
    _, k_pos = rope(
        torch.zeros(1, 1, 4, head_dim, device=device),
        tokens,
        position_ids=positions
    )

    # 计算相邻位置的点积
    for i in range(3):
        dot = (k_pos[0, 0, i] * k_pos[0, 0, i+1]).sum().item()
        print(f"位置 {i} 和 {i+1} 的点积: {dot:.4f}")


if __name__ == "__main__":
    test_rope()
