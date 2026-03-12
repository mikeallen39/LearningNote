# RoPE (Rotary Position Embedding) 手撕实现

旋转位置编码（RoPE）是一种将位置信息注入 Transformer 模型的优雅方法，广泛应用于 LLaMA、GPT-NeoX 等大语言模型。

## 核心思想

RoPE 通过**旋转向量**来编码位置信息。对于位置 `m` 的 token，其嵌入向量会被旋转 `m × θ` 角度，其中 `θ` 是与维度相关的频率。

### 数学原理

对于维度 `d` 的嵌入向量，将其分为 `d/2` 对：

```
(x_0, x_1), (x_2, x_3), ..., (x_{d-2}, x_{d-1})
```

每一对 `(x_{2i}, x_{2i+1})` 进行如下旋转：

```
[x'_{2i}]     [cos(mθ_i)  -sin(mθ_i)] [x_{2i}]
[x'_{2i+1}] = [sin(mθ_i)   cos(mθ_i)] [x_{2i+1}]
```

其中频率 `θ_i = base^(-2i/d)`，通常 `base = 10000`。

### 相对位置特性

RoPE 的核心优势：**两个 token 之间的点积只依赖于它们的相对位置差**。

证明：设 token A 在位置 m，token B 在位置 n，则：
- 旋转后的点积只与 `(m - n)` 有关
- 这使得模型能够自然地学习相对位置关系

## 代码结构

```
rope/
├── rope.py      # PyTorch 实现
└── README.md    # 本文档
```

### 主要实现

1. **`RotaryEmbedding`** - HuggingFace Transformers 风格
   - 使用 `cos/sin` 分离计算
   - 支持 `position_ids` 动态位置
   - 维度顺序: `[batch, num_heads, seq_len, head_dim]`

2. **`LlamaRotaryEmbedding`** - LLaMA 风格
   - 使用复数形式 `e^(iθ)`
   - 更简洁高效
   - 维度顺序: `[batch, seq_len, num_heads, head_dim]`

## 使用方法

### 基础用法

```python
from rope import RotaryEmbedding

# 初始化（通常 head_dim = hidden_size / num_heads）
rope = RotaryEmbedding(dim=64, max_position_embeddings=2048)

# 应用 RoPE
q_embed, k_embed = rope(q, k)  # q, k: [batch, num_heads, seq_len, head_dim]
```

### LLaMA 风格

```python
from rope import LlamaRotaryEmbedding

rope = LlamaRotaryEmbedding(dim=64, max_position_embeddings=2048)

# 注意维度顺序不同
q_embed, k_embed = rope(q, k)  # q, k: [batch, seq_len, num_heads, head_dim]
```

### 结合 Attention 使用

```python
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, position_ids=None):
        batch, seq_len, _ = x.shape

        # 计算 Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        # 转置为 [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 应用 RoPE
        q, k = self.rope(q, k, position_ids=position_ids)

        # 计算 attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)

        # 输出
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)
```

## 关键实现细节

### 1. 频率计算

```python
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
# 结果: [1/base^0, 1/base^(2/d), 1/base^(4/d), ..., 1/base^((dim-2)/d)]
```

### 2. 复数形式（LLaMA 风格）

```python
# 将实数对视为复数
x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

# 旋转角度 θ 对应的复数 e^(iθ)
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

# 复数乘法实现旋转
x_out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
```

### 3. rotate_half 方式（HuggingFace 风格）

```python
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

# 应用旋转
x_out = x * cos + rotate_half(x) * sin
```

## 运行测试

```bash
python rope.py
```

## 参考文献

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [HuggingFace Transformers - RoPE Implementation](https://github.com/huggingface/transformers)
