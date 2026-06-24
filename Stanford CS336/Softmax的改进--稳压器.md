# Softmax 概念
**logits 是模型给词表（vocab_size）里每个候选 token 打的原始分数；softmax 把这些分数变成概率，每个值都在 0~1 之间，总和 = 1。**

所以，其实Softmax只是一个固定的映射变换，也是为了提高稳定性，但是不是哪里都能加，只能在模型输出logits之后。
值得注意的是，**任何关于提高Transformer稳定性的位置，都可以加上Layernorm这个稳压器。**
softmax 的输入主要由 QKᵀ 决定

| 对比   | LayerNorm       | Softmax             |
| ---- | --------------- | ------------------- |
| 作用   | 稳定数值            | 转成概率                |
| 输出   | 不一定 0~1         | 一定 0~1              |
| 总和   | 不要求为 1          | 总和为 1               |
| 用在哪里 | Transformer 中间层 | Attention 权重 / 输出概率 |
| 目的   | 方便训练            | 做选择/分配权重            |

```
输入 token ids
↓
Token Embedding
得到 x: [batch, seq_len, d_model]
↓
第 1 个 Transformer Block
  ↓
  LayerNorm / RMSNorm
  ↓
  生成 Q, K, V
  ↓
  scores = QKᵀ / sqrt(d_head)
  ↓
  Softmax
  得到 attention weights
  ↓
  attention weights @ V
  得到 attention output
  ↓
  Output Projection
  ↓
  残差连接
  ↓
  MLP / FFN
  ↓
  残差连接
↓
第 2 个 Transformer Block
↓
...
↓
第 N 个 Transformer Block
↓
最终 hidden states
shape: [batch, seq_len, d_model]
↓
Final LayerNorm / RMSNorm
↓
LM Head / Linear
logits: [batch, seq_len, vocab_size]
↓
Softmax
probs: [batch, seq_len, vocab_size]
↓
选择下一个 token
```
