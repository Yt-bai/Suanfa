# Softmax 概念
**softmax 把一些杂乱的分数变成概率，每个概率值都在 0~1 之间，总和 = 1。**

所以，其实Softmax只是一个固定的映射变换，也是为了提高稳定性，但是不是哪里都能加，只有两个地方。

值得注意的是，**任何关于提高Transformer稳定性的位置，都可以加上Layernorm这个稳压器。**


| 对比   | LayerNorm       | Softmax             |
| ---- | --------------- | ------------------- |
| 作用   | 稳定数值            | 转成概率                |
| 输出   | 不一定 0~1         | 一定 0~1              |
| 总和   | 不要求为 1          | 总和为 1               |
| 用在哪里 | Transformer 中间层 | Attention 权重 / 输出概率 |
| 目的   | 方便训练            | 做选择/分配权重            |

# Softmax 位置
## 1.Attention 里的 Softmax，位置在 QK 打分之后，乘 V 之前。（因为Softmax的输入只和QK有关）**
这里的Softmax在当前样本的 **seq_len** 维度上做。比如一句话有 5 个 token：[我, 喜欢, 机器, 学习, EOS]

对于每一个 token，Attention Softmax 会在这 5 个 token 上分配权重：

当前 token 应该看：
```text
我 多少
喜欢 多少
机器 多少
学习 多少
EOS 多少
```
所以它处理的是：**同一个样本内部的 token 之间关系**，即**当前Token应该更关注这个seq_length列的哪个其他Token？对其他Token的注意力应该分配多少？**

## 2.输出层里的 Softmax，位置在 最后一个 Transformer block 之后，Linear/LM Head 之后。
这里的Softmax是在词表 **vocab_size** 维度上做。比如词表有 128000 个 token，模型最后会给每个词一个 logit：
```
“我”
“你”
“的”
“猫”
“apple”
...
```

然后 Softmax 把这 128000 个分数变成概率：

下一个 token 是哪个？

形状通常是：

[batch, seq_len, vocab_size]

Softmax 作用在 vocab_size 上。所以它处理的是，**模型接下来该生成哪个token。**

**整体结构：**

```text
输入 token ids
↓
Token Embedding
↓
Transformer Block 1
  ↓
  Attention
  ↓
  FFN / MLP
↓
Transformer Block 2
  ↓
  Attention
  ↓
  FFN / MLP
↓
...
↓
Transformer Block N
  ↓
  Attention
  ↓
  FFN / MLP
↓
最终 hidden states
↓
Final LayerNorm / RMSNorm
↓
LM Head / Linear
↓
logits
↓
Softmax
↓
next token
```
其中，
```text
FFN/MLP：在每一层 Transformer block 里面
LM Head：在所有 block 之后，负责映射到词表
Softmax：最后把词表 logits 变成概率
```

一次 forward/backward 处理的是：**一个batch_size所有样本数，每个样本含有seq_length个token，每个token用d_model个数字来编码表示。**

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

**整体结构中的Softmax嵌入：**
```
输入 hidden states: x
shape: [batch, seq_len, d_model]

↓ Linear 投影

Q = xWq
K = xWk
V = xWv

↓ 计算 token 之间相关性

scores = QKᵀ / sqrt(d_head)
shape: [batch, heads, seq_len, seq_len]

↓ Softmax

attention_weights = softmax(scores)
shape: [batch, heads, seq_len, seq_len]

↓ 用权重混合 V

attention_output = attention_weights @ V
shape: [batch, heads, seq_len, d_head]

↓ 合并 heads

concat heads
shape: [batch, seq_len, d_model]

↓ Output Projection

out = attention_output Wo

↓ 残差连接

x = x + out

↓ 后面进入 MLP / FFN
```


logits 是模型给词表（vocab_size）里每个候选 token 打的原始分数；
