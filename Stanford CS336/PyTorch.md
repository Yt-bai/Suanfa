### 用尽可能少的资源训练出尽可能好的模型
资源 --> 算力、内存，有时也包括数据

**我们的目的：最大化训练的计算效率**

**1.用1024张H100，在15万亿个Token上，训练一个 700亿 参数的模型，需要多长时间？**

```text
总的浮点运算数 total_flops = 参数量 *6 *token数
```
<img width="1542" height="741" alt="ea680eddb942f76a8976db2dbcfec832" src="https://github.com/user-attachments/assets/85edd2ea-bae0-4854-8da7-28ede79a9b39" />

**2.训练大模型的计算精度？**

目前普遍做法：混合精度训练 --> 一部分计算用一种精度，另一部分计算用另一种精度。

一般，BF16（bfloat16）用于参数、激活值和梯度，优化器状态使用FP32。若要启用混合精度训练，Pytorch提供了automatic mixed precision（AMP）library。

**3. einops -- Pytorch的张量操作库**

einops 是Pytorch中的一个非常流行的张量（Tensor）操作库，它的目标是让张量的 reshape、transpose、flatten、repeat 等操作变得更加直观、可读。

PyTorch 原生代码里，经常会看到：
```text
x = x.view(batch_size, -1)
x = x.permute(0, 2, 3, 1)
x = x.reshape(...)
```
维度稍微复杂一点就很难读。而用 einops 可以直接用字符串描述维度变换。
