### 用尽可能少的资源训练出尽可能好的模型
资源 --> 算力、内存，有时也包括数据

**我们的目的：最大化训练的计算效率**

**1.用1024张H100，在15万亿个Token上，训练一个 700亿 参数量的模型，需要多长时间？**

```text
总的浮点运算数 total_flops = 参数量 *6 *token数
```
<img width="1542" height="741" alt="ea680eddb942f76a8976db2dbcfec832" src="https://github.com/user-attachments/assets/85edd2ea-bae0-4854-8da7-28ede79a9b39" />

**2.训练大模型的计算精度？**

目前普遍做法：混合精度训练 --> 一部分计算用一种精度，另一部分计算用另一种精度。

一般，BF16（bfloat16）用于参数、激活值和梯度，优化器状态使用FP32。若要启用混合精度训练，Pytorch提供了automatic mixed precision（AMP）library。

**3.einops -- Pytorch的张量操作库**

einops 是Pytorch中的一个非常流行的张量（Tensor）操作库，它的目标是让张量的 reshape、transpose、flatten、repeat 等操作变得更加直观、可读。

PyTorch 原生代码里，经常会看到：
```text
x = x.view(batch_size, -1)
x = x.permute(0, 2, 3, 1)
x = x.reshape(...)
```
维度稍微复杂一点就很难读。而用 einops 可以直接用字符串描述维度变换。

**4.了解特定某种硬件能提供多少浮点运算次数 & 特定模型需要消耗多少浮点运算次数（也就是模型规模）**

特定模型需要消耗多少浮点运算是指：让某个大语言模型完成一次训练或一次推理，大约需要做多少次浮点数计算。

分为训练FLOPS和推理FLOPS，也就是大模型为了处理输入、生成输出，需要做多少计算。
<img width="2321" height="1279" alt="15f1e0fb7035ad1f651e0e92514e2a0c" src="https://github.com/user-attachments/assets/8cb0bf8a-ae95-473b-971a-ac3c078d48b2" />

**5.小总结**
1.矩阵乘加运算是计算的主力

2.每秒能跑多少次浮点运算取决于硬件性能和数据类型

3.MFU=实际浮点运算速度/理论速度
<img width="1473" height="270" alt="image" src="https://github.com/user-attachments/assets/cf0952b1-bccb-462d-97c5-16b246c4c05b" />

**6.计算吞吐量FLOP/s和内存Memory读写吞吐量bytes/s**

<img width="765" height="666" alt="image" src="https://github.com/user-attachments/assets/013955c7-66cd-4c3a-b940-d2a449f4576b" />
<img width="545" height="150" alt="image" src="https://github.com/user-attachments/assets/bac3f3dd-3b77-4ac6-b65b-0f54e3b827d9" />
<img width="976" height="329" alt="image" src="https://github.com/user-attachments/assets/8d26aca0-e04c-4204-ba1a-f9f08fe2f9ed" />
**7.计算瓶颈和内存瓶颈**
<img width="1718" height="685" alt="image" src="https://github.com/user-attachments/assets/1d65d5b3-55b9-42ae-afc5-79d6279cc898" />

