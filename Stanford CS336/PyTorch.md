## 用尽可能少的资源训练出尽可能好的模型
资源 --> 算力、内存，有时也包括数据

**我们的目的：最大化训练的计算效率**

### **1.用1024张H100，在15万亿个Token上，训练一个 700亿 参数量的模型，需要多长时间？**

```text
总的浮点运算数 total_flops = 参数量 *6 *token数
```
<img width="1542" height="741" alt="ea680eddb942f76a8976db2dbcfec832" src="https://github.com/user-attachments/assets/85edd2ea-bae0-4854-8da7-28ede79a9b39" />

### **2.训练大模型的计算精度？**

目前普遍做法：混合精度训练 --> 一部分计算用一种精度，另一部分计算用另一种精度。

一般，BF16（bfloat16）用于参数、激活值和梯度，优化器状态使用FP32。若要启用混合精度训练，Pytorch提供了automatic mixed precision（AMP）library。

### **3.einops -- Pytorch的张量操作库**

einops 是Pytorch中的一个非常流行的张量（Tensor）操作库，它的目标是让张量的 reshape、transpose、flatten、repeat 等操作变得更加直观、可读。

PyTorch 原生代码里，经常会看到：
```text
x = x.view(batch_size, -1)
x = x.permute(0, 2, 3, 1)
x = x.reshape(...)
```
维度稍微复杂一点就很难读。而用 einops 可以直接用字符串描述维度变换。

### **4.了解特定某种硬件能提供多少浮点运算次数 & 特定模型需要消耗多少浮点运算次数（也就是模型规模）**

特定模型需要消耗多少浮点运算是指：让某个大语言模型完成一次训练或一次推理，大约需要做多少次浮点数计算。

分为训练FLOPS和推理FLOPS，也就是大模型为了处理输入、生成输出，需要做多少计算。
<img width="2321" height="1279" alt="15f1e0fb7035ad1f651e0e92514e2a0c" src="https://github.com/user-attachments/assets/8cb0bf8a-ae95-473b-971a-ac3c078d48b2" />

### **5.小总结**
1.矩阵乘加运算是计算的主力

2.每秒能跑多少次浮点运算取决于硬件性能和数据类型

3.MFU=实际浮点运算速度/理论速度
<img width="1473" height="270" alt="image" src="https://github.com/user-attachments/assets/cf0952b1-bccb-462d-97c5-16b246c4c05b" />

### **6.计算吞吐量FLOP/s和内存Memory读写吞吐量bytes/s**

<img width="765" height="666" alt="image" src="https://github.com/user-attachments/assets/013955c7-66cd-4c3a-b940-d2a449f4576b" />
<img width="545" height="150" alt="image" src="https://github.com/user-attachments/assets/bac3f3dd-3b77-4ac6-b65b-0f54e3b827d9" />

计算速度和内存搬运速度：

<img width="976" height="329" alt="image" src="https://github.com/user-attachments/assets/8d26aca0-e04c-4204-ba1a-f9f08fe2f9ed" />

### **7.计算强度的概念**

```text
def arithmetic_intensity_relu():
    n=1024*1024
    x=torch.ones(n,dtype=torch.bfloat16, device=cuda_if_available())
    y=torch.relu(x)
```
**其中总共搬运了多少字节？**

bytes=(2*n)+(2*n) # 读取x，写回y，每个浮点数占2字节（bfloat16）

communication_time=bytes/h100_bytes_per_sec

---

**其中一共进行了多少次浮点运算？**

float=n #每次Relu运算都是把浮点数和0比大小，取更大的

computation_time=flops/h100_flop_per_sec

---

总时间假设取二者最大值，Total_time = Max（communication_time，computation_time）

### **8.瓶颈**
算法内存受限：communication_time > computation_time or accelerator_intensity > arithmetic_intensity

算法计算受限：communication_time < computation_time or accelerator_intensity < arithmetic_intensity

### 加速器强度 Accelerateor intensity --硬件属性
含义：这个加速器每传输一个byte，能完成多少工作量。每个加速器的规格表都会写上**加速器强度**

h100_accelerator_intensity=h100_flop_per_sec_ / h100_bytes_per_sec每搬运一个byte，能做多少次有用的浮点运算 #h100是295

其中，上面两个指标是由加速器属性确定的，每种加速器都有固定的这两个指标值。
### 算术强度 Arithmetic intensity --算法属性
含义：这个算法在每搬运一个byte时，究竟做了多少运算

arithmetic_intensity=flops / bytes  # ~0.5 希望能通过优化算法来提高算术强度。

### 目前一般都处于显存受限，Memory Bound
LLM 推理不是算力不够，而是带宽不够。用于计算的 GPU 大部分时间都在等权重从显存Memory里读出来。

---
算法：决定要做多少计算、搬多少数据 （菜谱）

GPU：负责做计算（厨师）

显存：负责存数据、喂数据给 GPU（食材传送带）

算法提出需求，显存提供数据，GPU执行计算。

<img width="754" height="955" alt="image" src="https://github.com/user-attachments/assets/a2996eff-d8ea-4b48-948a-b7675e222b91" />

算法的 Arithmetic Intensity决定它“吃算力”还是“吃带宽”。

GPU/显存的 Accelerator Intensity决定硬件更偏向“强计算”还是“强带宽”。

努力优化算法，提高算术强度。

### 神经网络训练的计算量(FLOPS)估算
前向传播计算量：2ND FLOPS -- 每个参数对每个数据点做一次乘法和一次加法

反向传播计算量：4ND FLOPS -- 是前向传播的两倍

总计算量：6ND FLOPS

其中，N是参数量，是模型中所有可学习权重的总数。对神经网络来说，主要包括每层的权重矩阵w和偏置项b。例如一个FNN，输入维度567，输出维度9234，那么这一层就有567*9234+9234个参数。整个模型所有层的总参数就是N。

D是训练数据点数，是模型在训练过程中实际真正看过的样本总数，通常以Token数量来衡量。若训练数据集有1Ttokens且训练一个epoch，啧D=1T；训练2个epoch，啧D=2T。Llama-2训练用了约2T tokens，GPT-4估计用了10+ tokens。

**6ND的意思是，训练总计算量=6×参数量×训练的tokens数量
