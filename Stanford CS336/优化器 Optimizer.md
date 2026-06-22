## 优化器
优化器用于模型的训练阶段，负责根据 loss 的梯度，决定怎么更新模型参数，让 loss 逐渐变小。

模型的训练流程：
```text
输入数据
↓
模型预测
↓
计算 loss
↓
反向传播算梯度
↓
优化器更新参数
```

其中，“优化器更新参数”就是
```text
loss.backward()      # 计算每个参数该往哪边改
optimizer.step()    # 真正修改参数
optimizer.zero_grad()
```

loss 告诉你“现在错得多严重”，gradient 告诉你“参数该往哪个方向改”，optimizer 决定“每次改多少、怎么改”。因此，优化器是参数更新的策略。
### 优化器分类和比较
<img width="961" height="705" alt="image" src="https://github.com/user-attachments/assets/2a19917d-0df5-4d04-9d25-d9193501c87e" />

### 关于Memory，CPU中叫内存（RAM），GPU中叫显存（HBM）
<img width="2915" height="1279" alt="6d0950f8dd096fa518c8936628c599ef" src="https://github.com/user-attachments/assets/edd9bcbe-0240-48f1-83a5-285ba0009e15" />

可见，占用内存的主要是优化器状态。其中AdaGrad每个参数需要4bytes来存储优化器状态，而Adam每个参数需要8bytes来存储优化器状态。

其中，Batch_size是每次喂给模型的样本数量。训练时不会把全部数据一次性塞进去（显存放不下），而是分批处理。比如训练数据集有100万条文本，batch_size=32就是每次取32条做一轮前向+反向传播，更新一次参数，然后再取下一批32条。

**对显存（HBM）的影响**：batch_size越大，同时在GPU里展开计算的中间值就越多，因此显存不够最直接的方法就是减小batch_size。

**对训练效果的影响**：batch_size越大，每步梯度更新越稳定，但太大也可能导致单步计算量过大，实践中要找平衡。

GPU显存（HBM）在训练中承担两个角色：数据存储和数据传输：

**1.存储：参数、激活值、梯度、优化器状态都必须放在HBM里，GPU核心只能访问自己的显存，不能从CPU内存或硬盘读数据。所以HBM的容量（GB）决定了我们能放下并训练多大的模型。**

**2.传输：计算时，数据要从HBM搬运到GPU内部计算单元（CUDA cores/Tensor cores）。这个传输速度就是HBM的带宽（TB/s）。如果带宽跟不上计算速度，计算单元就会空等数据，造成“带宽瓶颈memory-bound”。**

总体而言，优化器的状态不是计算的主要瓶颈。
