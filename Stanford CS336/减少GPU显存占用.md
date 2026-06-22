### 减少GPU显存占用
GPU显存不仅影响存储大模型的能力，有时还影响训练速度，因此要尽可能减少显存占用。

**几个方法：**

**1.优化器 Optimizer**
**2.梯度累积 Gradient_accumulation：用小显存模拟大的Batch_size，用时间换显存空间**

<img width="1305" height="468" alt="41dc3cba8f7b75c7166c896f771db3e3" src="https://github.com/user-attachments/assets/e74591a4-5774-4567-9e5c-c06c269e0210" />

<img width="1473" height="671" alt="1a9c4eea0fa307921bff2fd21e3a198f" src="https://github.com/user-attachments/assets/56f24352-6cf8-4915-b76e-412791af56f3" />

**3.激活检查点：另一种用时间换显存空间的技巧**

<img width="2376" height="1046" alt="27526ea163eb5c9f3897fccdf788ce83" src="https://github.com/user-attachments/assets/e586e27c-aa4c-4dee-ad4f-4fe132ed5518" />
