## 1.总览：在original transformer的基础上，各厂商发布了各种自研LLM框架（比如GLM、Qwen3等）

<img width="2413" height="1279" alt="4b454ba55d1d15a170e589fcd0a9ef35" src="https://github.com/user-attachments/assets/10038885-3a23-421b-b80c-dc0667570ec1" />

<img width="2419" height="1279" alt="d213a3b7303b84409a0cf03127b19945" src="https://github.com/user-attachments/assets/583fc556-22f9-4fee-8078-2c881d67d72d" />

现在基于原始transfomer的自研LLM框架层出不穷:

<img width="2271" height="1279" alt="62ffe293267aed12dee90a85d68cf304" src="https://github.com/user-attachments/assets/7990ff67-da97-41e6-a824-ffd8045df060" />

---

## 2.现代Transformer和Attention is all you need里的Transformer的最大不同
先梳理两个概念：Add & Layernorm

---

其中Add就是残差连接，也叫残差流，也叫残差。**残差**就是**把原输入 x 原封不动保留下来，再加上这一层学到的“改动量”。**

或者说，**残差连接就是让每一层只负责修改输入，而不是重新生成输入。**

```text
残差的公式：输出y=输入x+Attention或者MLP算出来的改动
```

---

Layernorm是指层归一化。作用是把每个 token 的hidden state隐藏层的各个特征值调整到一个稳定的数值范围（均值0，均方误差1），防止训练过程中数值越来越大或越来越小，利于训练的稳定。

**残差连接**和**Layernorm**没有任何关系，不用一起出现。传统Transformer（PostNorm）和现代Transformer（PreNorm）的最大区别是：**LayerNorm 放在 Attention/MLP 前面，还是后面。**

<img width="660" height="800" alt="image" src="https://github.com/user-attachments/assets/bfe5a92a-9c8f-4086-8724-3befa3f8ecf0" />
<img width="454" height="730" alt="image" src="https://github.com/user-attachments/assets/a813737d-59eb-4657-8b99-590fe7065abb" />
<img width="723" height="400" alt="image" src="https://github.com/user-attachments/assets/aef835b0-1609-4734-862a-34c3789507cb" />

关于什么叫做**深层训练更稳定，梯度更容易沿着残差路径传回去？**

前向时流的是 hidden state，也就是 x；反向时流的是梯度。Pre-LN 的好处是两者都有更直接的残差路径，尤其让梯度回传更顺。

<img width="859" height="864" alt="image" src="https://github.com/user-attachments/assets/fbe472f9-6bff-4449-be0a-0216de1555fd" />
<img width="499" height="765" alt="image" src="https://github.com/user-attachments/assets/82725094-016a-4109-9740-fb1608d3ee95" />
<img width="920" height="779" alt="image" src="https://github.com/user-attachments/assets/258016f4-a1bf-430a-b2a3-3c6b2a971bc1" />





