---
title: "深度学习笔记 04：CNN、RNN 与经典结构"
date: 2024-08-24 21:30:00
updated: 2024-08-24 21:30:00
categories:
  - 深度学习笔记
tags:
  - 深度学习
  - CNN
  - RNN
  - LSTM
  - ResNet
  - 笔记
math: true
category_bar: true
---

这一篇整理 CNN、RNN、LSTM、GRU 这些经典结构。这里不写成模型百科，只抓它们各自利用了什么数据结构假设。

可以先从一个问题出发：

> 如果 MLP 已经可以拟合复杂函数，为什么还需要 CNN、RNN 这些结构？

答案是：不同数据有不同结构。图像有局部空间结构，序列有时间顺序结构。专门的网络结构把这些先验写进模型里，让模型更高效地学习。

## 1. 归纳偏置（inductive bias）

Inductive bias 可以理解成模型对数据结构的默认假设。

MLP 的假设很弱。它把输入展开成向量后，每个输入维度和每个 hidden unit 都连接起来。

如果输入是一张图片，MLP 并不知道左上角像素和它旁边像素关系更近，也不知道同一个边缘模式可以出现在图片不同位置。

CNN 就把这些假设写进模型：

- 局部连接：相邻像素更相关
- 权重共享：同一个 pattern 可以出现在不同位置
- 平移等变：输入平移时，feature map 也对应平移

RNN 则把序列假设写进模型：

- 当前状态依赖过去状态
- 同一组参数可以反复处理不同时间步
- 顺序本身有意义

这些结构不只是为了减少参数，也是在告诉模型应该怎样看数据。

## 2. 卷积神经网络（CNN）为什么适合图像

图像有很强的局部结构。一个像素通常和周围像素关系更大，而不是和很远的像素关系同等重要。

CNN 使用 convolution kernel 在图像上滑动：

```text
small kernel
-> slide over image
-> produce feature map
```

比如一个 $3 \times 3$ kernel 每次只看局部区域。它可以检测边缘、角点、纹理等局部 pattern。

如果某个 kernel 学会检测垂直边缘，那么它可以在整张图上重复使用。同一个边缘检测器不需要为每个位置单独学一套参数。

这就是 weight sharing。

## 3. 卷积（convolution）的几个概念

常见概念：

| 概念 | 含义 |
|---|---|
| Kernel / filter | 用来扫描局部区域的小矩阵 |
| Feature map | kernel 扫描后得到的响应图 |
| Stride | 每次滑动的步长 |
| Padding | 在边缘补 0，控制输出尺寸 |
| Channel | 输入或输出的通道数 |

如果 stride 大，输出 feature map 会变小；如果 padding 合适，可以让输出保持原尺寸。

一个直觉例子：检测猫脸时，前面层可能检测边缘和纹理，中间层检测眼睛、耳朵、鼻子，后面层组合成更完整的猫脸结构。

CNN 的层级结构和图像视觉层级比较匹配。

## 4. 池化（pooling）：降低空间分辨率

Pooling 常用于降低 feature map 尺寸。

Max pooling 取局部窗口里的最大值：

```text
2x2 window -> max value
```

它的作用包括：

- 降低计算量
- 增大感受野
- 提供一定平移不变性

比如一个边缘稍微移动几像素，max pooling 后的响应可能仍然相似。

不过现代 CNN 里不一定大量依赖 pooling，有些结构会用 stride convolution 来完成下采样。

## 5. ResNet：为什么残差连接（residual connection）重要

网络变深后，训练会变难。一个问题是：更深的网络理论上表达能力更强，但实际训练时可能反而效果变差。

ResNet 的核心是 residual connection：

$$
y = F(x) + x
$$

模型不直接学习完整映射 $H(x)$，而是学习 residual：

$$
F(x) = H(x) - x
$$

直觉上，如果某几层暂时不需要做复杂变换，模型可以让 $F(x)$ 接近 0，这样输出接近 $x$。这让深层网络更容易优化。

Residual connection 也让梯度可以更直接地传回前面层。这个思想后来不只用于 CNN，也广泛出现在 Transformer 里。

## 6. RNN：处理序列的基本思路

RNN 用 hidden state 记录过去信息。

最基本形式：

$$
h_t = \phi(W_xx_t + W_hh_{t-1} + b)
$$

$$
y_t = W_yh_t
$$

其中，$x_t$ 是当前时间步输入，$h_{t-1}$ 是上一个时间步的 hidden state。

比如句子：

```text
I really like this movie
```

RNN 会按顺序读 token，每一步更新 hidden state。理论上，最后的 hidden state 可以包含前面读到的信息。

RNN 的优点是结构自然适合序列；缺点是长序列训练困难，难以捕捉很长距离的依赖。

## 7. RNN 的梯度问题

RNN 的反向传播要沿时间展开，也叫 backpropagation through time。

如果序列很长，gradient 会经过很多时间步反传。这和深层网络里的梯度消失/爆炸类似。

一个常见问题是 long-term dependency。

比如：

```text
The book that I bought last week and left on the table is interesting.
```

如果要判断主语和谓语关系，模型需要记住很早之前的 "book"。普通 RNN 很容易在长距离依赖上表现差。

## 8. LSTM 和 GRU：用门控机制控制记忆

LSTM 的核心是 cell state 和 gate。

它通过门控机制控制：

- 忘掉哪些旧信息
- 写入哪些新信息
- 输出哪些状态

可以把 LSTM 想成比普通 RNN 多了一条更稳定的记忆通道。

GRU 是更简化的门控 RNN，参数更少，训练更轻一些。

常见对比：

| 模型 | 核心 | 特点 |
|---|---|---|
| RNN | hidden state | 简单，但长依赖困难 |
| LSTM | cell state + gates | 更擅长长依赖，参数较多 |
| GRU | simplified gates | 比 LSTM 简洁 |

LSTM / GRU 不是彻底解决长距离依赖，只是比普通 RNN 更稳定。

## 9. 为什么注意力机制（attention）后来替代很多 RNN

RNN 有一个天然问题：它按时间一步步处理序列，难以并行。

如果序列长度是 1000，RNN 需要从第 1 步一路算到第 1000 步。后一步依赖前一步 hidden state。

Attention 的思路不同。它允许当前位置直接看序列中其他位置，不一定要把信息压进一个 hidden state 里一步步传递。

这带来两个优势：

- 更容易建模长距离依赖
- 更容易并行训练

所以在 NLP 和大模型里，Transformer 逐渐成为主流。

## 10. 几个点

CNN 的关键不是“卷积公式”，而是局部连接和权重共享。它利用了图像的空间结构。

ResNet 的 residual connection 让深层网络更容易优化，这个思想后来影响很大。

RNN 适合序列，但长距离依赖和并行效率是它的核心瓶颈。

LSTM / GRU 通过门控机制缓解 RNN 的梯度和记忆问题，但没有完全摆脱序列计算瓶颈。

CNN、RNN、Transformer 的区别，本质上是它们对数据结构的不同假设。

## 参考资料

- [Stanford CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Dive into Deep Learning: Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
