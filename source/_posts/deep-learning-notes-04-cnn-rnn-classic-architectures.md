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

本篇整理序列模型，主要参考Stanford 224n的sequence model一课。整理 RNN、LSTM、GRU等模型


## RNN：处理序列的基本思路

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

## RNN 的梯度问题

RNN 的反向传播要沿时间展开，也叫 backpropagation through time。（有点帅）

如果序列很长，gradient 会经过很多时间步反传。这和深层网络里的梯度消失/爆炸类似。

一个常见问题是 long-term dependency。

比如：

```text
The book that I bought last week and left on the table is interesting.
```

如果要判断主语和谓语关系，模型需要记住很早之前的 "book"。普通 RNN 很容易在长距离依赖上表现差。

## LSTM 和 GRU：用门控机制控制记忆

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

## RNN的弊端

RNN 有一个天然问题：它按时间一步步处理序列，难以并行。

如果序列长度是 1000，RNN 需要从第 1 步一路算到第 1000 步。后一步依赖前一步 hidden state。导致了两个核心问题

1. 梯度消失/梯度爆炸
2. 无法并行计算，每一步都依赖上一个步骤的结果，计算速度很慢

这两个问题都被下一个文章中的attention解决了。


## 参考资料

- [Stanford CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Dive into Deep Learning: Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
