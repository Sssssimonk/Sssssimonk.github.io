---
title: "深度学习笔记 02：MLP、前向传播与反向传播"
date: 2024-08-10 20:30:00
updated: 2024-08-10 20:30:00
categories:
  - 深度学习笔记
tags:
  - 深度学习
  - MLP
  - 反向传播
  - 计算图
  - 笔记
math: true
category_bar: true
---

这一篇整理最基础的神经网络：多层感知机（MLP），以及神经网络到底如何通过前向传播和反向传播完成训练。

可以先抓住一句话：

> Forward pass 负责算预测和 loss，backpropagation 负责把 loss 对每个参数的梯度算出来，optimizer 再根据梯度更新参数。

这条链路是 deep learning 的训练核心。

## 神经元（neuron）在做什么

一个最简单的 neuron 可以写成：

$$
z = w^\top x + b
$$

$$
h = \sigma(z)
$$

其中，$x$ 是输入，$w$ 是权重，$b$ 是 bias，$\sigma$ 是 activation function。

如果没有 activation，这就是线性模型的一部分；加上 activation 后，它可以引入非线性。

一个具体例子：

```text
x = [study_hours, sleep_hours]
w = [0.8, 0.3]
b = -2
```

这个 neuron 可能在粗略判断“考试通过倾向”。学习时间和睡眠时间越高，$z$ 越大，经过 sigmoid 后输出越接近 1。

当然真实神经网络里的 neuron 不会这么直接可解释，但这个例子能帮助理解它的基本计算。

## 层（layer）就是一组神经元

把多个 neuron 放在一起，就是一层。

线性层通常写成：

$$
z = Wx + b
$$

$$
h = \sigma(z)
$$

这里 $W$ 是矩阵，$b$ 是向量。每一行权重可以看成一个 neuron。

如果输入是 batch，写法会变成：

$$
H = \sigma(XW^\top + b)
$$

实际框架里不会手动写每个 neuron，而是通过矩阵乘法一次算完整层。

## 多层感知机（MLP）

MLP 由多个 fully connected layer 组成。

一个两层 MLP 可以写成：

$$
h = \sigma(W_1x + b_1)
$$

$$
\hat{y} = W_2h + b_2
$$

其中，$h$ 是 hidden representation。

分类任务里，最后通常接 softmax：

$$
p(y=k|x) = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

回归任务里，最后可以直接输出连续值。

MLP 的缺点是没有针对图像、序列、文本结构做特殊假设。它很通用，但也很“笨”：如果输入是图片，MLP 不知道相邻像素之间有局部关系；如果输入是序列，MLP 不知道 token 顺序有意义。

所以后面才会出现 CNN、RNN、Transformer 这些结构。

## 前向传播（forward pass）

Forward pass 就是从输入一路算到输出和 loss。

以二分类 MLP 为例：

```text
x
-> linear layer
-> ReLU
-> linear layer
-> sigmoid
-> binary cross entropy loss
```

写成公式：

$$
h = \text{ReLU}(W_1x+b_1)
$$

$$
\hat{y} = \sigma(W_2h+b_2)
$$

$$
L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

Forward pass 只解决一个问题：当前参数下，模型预测得怎么样。

但训练还需要知道：如果 loss 大，参数应该怎么改。

## 反向传播（backpropagation）

Backpropagation 的核心是 chain rule。

如果：

$$
L = L(\hat{y}), \quad \hat{y}=f(h), \quad h=g(\theta)
$$

那么：

$$
\frac{\partial L}{\partial \theta}
=
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial h}
\frac{\partial h}{\partial \theta}
$$

这就是反向传播的直觉：loss 对前面参数的影响，要沿着计算路径一层层传回去。

它不是一种神秘算法，本质上就是高效地在计算图上应用链式法则。

## 一个小计算图例子

看一个最简单的例子：

$$
z = wx + b
$$

$$
\hat{y} = z
$$

$$
L = (\hat{y} - y)^2
$$

现在要求 $L$ 对 $w$ 的梯度。

根据 chain rule：

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial z}
\frac{\partial z}{\partial w}
$$

分别计算：

$$
\frac{\partial L}{\partial \hat{y}} = 2(\hat{y}-y)
$$

$$
\frac{\partial \hat{y}}{\partial z} = 1
$$

$$
\frac{\partial z}{\partial w} = x
$$

所以：

$$
\frac{\partial L}{\partial w} = 2(\hat{y}-y)x
$$

如果预测太大，$\hat{y}-y$ 为正，梯度方向会让 $w$ 下降；如果预测太小，梯度方向会让 $w$ 上升。这个例子很简单，但多层网络也是同一个逻辑，只是计算图更大。

## 自动求导（automatic differentiation）

实际训练时通常不会手动推导每个参数的梯度。PyTorch、TensorFlow 这类框架会构建计算图，然后自动反向传播。

PyTorch 里的训练大概是：

```python
optimizer.zero_grad()
pred = model(x)
loss = criterion(pred, y)
loss.backward()
optimizer.step()
```

这里：

- `pred = model(x)` 是 forward pass
- `loss.backward()` 是 backpropagation
- `optimizer.step()` 是参数更新

`zero_grad()` 也很重要，因为 PyTorch 默认会累积 gradient。如果不清零，下一轮的 gradient 会叠加到上一轮上。

## 激活函数（activation function）的作用

Activation function 提供非线性。

常见 activation：

| 函数 | 形式 | 特点 |
|---|---|---|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | 输出在 0 到 1，容易梯度饱和 |
| Tanh | $\tanh(x)$ | 输出在 -1 到 1，也可能饱和 |
| ReLU | $\max(0,x)$ | 简单高效，深度网络常用 |
| GELU | 平滑版本 | Transformer 里常见 |

Sigmoid 和 tanh 在输入绝对值很大时，曲线会变平，梯度接近 0。深层网络里这会导致前面层学得很慢。

ReLU 的优点是正半轴梯度稳定，计算简单。但 ReLU 也可能出现 dead neuron：如果某个 neuron 长期输出 0，它对应参数可能很难更新。



## 参考资料

- [Deep Learning Book: Chapter 6](https://www.deeplearningbook.org/)
- [Dive into Deep Learning: Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/index.html)
- [CS231n: Backpropagation](https://cs231n.github.io/optimization-2/)
- [PyTorch Autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
