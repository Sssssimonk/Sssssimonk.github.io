---
title: "深度学习笔记 01：从特征工程到表示学习"
date: 2024-08-03 20:00:00
updated: 2024-08-03 20:00:00
categories:
  - 深度学习笔记
tags:
  - 深度学习
  - 神经网络
  - 笔记
math: true
category_bar: true
---

神经网络和机器学习的本质还是一样的，都是为了找到一组参数w，使得 y hat = w1 * x + w2 * x + b 的预测值更接近真实的标签y。


<!-- more -->

$$
\hat{y} = f_\theta(x)
$$

其中，$x$ 是输入，$\hat{y}$ 是预测，$\theta$ 是网络参数。

训练的目标仍然是找一组参数，让模型预测更接近真实标签：

$$
\theta^* = \arg\min_\theta L(f_\theta(x), y)
$$

所以 deep learning 的基本框架仍然是：

```text
input
-> model prediction
-> loss
-> gradient
-> parameter update
```

区别在于 $f_\theta$ 变得很灵活，可以由很多层可学习变换组成。

## 表示（representation）

传统机器学习里，特征工程经常决定上限。

比如判断一封邮件是不是垃圾邮件，可能会手动构造特征：

| 特征 | 含义 |
|---|---|
| `contains_free` | 是否出现 free |
| `num_links` | 邮件里的链接数量 |
| `sender_reputation` | 发件人信誉 |
| `has_attachment` | 是否有附件 |

模型看到的不是原始邮件，而是这些被设计好的 feature。

深度学习的思路是：把原始数据输入网络，让网络逐层学出中间表示。

以图像为例，早期层可能学边缘和纹理，中间层学局部形状，后面层学更接近语义的结构。这个过程不是人为写规则，而是通过 loss 和 gradient 学出来的。

这就是 representation learning。

Representation 可以理解成模型内部对输入的重新编码。

假设输入是一张图片，原始像素只是一个大矩阵。对模型来说，像素本身不直接等于“猫”“狗”“车”。网络需要把像素变成更有用的中间特征：

```text
pixels
-> edges / textures
-> parts
-> object-level features
-> prediction
```

在 NLP 里也类似。一个词或者一句话会被表示成 embedding。Embedding 不是原始文本，而是模型学出来的一组向量，用来承载语义和上下文信息。

比如：

```text
"king" -> [0.21, -0.18, 0.73, ...]
"queen" -> [0.19, -0.11, 0.69, ...]
```

这些数字本身没有人类可读含义，但它们让模型可以计算相似度、组合上下文、做预测。

单层线性模型表达能力有限：

$$
\hat{y} = Wx + b
$$

如果数据关系本身很复杂，只靠一个线性变换很难表达。

多层网络的想法是把简单变换组合起来：

$$
h_1 = \sigma(W_1x + b_1)
$$

$$
h_2 = \sigma(W_2h_1 + b_2)
$$

$$
\hat{y} = W_3h_2 + b_3
$$

每一层都在前一层表示的基础上再做一次变换。这样模型可以逐步把原始输入变成更适合任务的表示。

## 非线性为什么重要

如果没有 activation function，多层线性网络仍然等价于一个线性模型。

比如：

$$
h = W_1x
$$

$$
\hat{y} = W_2h = W_2W_1x
$$

这里 $W_2W_1$ 仍然只是一个新的矩阵。堆很多层也没有本质变化。

所以神经网络需要在层与层之间加入非线性函数：

$$
h = \sigma(Wx+b)
$$

常见 activation 有 ReLU、sigmoid、tanh。现代深度网络里 ReLU 及其变体很常见，因为它简单、梯度更稳定、训练效率高。

## 深度学习的成功条件

Deep learning 不是单靠一个想法突然成功。它通常需要几个条件同时出现：

| 条件 | 作用 |
|---|---|
| 大数据 | 让大模型有足够样本学习表示 |
| GPU / TPU | 让矩阵计算足够快 |
| 更好的优化方法 | 让深层网络能稳定训练 |
| 更好的结构设计 | CNN、ResNet、Transformer 等 |
| 软件生态 | PyTorch、TensorFlow、CUDA 等 |

如果数据很少、任务很简单，传统机器学习模型可能仍然更合适。Deep learning 的优势通常在原始输入复杂、人工特征难设计、数据规模足够大的场景里更明显。


## 参考资料

- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Dive into Deep Learning](https://d2l.ai/)
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
