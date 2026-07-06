---
title: "深度学习笔记 03：深度网络训练与优化器"
date: 2024-08-17 21:00:00
updated: 2024-08-17 21:00:00
categories:
  - 深度学习笔记
tags:
  - 深度学习
  - 优化器
  - 正则化
  - 笔记
math: true
category_bar: true
---

这一篇整理深度网络训练里最常遇到的一组问题：梯度为什么会不稳定，初始化和归一化为什么重要，以及优化器从 SGD 到 AdamW 到底在改什么。

可以先把训练过程看成一条链：

```text
initialize parameters
-> forward pass
-> compute loss
-> backprop gradients
-> optimizer updates parameters
-> repeat
```

深度学习训练难，主要不是因为这个流程难写，而是因为每一步都可能影响梯度传播和收敛稳定性。

## 1. 梯度消失与梯度爆炸

深层网络里，gradient 要从 loss 一层层传回前面层。

如果每一层的梯度因子都小于 1，多层相乘后会越来越小，这就是 vanishing gradients。

如果每一层的梯度因子都大于 1，多层相乘后会越来越大，这就是 exploding gradients。

一个粗略例子：

$$
0.5^{20} \approx 9.5 \times 10^{-7}
$$

如果反向传播里连续乘很多个 0.5，前面层几乎收不到有效梯度。

反过来：

$$
1.5^{20} \approx 3325
$$

如果连续乘很多个大于 1 的因子，梯度可能爆炸，训练变得不稳定。

这也是为什么 deep network 不是简单“多堆几层”就行。层数变深后，梯度能不能稳定传过去本身就是问题。

## 2. 参数初始化（initialization）

初始化的目标不是找一个好模型，而是给训练一个合适起点。

如果权重太小，信号逐层传递后可能越来越弱；如果权重太大，activation 和 gradient 可能变得不稳定。

常见初始化包括 Xavier initialization 和 He initialization。

Xavier initialization 适合 tanh / sigmoid 这类 activation，核心想法是让每层输入输出的方差保持相对稳定。

He initialization 更适合 ReLU，因为 ReLU 会把一部分负值截断为 0。

直觉上，初始化是在控制每一层信号的尺度。尺度合适，forward 和 backward 都更容易稳定。

## 3. 归一化（normalization）

Normalization 的作用是让中间表示的分布更稳定。

Batch Normalization 通常对 batch 维度做归一化：

$$
\hat{x} = \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

其中，$\mu_B$ 和 $\sigma_B^2$ 是当前 batch 的均值和方差。

BatchNorm 在 CNN 里很常见。它能让训练更稳定，也允许使用更大的 learning rate。

Layer Normalization 则通常对单个样本内部的 hidden dimension 做归一化。Transformer 里更常用 LayerNorm，因为序列任务里 batch 统计不一定稳定，而且不同 token 长度和并行方式会让 BatchNorm 不方便。

可以这样粗略理解：

| 方法 | 主要归一化维度 | 常见场景 |
|---|---|---|
| BatchNorm | batch 维度 | CNN |
| LayerNorm | feature / hidden 维度 | Transformer、RNN |

Normalization 不只是让数值好看，它会影响优化过程。

## 4. 随机失活（dropout）：训练时故意丢掉一部分连接

Dropout 是一种正则化方法。训练时随机把一部分 hidden units 置为 0：

```text
h = dropout(h, p=0.5)
```

它的直觉是：不要让模型过度依赖某几个 neuron，而是让不同子网络都能工作。

测试时通常不再随机丢弃，而是使用完整网络，并对激活值做相应缩放。

Dropout 在早期深度网络里很常见。现在大模型里也会用 dropout，但很多大规模预训练模型会把 dropout 设得比较小，甚至某些设置下不用。

## 5. 学习率（learning rate）和 scheduler

Learning rate 控制每次参数更新的步长。

如果太小，训练慢；如果太大，loss 可能震荡甚至发散。

深度学习里常见 scheduler：

- warmup：训练初期逐渐增大学习率
- step decay：训练到某些 epoch 后降低学习率
- cosine decay：按余弦曲线逐渐降低学习率

Warmup 在 Transformer 训练里尤其常见。训练初期参数还很随机，如果一开始 learning rate 太大，更新可能过猛。Warmup 让模型先稳定进入训练状态。

## 6. SGD：最基础的随机梯度下降

Stochastic Gradient Descent 使用 mini-batch gradient 更新参数：

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

其中，$\eta$ 是 learning rate，$g_t$ 是当前 mini-batch 上的 gradient。

SGD 的优点是简单、泛化表现常常不错。缺点是更新方向受当前 batch 影响，可能噪声较大，收敛也可能比较慢。

比如 loss surface 像一个狭长山谷，SGD 可能在山谷两侧来回震荡，同时沿着谷底方向前进很慢。

## 7. 动量法（momentum）：加入历史方向

Momentum 的想法是：不要只看当前 gradient，也看过去一段时间的更新方向。

常见形式：

$$
v_t = \beta v_{t-1} + g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

其中，$v_t$ 是 velocity，$\beta$ 控制保留多少历史方向。

直觉上，如果多个 step 的 gradient 方向一致，Momentum 会加速；如果某个方向来回震荡，正负更新会互相抵消。

这很像推一个球下山。SGD 每一步都只看当前坡度，Momentum 让球带一点惯性。

## 8. RMSProp：给每个参数自适应学习率

RMSProp 关注的是每个参数最近 gradient 的平方平均。

$$
s_t = \rho s_{t-1} + (1-\rho)g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t}+\epsilon}g_t
$$

如果某个参数的 gradient 一直很大，$s_t$ 会变大，于是它的有效学习率会变小。

如果某个参数的 gradient 一直很小，$s_t$ 小，有效学习率相对更大。

RMSProp 的直觉是：不同参数不一定应该用同一个步长。频繁大幅变化的方向要走得谨慎，变化小的方向可以走得相对积极。

## 9. Adam：动量法（momentum）+ RMSProp

Adam 可以看作把 Momentum 和 RMSProp 结合起来。

它维护一阶矩估计：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

也维护二阶矩估计：

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

然后做 bias correction：

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

最后更新：

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

Adam 的直觉：

- $m_t$ 像 Momentum，记录梯度方向的滑动平均
- $v_t$ 像 RMSProp，记录梯度平方的滑动平均
- 分母让每个参数有自适应步长

Adam 的默认参数经常是：

```text
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
```

Adam 的优点是上手快、收敛稳定、对 learning rate 没有 SGD 那么敏感。这也是它在深度学习里非常常用的原因。

## 10. AdamW：把 weight decay 解耦出来

AdamW 是理解现代训练时很重要的优化器。

先看 Adam 里常见的 L2 regularization。如果把 L2 penalty 加到 loss 里：

$$
L'(\theta)=L(\theta)+\frac{\lambda}{2}\|\theta\|^2
$$

那么梯度会变成：

$$
g'_t = g_t + \lambda \theta_t
$$

在 SGD 里，这和 weight decay 基本等价：

$$
\theta_{t+1} = \theta_t - \eta(g_t + \lambda\theta_t)
$$

可以整理成：

$$
\theta_{t+1} = (1-\eta\lambda)\theta_t - \eta g_t
$$

也就是每步先把权重衰减一点，再按 gradient 更新。

但在 Adam 里，情况不一样。Adam 会用二阶矩 $\sqrt{\hat{v}_t}$ 对梯度做自适应缩放。如果把 $\lambda\theta_t$ 混进 gradient，它也会被 Adam 的自适应项缩放。

这样 weight decay 就不再是“统一把权重变小”，而是和每个参数的梯度历史纠缠在一起。

AdamW 的做法是把 weight decay 从 gradient update 里解耦出来：

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\lambda\theta_t
$$

也可以理解成：

- Adam update: use gradient statistics to update parameters
- Weight decay: separately shrink weights

这就是 AdamW 的核心变化：weight decay 不再被 Adam 的自适应梯度缩放影响。

## 11. Adam 和 AdamW 的区别怎么记

可以用一句话记：

> Adam 里的 L2 regularization 是把 weight decay 混进 gradient；AdamW 是把 weight decay 作为单独的参数衰减步骤。

在 Adam 里：

```text
gradient = gradient + weight_decay * parameter
adaptive_update(gradient)
```

在 AdamW 里：

```text
adaptive_update(gradient)
parameter = parameter - lr * weight_decay * parameter
```

这也是为什么训练 Transformer / LLM 时，AdamW 比 Adam 更常见。它让 weight decay 的作用更接近正则化直觉。

## 12. 优化器对比

| 优化器 | 核心想法 | 优点 | 常见问题 |
|---|---|---|---|
| SGD | 当前 mini-batch gradient | 简单，泛化常不错 | 收敛慢，对 lr 敏感 |
| Momentum | 加入历史方向 | 减少震荡，加速一致方向 | 多一个 momentum 参数 |
| RMSProp | 自适应缩放每个参数 | 处理不同尺度梯度 | 只看二阶信息 |
| Adam | Momentum + RMSProp | 稳定、好用、收敛快 | weight decay 处理不理想 |
| AdamW | Adam + decoupled weight decay | 现代深度学习常用 | 仍然需要调 lr 和 decay |

实际训练里，优化器不是越复杂越好。SGD 在一些视觉任务里仍然很强，AdamW 在 Transformer 训练里非常常见。

## 13. 几个点

深度网络训练的难点不是只会不会写 `loss.backward()`，而是梯度是否稳定、参数尺度是否合理、优化器是否适合任务。

Initialization 和 normalization 都在服务同一个目标：让信号和梯度在网络中稳定传播。

SGD 只看当前 gradient，Momentum 加入历史方向，RMSProp 加入梯度平方的滑动平均，Adam 同时使用一阶和二阶矩估计。

AdamW 的关键不是“比 Adam 多一个 W”，而是 decoupled weight decay。这个区别在现代大模型训练里很重要。

Learning rate 仍然是最重要的超参数之一。用了 AdamW 也不代表不需要认真调学习率。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Dive into Deep Learning: Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html)
- [CS231n: Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/)
