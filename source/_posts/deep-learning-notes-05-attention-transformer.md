---
title: "深度学习笔记 05：注意力机制"
date: 2024-08-31 22:00:00
updated: 2024-08-31 22:00:00
categories:
  - 深度学习笔记
tags:
  - 深度学习
  - 注意力机制
  - 自注意力
  - 笔记
math: true
category_bar: true
---

这一篇只整理 attention 本身。

Transformer block、Pre-Norm、FFN、RoPE、GQA 这些更偏模型架构的内容，放到现代大模型章节里讲。这里先把最核心的问题说清楚：

> 模型处理当前位置时，怎么从其他位置拿到有用信息。

## 1. RNN 的瓶颈

RNN 处理序列时，每个时间步依赖上一个 hidden state：

$$
h_t = f(x_t, h_{t-1})
$$

这带来两个问题。

第一，长距离信息要经过很多步传递，容易衰减。

第二，计算难以并行，因为第 $t$ 步必须等第 $t-1$ 步算完。

比如翻译一句话时，目标词可能需要关注源句中很远的词。RNN 要把这些信息逐步压缩进 hidden state，压力很大。

Attention 的想法是：不要只依赖最后一个 hidden state，而是让模型需要什么就直接去看什么。

## 2. 注意力机制（attention）的直觉

Attention 可以理解成一个信息检索过程。

当模型处理当前位置时，它会问：

```text
当前位置需要从哪些位置拿信息？
```

比如：

```text
The animal didn't cross the street because it was too tired.
```

这里 `it` 指的是 `animal`，不是 `street`。模型处理 `it` 时，需要更关注前面的 `animal`。

Attention 学的就是这种关联强度。

## 3. 查询、键和值（query、key、value）

Attention 常用 query、key、value 来描述。

可以用检索类比：

- Query：我现在想找什么
- Key：每个位置提供的索引
- Value：每个位置真正携带的信息

对每个 token，模型都会生成三个向量：

$$
q_i = x_iW_Q
$$

$$
k_i = x_iW_K
$$

$$
v_i = x_iW_V
$$

当前位置的 query 会和所有位置的 key 做相似度计算，得到 attention score。score 越高，说明当前位置越应该从那个位置取信息。

## 4. 缩放点积注意力（scaled dot-product attention）

Transformer 里常用 scaled dot-product attention：

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

分成三步看：

1. $QK^\top$ 计算 query 和 key 的相似度
2. 除以 $\sqrt{d_k}$ 控制数值尺度
3. softmax 得到权重，再对 $V$ 做加权平均

为什么要除以 $\sqrt{d_k}$？

如果向量维度很高，dot product 的数值可能变大，softmax 会变得过于尖锐。最后结果就是某一个 token 权重接近 1，其他 token 权重接近 0，梯度也容易不稳定。

缩放项不是为了改变 attention 的本质，而是让数值更稳。

## 5. 自注意力（self-attention）：序列内部互相看

Self-attention 指 query、key、value 都来自同一个序列。

也就是说，每个 token 都可以关注同一句话里的其他 token。

比如：

```text
The bank approved the loan.
```

和：

```text
The bank is near the river.
```

`bank` 的含义要靠上下文判断。Self-attention 让 `bank` 可以关注 `loan` 或 `river`，从而形成不同语义表示。

这也是 attention 很适合语言任务的原因：token 的表示不是孤立的，而是上下文化的。

## 6. Attention matrix 在看什么

如果序列长度是 $n$，那么 $QK^\top$ 会得到一个 $n \times n$ 的矩阵。

第 $i$ 行可以理解成：

```text
第 i 个 token 在看哪些 token
```

第 $j$ 列可以理解成：

```text
哪些 token 在看第 j 个 token
```

这也是为什么 attention 常被拿来可视化。虽然不能把 attention weight 直接等同于“模型解释”，但它至少能显示信息流的大致方向。

一个简单例子：

```text
I gave the book to Alice because she likes reading.
```

处理 `she` 时，attention matrix 里对应那一行可能会给 `Alice` 更高权重。这样模型才有机会把指代关系接上。

## 7. Mask：哪些位置不能看

Attention 里经常会加 mask。

Mask 的作用是：在 softmax 前把某些位置的 score 变成很小的数，让它们的权重接近 0。

最常见的是 causal mask。

语言模型做 next token prediction 时，第 $t$ 个位置不能看未来 token，否则训练就作弊了。

所以 causal attention 会让第 $t$ 个位置只能看 $1$ 到 $t$：

```text
token 1: 看 1
token 2: 看 1, 2
token 3: 看 1, 2, 3
token 4: 看 1, 2, 3, 4
```

如果没有 causal mask，模型预测下一个词时已经看到了答案，训练 loss 会很好看，但推理时完全没法用。

Padding mask 也是同类东西。batch 里不同句子长度不一样，短句后面补的 padding token 不应该参与 attention。

## 8. 多头注意力（multi-head attention）

Multi-head attention 不是只做一次 attention，而是并行做多组 attention：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

然后把多个 head 拼接起来：

$$
\text{MultiHead}(Q,K,V)
=
\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
$$

直觉上，不同 head 可以关注不同关系。

一个 head 可能更关注局部搭配，一个 head 可能更关注指代关系，一个 head 可能更关注句法结构。当然实际 head 不一定这么干净可解释，但多头机制给了模型并行建模不同关系的空间。

## 9. Attention 和 Transformer 的关系

Attention 不是 Transformer 的全部。

Transformer 还包括 residual connection、normalization、feed-forward network、position encoding 等结构。

但 attention 是 Transformer 最核心的信息交互模块。它决定了 token 之间怎么通信；FFN 决定每个 token 的表示怎么被加工；residual 和 normalization 决定深层网络能不能稳定训练。

所以这篇只解决一个基础问题：attention 怎么让 token 互相看。下一步再看 Transformer 是怎么把 attention 组织成大模型架构的。

## 几个点

- Attention 的核心是根据 query-key 相似度，从 value 里加权取信息。
- Self-attention 让同一个序列内部的 token 互相建模。
- Multi-head attention 给模型多个并行视角，不同 head 可以学不同关系。
- Mask 很关键。尤其是 causal mask，它决定语言模型只能从左到右生成。
- Attention weight 可以帮助理解信息流，但不要简单当成完整解释。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Dive into Deep Learning: Attention Mechanisms](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
