---
title: "深度学习笔记 05：注意力机制与 Transformer"
date: 2024-08-31 22:00:00
updated: 2024-08-31 22:00:00
categories:
  - 深度学习笔记
tags:
  - 深度学习
  - 注意力机制
  - Transformer
  - 自注意力
  - 笔记
math: true
category_bar: true
---

这一篇整理注意力机制和 Transformer。它是 Deep Learning 章节和后面现代大模型章节之间的桥。

这篇不展开 GPT、BERT、MoE、RLHF 这些内容，只看 Transformer 为什么出现，以及 self-attention 到底在做什么。

可以先抓住一句话：

> Attention 让模型在处理某个位置时，直接从其他位置取信息，而不是把所有历史信息都压进一个固定 hidden state。

## 1. RNN 的瓶颈

RNN 处理序列时，每个时间步依赖上一个 hidden state：

$$
h_t = f(x_t, h_{t-1})
$$

这带来两个问题。

第一，长距离信息要经过很多步传递，容易衰减。

第二，计算难以并行，因为第 $t$ 步必须等第 $t-1$ 步算完。

比如翻译句子时，目标词可能需要关注源句中很远的词。RNN 要把这些信息逐步压缩进 hidden state，压力很大。

Attention 的想法是：不要只依赖最后一个 hidden state，而是让模型需要什么就去看什么。

## 2. 注意力机制（attention）的直觉

Attention 可以理解成一个信息检索过程。

当模型处理当前位置时，它会问：

```text
当前 token 需要从哪些其他 token 获取信息？
```

比如句子：

```text
The animal didn't cross the street because it was too tired.
```

这里 "it" 指的是 "animal"，而不是 "street"。模型处理 "it" 时，需要关注前面的 "animal"。

Attention 就是在学习这种关联强度。

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

当前位置的 query 会和所有位置的 key 做相似度计算，得到 attention score。

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

如果向量维度很高，dot product 的数值可能变大，softmax 会变得过于尖锐，梯度不稳定。缩放项可以缓解这个问题。

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

"bank" 的含义要靠上下文判断。Self-attention 让 "bank" 可以关注 "loan" 或 "river"，从而形成不同语义表示。

这也是 Transformer 适合语言建模的原因之一：每个 token 的表示不是孤立的，而是上下文化的。

## 6. 多头注意力（multi-head attention）

Multi-head attention 不是只做一次 attention，而是并行做多组 attention：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

然后把多个 head 拼接起来：

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
$$

直觉上，不同 head 可以关注不同关系。

一个 head 可能关注主谓关系，一个 head 可能关注指代关系，一个 head 可能关注局部相邻 token。当然实际 head 不一定这么干净可解释，但多头机制给了模型并行建模不同关系的能力。

## 7. 位置编码（positional encoding）：Transformer 怎么知道顺序

Self-attention 本身不包含顺序信息。

如果只看 attention 公式，输入 token 换个顺序，模型并不会天然知道谁在前谁在后。

所以 Transformer 需要加入 positional encoding。

原始 Transformer 使用正弦余弦位置编码：

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

现代模型里也常见 learned positional embedding、RoPE 等位置编码方式。

位置编码的核心目的很简单：让模型知道 token 的顺序和相对位置。

## 8. Transformer 模块（Transformer block）的基本结构

一个 Transformer block 通常包含：

```text
self-attention
-> residual connection
-> layer normalization
-> feed-forward network
-> residual connection
-> layer normalization
```

Feed-forward network 通常是对每个 token 独立应用的 MLP：

$$
FFN(x)=W_2\sigma(W_1x+b_1)+b_2
$$

Self-attention 负责 token 之间的信息交互，FFN 负责对每个 token 的表示做非线性变换。

Residual connection 让深层网络更容易训练，LayerNorm 让表示尺度更稳定。

## 9. 编码器（encoder）和解码器（decoder）

原始 Transformer 有 encoder 和 decoder。

Encoder 读入输入序列，生成上下文表示。机器翻译里，encoder 读源语言句子。

Decoder 生成输出序列。它有 masked self-attention，保证生成第 $t$ 个 token 时不能偷看未来 token。

简单对比：

| 结构 | 作用 | 典型模型 |
|---|---|---|
| Encoder-only | 理解输入 | BERT |
| Decoder-only | 自回归生成 | GPT |
| Encoder-decoder | 输入到输出转换 | T5、原始翻译 Transformer |

这部分后面讲现代大模型时还会展开。这里只要先记住：GPT 类模型主要是 decoder-only Transformer。

## 10. 为什么 Transformer 更适合大规模训练

相比 RNN，Transformer 的优势很明显。

RNN 必须按时间步顺序计算：

```text
h1 -> h2 -> h3 -> ... -> hn
```

Transformer 的 self-attention 可以并行处理整段序列：

```text
all tokens -> attention matrix -> contextual representations
```

这让它更适合 GPU 并行计算。

此外，self-attention 让任意两个 token 之间的信息路径更短。RNN 里第 1 个 token 到第 100 个 token 要经过很多步，Transformer 里可以直接通过 attention 建立联系。

代价是 self-attention 的计算和显存通常随序列长度平方增长：

$$
O(n^2)
$$

所以长上下文 Transformer 需要各种效率优化。

## 11. 几个点

Attention 的本质是根据相关性对 value 做加权汇聚。

Query、key、value 不需要背成术语。Query 是当前位置的问题，key 是其他位置的索引，value 是真正要取的信息。

Self-attention 让每个 token 的表示变成上下文化表示。

Multi-head attention 给模型多个并行视角，不同 head 可以关注不同关系。

Transformer 能替代很多 RNN 场景，一个关键原因是并行训练效率高，另一个关键原因是长距离依赖路径短。

Transformer 不是大模型本身，但它是现代 LLM 的基础结构。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Dive into Deep Learning: Attention Mechanisms](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
