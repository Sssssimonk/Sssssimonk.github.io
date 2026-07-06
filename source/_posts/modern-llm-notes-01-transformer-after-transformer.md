---
title: "现代大模型笔记 01：Transformer 之后到底变了什么"
date: 2025-09-06 20:00:00
updated: 2025-09-06 20:00:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - 大模型
  - Transformer
  - 注意力机制
  - RoPE
  - KV cache
  - 笔记
math: true
category_bar: true
---


## 1. 从 encoder-decoder 到 decoder-only

原始 Transformer是seq2seq架构，Encoder 负责读完整个输入句子，decoder 负责一步一步生成目标句子。

但现在很多通用语言模型，比如 GPT、LLaMA、Qwen、DeepSeek，主要采用 decoder-only 结构。它没有单独的 encoder，而是把所有任务都改写成 next token prediction：

$$
p(x_t \mid x_1, x_2, ..., x_{t-1})
$$

也就是说，模型只需要学会一件事：给定前面的 token，预测下一个 token。

这个形式非常统一。翻译、问答、总结、代码生成、数学推理，都可以变成“前面是一段 prompt，后面继续生成答案”。

这里有一个重要区别：

- BERT 这类 encoder-only 模型更像是在理解一整段文本
- GPT 这类 decoder-only 模型更像是在从左到右生成文本

LLM 选择 decoder-only，不是因为 encoder 不好，而是因为自回归生成和大规模预训练的目标天然对齐，并且比起seq2seq更容易训练。

## 2. Causal attention：只能看过去

Decoder-only 模型使用 causal self-attention。

普通 self-attention 中，每个 token 可以看见所有 token。但语言生成时，当前位置不能偷看未来 token，否则 next token prediction 就变成了作弊。

所以 causal attention 会加一个 mask，让第 $t$ 个位置只能关注 $1$ 到 $t$ 的 token。

如果一句话是：

> The cat sits on the

模型预测下一个词时，可以看见 "The cat sits on the"，但不能提前看见答案 "mat"。

这个 mask 看起来只是训练细节，但它决定了模型的能力形态：LLM 是一步一步生成的，不是一次性把完整答案算出来。

这也解释了为什么长回答里有时会出现前后不一致。模型每一步都基于已经生成的内容继续走，如果前面方向偏了，后面会被带偏。

## 3. Normalization

**Pre-Norm：深层网络更容易训练**

原始 Transformer 使用 Post-Norm，大概形式是：

$$
x_{l+1} = \mathrm{LayerNorm}(x_l + F(x_l))
$$

现代 LLM 更常见的是 Pre-Norm：

$$
x_{l+1} = x_l + F(\mathrm{Norm}(x_l))
$$

差别是 normalization 放在 residual branch 前面还是后面。

Pre-Norm 的好处是深层模型训练更稳定。直觉上，residual path 更像一条“干净的高速通道”，梯度可以更直接地往前传，不会每层都被 normalization 包住。

模型层数变深后，pre-norm带来的训练稳定性让它更受青睐。

**RMSNorm：少算一点，但保留关键尺度控制**

之前使用的LayerNorm 会减均值、除标准差：

$$
\mathrm{LayerNorm}(x) = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} \odot g + b
$$

RMSNorm 去掉了减均值，只保留 root mean square 的尺度归一化：

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}} \odot g
$$

它关心的是向量整体尺度，而不是每一维相对均值的位置。

这有两个好处。

第一，计算更简单。对超大模型来说，每个 block 里省一点，整体就能省很多。

第二，它保留了最关键的稳定性作用：控制 activation 的尺度。


## 5. Activation function：从 ReLU 走向 SwiGLU

Transformer block 里除了 attention，还有一个 feed-forward network。

原始 Transformer 用的是：

$$
\mathrm{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2
$$

现代 LLM 常用 SwiGLU 这类 gated FFN：

$$
\mathrm{SwiGLU}(x) = \mathrm{Swish}(xW_1) \odot (xW_2)
$$

再接一个输出投影。

Gating 的直觉是：模型不只是对特征做非线性变换，还学会“哪些通道该打开，哪些通道该压下去”。

如果把 FFN 看成每个 token 独立做一次特征加工，那么 SwiGLU 相当于给这次加工加了一个动态开关。

不过现在一般都转变成MoE了。

## 6. RoPE：把位置信息放进旋转里

Self-attention 本身不包含顺序信息。对 attention 来说，如果不加位置编码，"A B C" 和 "C B A" 只是 token 集合不同排列，很难知道谁在前谁在后。

原始 Transformer 用 absolute positional encoding，把位置向量加到 token embedding 上。

现代 LLM 常用 RoPE，也就是 rotary position embedding。它不是简单把位置向量加进去，而是在 query 和 key 上做旋转，使 attention score 自然带上相对位置信息。

可以粗略理解成：

$$
q_m^\top k_n \rightarrow \text{a function of token content and relative distance } (m-n)
$$

RoPE 的关键好处是相对位置关系更自然，也更适合做一定程度的位置外推。

RoPE scaling 可以帮助模型扩展 context length，但不等于模型自动学会利用超长上下文。位置能编码是一回事，模型能不能从 100K token 里稳定找出关键证据，是另一回事。

## 7. Multi-head attention 到 MQA / GQA

标准 multi-head attention 里，每个 head 都有自己的 $Q, K, V$：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

多头的作用是让模型从不同子空间看 token 之间的关系。

但推理时有一个很大的成本：KV cache。

生成第 $t$ 个 token 时，前面所有 token 的 key 和 value 都要保留下来。context 越长，KV cache 越大。

MQA（multi-query attention）让多个 query heads 共享一组 key/value。GQA（grouped-query attention）则折中一下，让一组 query heads 共享一组 key/value。

简单的说：

- MHA：每个注意力头都有自己的 K/V，效果强但 cache 大
- MQA：所有头共享 K/V，cache 小但可能损失表达力
- GQA：分组共享 K/V，在效果和成本之间折中


## 10. 一个现代 LLM block 大概长什么样

把这些东西合起来，一个现代 decoder-only LLM block 可以粗略写成：

$$
x' = x + \mathrm{Attention}(\mathrm{RMSNorm}(x))
$$

$$
y = x' + \mathrm{SwiGLU}(\mathrm{RMSNorm}(x'))
$$

attention 里面可能包含 RoPE、GQA、KV cache、FlashAttention 等实现细节。

## 12. 小结

现代 LLM 没有抛弃 Transformer，而是在 Transformer 的关键部件上做了系统性替换。

我的理解可以压缩成一句话：

> Transformer 提供了大模型的基本计算图，现代 LLM 的创新很多是在稳定训练、提高吞吐、降低推理成本、扩展上下文这几个方向上补工程和结构细节。

所以读后面的 MoE、长上下文、reasoning model、多模态模型时，不要只看最外层的概念。很多真正影响模型能力和成本的东西，藏在 block 级别的设计里。

## 参考资料

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
