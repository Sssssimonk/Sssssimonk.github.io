---
title: "现代大模型笔记 01：Transformer 架构与现代改进"
date: 2025-09-06 20:00:00
updated: 2025-09-06 20:00:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - 大模型
  - Transformer
  - RoPE
  - KV cache
  - 笔记
math: true
category_bar: true
---

![transformer架构图](/img/transformer_architecture.png)

这一篇整理 Transformer 架构本身，以及它在现代 LLM 里发生了哪些变化。

Attention 的基础公式已经在深度学习笔记里讲过了，这里不再重复推 Q/K/V。这里更关心：

```text
一个 Transformer block 长什么样，为什么现代 LLM 会改成现在这个形态。
```

## 1. 原始 Transformer 是 encoder-decoder

原始 Transformer 是 seq2seq 架构。

Encoder 负责读完整个输入句子，decoder 负责一步一步生成目标句子。

这个结构很适合机器翻译：

```text
source sentence -> encoder -> decoder -> target sentence
```

Encoder 里是 bidirectional self-attention，每个 token 可以看完整输入。

Decoder 里是 causal self-attention，加上 cross-attention 去看 encoder 输出。

所以原始 Transformer 不是 GPT 现在这种纯 decoder-only 架构。

## 2. 为什么现代 LLM 多用 decoder-only

GPT、LLaMA、Qwen、DeepSeek 这类通用语言模型，大多采用 decoder-only。

它没有单独的 encoder，而是把任务统一成 next token prediction：

$$
p(x_t \mid x_1, x_2, ..., x_{t-1})
$$

翻译、问答、总结、代码生成、数学推理，都可以改写成：

```text
前面是一段 prompt，后面继续生成答案
```

这个形式简单，而且和大规模自监督预训练天然对齐。

BERT 这类 encoder-only 模型更像是在理解一整段文本；GPT 这类 decoder-only 模型更像是在从左到右生成文本。

Decoder-only 不一定在所有任务上都“结构最优”，但它足够统一，容易扩展，也方便把不同任务都塞进同一个生成接口里。

## 3. Causal mask 决定了生成形态

Decoder-only 模型使用 causal self-attention。

普通 self-attention 中，每个 token 可以看见所有 token。但语言生成时，当前位置不能偷看未来 token，否则 next token prediction 就变成作弊。

所以 causal attention 会加 mask，让第 $t$ 个位置只能关注 $1$ 到 $t$ 的 token。

如果 prompt 是：

```text
The cat sits on the
```

模型预测下一个词时，只能看见这段前缀，不能提前看见答案。

这个 mask 看起来只是训练细节，但它决定了 LLM 的能力形态：模型不是一次性把完整答案算出来，而是一步一步生成。前面生成偏了，后面就会被带偏。

## 4. Transformer block：attention + FFN + 残差

一个现代 decoder-only Transformer block 大概长这样：

```text
x
-> Norm
-> Self-Attention
-> Residual Add
-> Norm
-> FFN / MLP
-> Residual Add
```

Self-attention 负责 token 之间的信息交互。

FFN 负责对每个 token 的表示做非线性加工。

Residual connection 让深层网络更容易训练：

$$
x_{l+1} = x_l + F(x_l)
$$

如果没有 residual，几十层、上百层模型很难稳定训练。Residual path 可以理解成一条信息和梯度的主干通路，每个 block 在上面叠加修改。

## 5. Pre-Norm：深层模型更稳

原始 Transformer 使用 Post-Norm，大概形式是：

$$
x_{l+1} = \mathrm{LayerNorm}(x_l + F(x_l))
$$

现代 LLM 更常见的是 Pre-Norm：

$$
x_{l+1} = x_l + F(\mathrm{Norm}(x_l))
$$

差别是 normalization 放在 residual branch 前面还是后面。

Pre-Norm 的好处是深层模型训练更稳定。直觉上，residual path 更像一条比较干净的通路，梯度可以更直接地传，不会每层都先被 normalization 包住。

所以在现代 LLM 里，Pre-Norm 基本成了主流选择。

## 6. RMSNorm：保留尺度控制，少做一点计算

LayerNorm 会减均值、除标准差：

$$
\mathrm{LayerNorm}(x)
=
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} \odot g + b
$$

RMSNorm 去掉了减均值，只保留 root mean square 的尺度归一化：

$$
\mathrm{RMSNorm}(x)
=
\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}} \odot g
$$

它关心的是向量整体尺度，而不是每一维相对均值的位置。

这有两个好处。

第一，计算更简单。单层省一点，放到几十层、几百亿参数规模上就很明显。

第二，它保留了最关键的稳定性作用：控制 activation 的尺度。

所以很多现代 LLM 用 RMSNorm 替代 LayerNorm。

## 7. FFN：从 ReLU 到 SwiGLU，再到 MoE

Transformer block 里除了 attention，还有 FFN。

原始 Transformer 用的是：

$$
\mathrm{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2
$$

现代 LLM 常用 gated FFN，比如 SwiGLU：

$$
\mathrm{SwiGLU}(x)
=
\mathrm{Swish}(xW_1) \odot (xW_2)
$$

再接一个输出投影。

Gating 的直觉是：模型不只是对特征做非线性变换，还学会哪些通道该打开、哪些通道该压下去。

如果把 FFN 看成每个 token 独立做一次特征加工，那么 SwiGLU 相当于给这次加工加了一个动态开关。

MoE 可以看成 FFN 的进一步扩展：不是每个 token 都走同一个大 MLP，而是路由到少数 experts。这样可以增加总参数量，但每个 token 的实际计算量不一定同比增加。

## 8. RoPE：把位置信息放进 attention score

Self-attention 本身不包含顺序信息。对 attention 来说，如果不加位置编码，`A B C` 和 `C B A` 很难区分谁在前谁在后。

原始 Transformer 用 absolute positional encoding，把位置向量加到 token embedding 上。

现代 LLM 常用 RoPE，也就是 rotary position embedding。它不是简单把位置向量加进去，而是在 query 和 key 上做旋转，使 attention score 自然带上相对位置信息。

可以粗略理解成：

$$
q_m^\top k_n
\rightarrow
\text{content similarity with relative position } (m-n)
$$

RoPE 的好处是相对位置关系更自然，也更适合一定程度的位置外推。

但 RoPE scaling 不等于模型自动会用超长上下文。位置能编码是一回事，模型能不能从 100K token 里稳定找证据，是另一回事。

## 9. MHA、MQA、GQA、MLA：注意力结构也在服务推理效率

标准 multi-head attention 里，每个 head 都有自己的 $Q, K, V$。

这在训练和表达力上很自然，但推理时会遇到 KV cache 问题。

生成第 $t$ 个 token 时，前面所有 token 的 key 和 value 都要保留下来。context 越长、层数越多，KV cache 越大。

所以现代模型经常改 attention 的 K/V 结构。

- MHA：每个 query head 都有自己的 K/V，表达力强，但 cache 大
- MQA：所有 query heads 共享一组 K/V，cache 小，但表达力可能受影响
- GQA：一组 query heads 共享一组 K/V，在效果和成本之间折中
- MLA：把 K/V 压到 latent 表示里缓存，需要时再恢复，进一步降低 KV cache

这里要注意：这些改动看起来是 attention 架构变化，但背后很多时候是在优化 inference memory footprint。

模型结构不是只为训练 loss 服务，也要为推理吞吐和显存服务。

## 10. 几个点

- 原始 Transformer 是 encoder-decoder，现代通用 LLM 多是 decoder-only。
- Decoder-only 的核心目标是 next token prediction，causal mask 决定它从左到右生成。
- Transformer block 不是只有 attention，还包括 FFN、residual、normalization。
- Pre-Norm、RMSNorm、SwiGLU 这些改动主要服务训练稳定性和计算效率。
- MQA、GQA、MLA 这类注意力变化，和 KV cache、推理成本强相关。

## 参考资料

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
