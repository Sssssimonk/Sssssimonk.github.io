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

本篇整理Transformer架构以及它在现代 LLM 里发生了哪些变化。
从图中可以看到，几个核心的步骤分别是：
1. word embedding
2. positional embedding
3. 注意力机制
4. Residual Connect
5. Normalization
6. FFN

## 原始 Transformer的encoder-decoder结构

原始 Transformer 是 seq2seq 架构。

Encoder 负责读完整个输入句子输出一个隐藏向量给decoder，decoder负责一步一步生成目标句子。


Encoder里是bidirectional self-attention，每个 token 可以看完整输入。Decoder 里是 causal self-attention，加上cross-attention 去看encoder输出。

> 那为什么现代 LLM 多用 decoder-only

GPT、LLaMA、Qwen、DeepSeek 这类通用语言模型，大多采用 decoder-only。

它没有单独的 encoder，而是把任务统一成 next token prediction：

$$
p(x_t \mid x_1, x_2, ..., x_{t-1})
$$


主要是形式简单，而且和大规模**自监督**预训练天然对齐。

## Transformer block：attention + FFN + 残差

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

### Normalization：深层模型更稳

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

**RMSNorm：保留尺度控制，少做一点计算** 

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


### FFN：从ReLU 到 SwiGLU，再到 MoE

Transformer block 里除了 attention，还有FFN。

原始 Transformer 用的是ReLu作为激活函数，后续有相应的改进，比如Leaky ReLU，GeLU之类的，目的是让更多的残差（信息）通过梯度来更新参数。
$$
\mathrm{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2
$$

随着技术进步，SwiGLU进入了视野：
SwiGLU主要有两个组成部分，Swish 和GLU。


$$
\mathrm{SwiGLU}(x)
=
\mathrm{Swish}(xW_1) \odot (xW_2)
$$

再接一个输出投影。

Gating 的直觉是：模型不只是对特征做非线性变换，还学会哪些通道该打开、哪些通道该压下去。
如果把 FFN 看成每个 token 独立做一次特征加工，那么 SwiGLU 相当于给这次加工加了一个动态开关。

MoE 可以看成 FFN 的进一步扩展：不是每个 token 都走同一个大 MLP，而是路由到少数 experts。这样可以增加总参数量，但每个 token 的实际计算量不一定同比增加。后续文章中会文门提到

## RoPe 位置编码

Self-attention 本身不包含顺序信息。对 attention 来说，如果不加位置编码，`A B C` 和 `C B A` 很难区分谁在前谁在后。

原始 Transformer用absolute positional encoding，直接把位置向量加到 token embedding 上。

现代 LLM 常用 RoPE，也就是 rotary position embedding。最早来自于苏剑林的Reformer工作。[RoPe解读博客链接](https://spaces.ac.cn/archives/8265/comment-page-1)

它不是简单把位置向量加进去，而是在 query 和 key 上做旋转，使 attention score 自然带上相对位置信息。而且这种方式天然适合长度外度，可以通过一些方式让模型理解**超出训练长度的**文本

可以粗略理解成：

$$
q_m^\top k_n
\rightarrow
\text{content similarity with relative position } (m-n)
$$



### 注意力机制的变迁

标准 multi-head attention 里，每个 head 都有自己的 $Q, K, V$。

这在训练和表达力上很自然，但推理时会遇到 KV cache 问题。

生成第 $t$ 个 token 时，前面所有 token 的 key 和 value 都要保留下来。context 越长、层数越多，KV cache 越大。

所以现代模型经常改 attention 的 K/V 结构。

- MHA：每个 query head 都有自己的 K/V，表达力强，但 cache 大
- MQA：所有 query heads 共享一组 K/V，cache 小，但表达力可能受影响
- GQA：一组 query heads 共享一组 K/V，在效果和成本之间折中
- MLA：把 K/V 压到 latent 表示里缓存，需要时再恢复，进一步降低 KV cache

下为MHA的简易实现
```python
import torch.nn as nn
import torch

class MHA(nn.Module):

    def __init__(self, head_num, hidden_size):
        super().__init__(self)

        self.head_num = head_num
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // head_num

        self.q = torch.Linear(hidden_size, hidden_size)
        self.k
        self.v

        self.o


    def forward(self, hidden_state, masking=None, use_cache=True, past_kv):
        # 0. 设置dk, batch_size
        # 1. 变化qkv，拆分出多头
        # 2. 拼接kv cache
        # 3. 计算attention score
        # 4. 加入masking
        # 5. 计算attention output
        # 6. 计算attention prob
        # 7. 计算output，拼接多头

        batch_size = hidden_state.size(0)
        d_k = torch.sqrt(self.head_dim)

        q = self.q(hidden_state)
        k = self.k(hidden_state)
        v = self.v(hidden_state)

        # [batch, seq_len, hidden_size]
        q = q.view(batch_size, -1, self.head_num, self.head_dim).tranpose(1, 2)
        # [batch, head_num, seq_len, head_dim]
        k = k.view(batch_size, -1, self.head_num, self.head_dim).tranpose(1, 2)
        v

        if kv_cache:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        past_kv = (k, v)

        attention_score = torch.matmul(q, k.tranpose(-1, -2)) / d_k

        if masking:
            attention_score += masking

        attention_prob = torch.softmax(attention_score, dim = -1)
        attention_output = torch.matmul(attention_prob, v)

        o = torch.view(attention_output)

        o = self.o(attention_output)


        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        attention_score = torch.matmul(q, k.transpose(-1, -2)) / self.key_dim**0.5
        attention_score += attn_mask

        attention_prob = torch.softmax(attention_score, dim=-1)

        out = torch.matmul(attention_prob, v) 

        self.tok_embedding += self.pos_embedding
        attn_mask = torch.ones(N, L, L, dtype=torch.bool, device=self.tok_embedding.device)

        attention_output = self.tok_embedding + self.layer_norm_1(self.self_attention(self.tok_embedding, attn_mask))
        attention_output = attention_output + self.layer_norm_2(self.fnn(attention_output))

        logit = self.head(attention_output)

```




## 参考资料

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- 苏剑林的科学空间，[RoPe解读博客链接](https://spaces.ac.cn/archives/8265/comment-page-1)
