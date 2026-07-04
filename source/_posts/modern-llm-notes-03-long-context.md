---
title: "现代大模型笔记 03：长上下文，注意力、KV cache 与位置编码"
date: 2025-09-20 21:00:00
updated: 2025-09-20 21:00:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - 大模型
  - 长上下文
  - 注意力机制
  - KV cache
  - RoPE
  - FlashAttention
  - 笔记
math: true
category_bar: true
---

这一篇整理长上下文。

很多模型现在都会宣传 32K、128K、1M context window。这个数字很吸引人，但它很容易被误解。

长上下文不是“把 max length 参数调大”。真正困难的地方有三类：

- attention 计算量变大
- KV cache 显存变大
- 上下文变长后，模型未必能稳定用好信息

所以这篇重点不是列方法，而是看长上下文到底在解决什么瓶颈。

## 1. Self-attention 为什么贵

标准 self-attention 是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

如果 sequence length 是 $n$，那么 $QK^\top$ 是一个 $n \times n$ 的矩阵。

这意味着 attention 的时间和显存复杂度通常和 $n^2$ 有关。

当上下文从 4K 增加到 32K，长度变成 8 倍，attention matrix 的规模大约变成 64 倍。

这就是为什么长上下文不是线性加 token 那么简单。

当然，实际实现里有 FlashAttention、分块计算、并行策略等优化，不一定真的显式保存完整 attention matrix。但底层的两两 token 交互压力仍然存在。

## 2. Prefill 和 decode 是两种成本

LLM 推理可以拆成两个阶段。

Prefill 阶段：模型一次性处理 prompt，把所有输入 token 的 hidden states 和 KV cache 算出来。

Decode 阶段：模型每次生成一个新 token，并复用前面的 KV cache。

这两个阶段的瓶颈不同。

长 prompt 会让 prefill 很贵，因为 prompt 内部 token 之间要做 attention。

长生成会让 decode 过程中 KV cache 越来越大，因为每一步都要关注过去所有 token。

所以同样是“长上下文”，有两种场景：

- 读一个很长的文档，然后回答几个问题
- 持续进行很长的对话或生成很长的内容

它们的系统瓶颈并不完全一样。

## 3. KV cache 为什么重要

自回归生成时，第 $t$ 步要用前面所有 token 的 key 和 value。

如果每一步都重新计算前面所有 token，成本会爆炸。

KV cache 的做法是把历史 token 的 key/value 存下来。

这样第 $t$ 步只需要算新 token 的 query，然后和缓存里的 key/value 做 attention。

但 cache 本身很占显存。

粗略看，KV cache 大小和这些因素成正比：

$$
\text{layers} \times \text{sequence length} \times \text{KV heads} \times \text{head dimension}
$$

所以 sequence length 越长，cache 越大。模型层数越多，cache 也越大。

这也是为什么 GQA、MQA、MLA 这类方法重要。它们都在不同程度上减少 key/value 的存储或计算成本。

## 4. MHA、MQA、GQA 与 MLA

标准 MHA 中，每个 query head 都有自己的 key/value head。

MQA 让所有 query heads 共享一组 key/value。

GQA 折中，让一组 query heads 共享一组 key/value。

它们主要是在减少 KV cache。

DeepSeek 的 MLA 更进一步。MLA 把 key/value 压到一个 latent 表示里，再在需要时恢复出用于 attention 的表示。它的目标也是降低 KV cache，但做法不是简单减少 KV heads，而是改变缓存内容的表示方式。

这里可以抓住一个直觉：

> 长上下文推理里，缓存什么、缓存多少、怎么缓存，是和 attention 公式同样重要的问题。

很多论文看起来在改 attention，实际是在改 inference memory footprint。

## 5. FlashAttention 解决的不是同一个问题

FlashAttention 优化的是 attention 计算过程中的内存读写。

标准实现可能会把 $QK^\top$ 这样的中间矩阵写入 HBM，再读回来做 softmax 和乘 $V$。

FlashAttention 用 tiling 和 online softmax，把计算分块放进更快的 SRAM，减少 HBM 访问。

所以它不是让 attention 从 $O(n^2)$ 变成 $O(n)$，而是让同样的 exact attention 在 GPU 上跑得更高效、占用更少显存。

这点很关键。

如果一个方法宣称“更长上下文”，要看它是在：

- 降低理论复杂度
- 降低实际显存占用
- 降低 KV cache
- 改善位置外推
- 提升长文本检索能力

这些不是一回事。

## 6. 位置编码的外推问题

模型训练时如果只见过 4K 长度，直接推到 32K 可能出问题。

因为 position embedding 或 RoPE 的位置范围发生了变化。

RoPE 的好处是相对位置信息更自然，但它也有频率外推问题。位置很大时，旋转角度分布可能和训练时不一样。

所以会有 RoPE scaling、NTK-aware scaling、YaRN 等方法。

这些方法的目标不是让模型“突然理解长文档”，而是先解决位置编码在更长长度下不崩。

换句话说：

> 位置外推解决的是模型能不能在更长坐标系里正常工作，不等于模型真的会做长程推理。

这也是为什么有些模型标称 128K，但实际问很长文档里的细节时，表现并不稳定。

## 7. Lost in the middle

长上下文还有一个能力问题：模型可能看得见，但用不好。

一种常见现象是 lost in the middle。信息放在开头或结尾时，模型更容易找到；信息放在中间时，模型容易漏掉。

这说明 context window 大小和 effective context length 不一样。

Context window 是系统允许输入多少 token。

Effective context length 是模型在任务中能可靠利用多少 token。

这两个数字差很多时，长上下文就更像“能塞进去”，而不是“能读明白”。

## 8. 长上下文任务的类型

我觉得长上下文至少要分三类看。

第一类是 retrieval-heavy 任务。

比如在长文档里找一个事实。这里关键是模型能不能定位证据。

第二类是 synthesis-heavy 任务。

比如总结一本书、多篇论文、一个项目代码库。这里关键不是找一句话，而是整合多个位置的信息。

第三类是 reasoning-heavy 任务。

比如给很长的实验日志，判断哪一步导致结果异常。这里需要模型在长上下文里做因果判断。

很多 benchmark 只测第一类，但真实使用里后两类更难。

## 9. Context compression

如果上下文太长，一种思路是压缩。

压缩可以发生在很多层面：

- 文本层面：先摘要，再输入模型
- 检索层面：只取相关 chunks
- 表示层面：把多个 token 压成少量 memory tokens
- KV 层面：压缩或丢弃部分 cache

这些方法的风险是信息丢失。

压缩越强，成本越低，但越可能丢掉后面问题需要的细节。

所以长上下文和 RAG 不是完全替代关系。长上下文让模型能直接读更多内容，RAG 让系统先筛掉明显无关内容。实际系统里经常两者结合。

## 10. Sparse attention

Sparse attention 试图减少 token 两两交互。

不是每个 token 都关注所有 token，而是只关注局部窗口、全局 token，或者某种稀疏模式。

它的好处是复杂度更低。

问题是稀疏模式可能限制信息流。如果任务需要两个相距很远的位置直接交互，稀疏 attention 未必能很好处理。

所以 sparse attention 的关键问题是：稀疏模式是不是和任务结构匹配。

代码、文档、视频、对话的长程依赖形态不一样。一个固定稀疏模式很难对所有任务都最优。

## 11. 为什么 128K 不等于真正会用 128K

假设模型支持 128K context。

这至少包含三层含义：

第一，系统层面可以接收 128K token。

第二，模型结构和位置编码在 128K 下不会明显崩。

第三，模型在真实任务中能稳定利用 128K 里的关键信息。

很多时候宣传的是第一层或第二层，用户期待的是第三层。

这就是长上下文体验落差的来源。

所以读模型报告时，不能只看 max context length，还要看它怎么评估长上下文能力。Needle-in-a-haystack 只能说明模型能不能找针，不能完全说明它能不能理解一整份复杂材料。

## 12. 读长上下文论文时看什么

我会看这些问题：

- 训练时最大长度是多少，推理时最大长度是多少
- 使用什么位置编码和 scaling 方法
- attention 是 exact、sparse、linear，还是其他变体
- KV cache 有没有压缩
- prefill 和 decode 的吞吐分别如何
- benchmark 是检索型，还是综合理解型
- 长度增加后性能下降曲线怎样
- 有没有和 RAG 或 compression 结合

尤其要注意：有些方法提升的是“能跑更长”，有些方法提升的是“用得更好”。这两个目标相关，但不是同一个目标。

## 13. 小结

长上下文的核心不是把窗口开大，而是同时处理三个问题：

- 算得动
- 存得下
- 用得好

FlashAttention、GQA、MLA 主要偏向算得动和存得下。

RoPE scaling、YaRN 主要解决位置外推。

长文本训练、检索式评估、context compression 更接近用得好的问题。

所以看一个长上下文模型时，我不会只看 32K、128K、1M 这些数字。真正要问的是：它在什么任务上证明了自己能稳定使用这么长的上下文。

## 参考资料

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Peng et al., [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- Liu et al., [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
