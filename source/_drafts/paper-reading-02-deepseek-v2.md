---
title: "论文精读 02：DeepSeek-V2，MLA 与 DeepSeekMoE 如何降低大模型成本"
date: 2026-08-21 21:00:00
updated: 2026-08-23 23:30:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - 论文精读
  - DeepSeek-V2
  - MLA
  - MoE
  - KV Cache
math: true
category_bar: true
---

DeepSeek-V2 的参数表看起来很夸张：总参数 236B，但每个 token 只激活 21B。论文又给出另外三个数字：相对 DeepSeek 67B，训练成本降低 42.5%，KV cache 减少 93.3%，最大生成吞吐提升到 5.76 倍。

这些数字经常被一起引用，容易让人以为它们来自同一个技巧。其实 DeepSeek-V2 的架构主线很清楚：MLA 主要解决 attention 推理时的 KV cache，DeepSeekMoE 主要解决 FFN 的训练与计算成本。两者碰巧都在做同一件更大的事——把模型容量和每个 token 实际付出的成本拆开。

<!-- more -->

## 先从成本在哪里说起

一个 Transformer block 里，最重的两块通常是 attention 和 FFN。

推理时，attention 为了让新 token 看到历史，需要保存每一层、每个历史 token 的 Key 和 Value，也就是 KV cache。上下文越长、batch 越大，这部分显存越容易成为瓶颈。

FFN 则占了模型参数和计算的大头。如果每个 token 都经过同一个巨大的 dense FFN，扩大模型容量就会直接扩大每 token 的计算量。

DeepSeek-V2 分别下手：

- MLA：历史 token 不再缓存完整的多头 K/V，而是缓存一个低维 latent；
- DeepSeekMoE：模型保留大量专家参数，但每个 token 只走少数专家。

所以“236B 总参数，21B 激活参数”不是一个宣传口径上的小把戏，而是整个架构的目标。

## MHA 的 KV cache 为什么贵

标准 Multi-Head Attention 中，第 $t$ 个 token 的隐状态 $h_t$ 会投影成每个 head 的 Query、Key 和 Value：

$$
q_t=W^Qh_t,\qquad k_t=W^Kh_t,\qquad v_t=W^Vh_t
$$

自回归生成时，过去 token 的 $k_t,v_t$ 不会再变，所以可以缓存起来。问题是每一层都要为每个 token 缓存所有 head 的 K/V，缓存量大致随下面几项线性增长：

$$
\text{KV cache}\propto L_{\text{seq}}\times L_{\text{layer}}\times n_h\times d_h\times 2
$$

MQA 让所有 query head 共享一组 K/V，缓存很省，但容易损失性能；GQA 在两者之间，让若干 query head 共享一组 K/V。

MLA 走的不是“到底保留几组 K/V”这条路，而是先问：完整 K/V 是否存在一个更低维的公共表示？

## MLA 的核心：缓存 latent，而不是缓存 K 和 V

MLA 对 Key 和 Value 做联合低秩压缩：

$$
c_t^{KV}=W^{DKV}h_t
$$

$$
k_t^C=W^{UK}c_t^{KV},\qquad v_t^C=W^{UV}c_t^{KV}
$$

其中 $c_t^{KV}\in\mathbb{R}^{d_c}$ 是压缩后的 latent，且 $d_c\ll n_hd_h$。

直觉上，MHA 是把一本书拆成很多份笔记并全部存下来；MLA 只存一份压缩索引，需要时再通过 up-projection 恢复各个 head 使用的表示。

推理时真正需要缓存的主要是 $c_t^{KV}$。更进一步，论文指出 $W^{UK}$ 可以吸收到 query 投影中，$W^{UV}$ 可以吸收到输出投影中，因此并不一定要显式恢复完整 K/V 后再做 attention。

这也是 MLA 比普通 low-rank approximation 更有意思的地方。它不只是训练完再压缩一个权重矩阵，而是把“可缓存的低维状态”直接写进 attention 架构。

DeepSeek-V2 中 $d_c=4d_h$。论文给出的换算是：它每 token 的 KV cache 大约相当于只有 2.25 个 group 的 GQA，但模型效果可以强于 MHA。

## RoPE 带来的麻烦：位置编码不能随便被吸收

如果事情只到低秩压缩，MLA 会简单很多。真正麻烦的是 RoPE。

Rotary Position Embedding 会对 Query 和 Key 应用依赖位置的旋转。这个变换发生在每个位置上，因此带 RoPE 的 Key 不能像普通线性投影那样完全吸收到别的矩阵里。直接把 RoPE 用在重建后的全部 Key 维度，会破坏 MLA 想要的缓存优势。

论文采用 decoupled RoPE：把 attention 表示拆成内容部分和位置部分。

$$
q_{t,i}=[q^C_{t,i};q^R_{t,i}],\qquad
k_{t,i}=[k^C_{t,i};k^R_t]
$$

内容 Key $k^C$ 来自压缩 latent；较小的 $k^R$ 专门承载 RoPE，并在不同 head 间共享。这样缓存由两部分组成：低维的 $c_t^{KV}$，以及尺寸较小的位置 Key。

这一步看起来像实现细节，实际上决定了 MLA 能不能同时保住长上下文性能和缓存压缩率。很多漂亮的低秩公式，一遇到位置编码和高效 kernel 就不再漂亮了；MLA 的工程价值恰恰在于它把这些约束一起处理了。

## DeepSeekMoE：不是简单把 FFN 分成很多专家

普通 MoE 会准备多个 FFN expert，由 router 为每个 token 选择 Top-K。这样总参数可以很大，但每个 token 只激活少数参数。

DeepSeekMoE 在这个框架上强调两个设计。

第一个是 fine-grained expert segmentation。

与其使用少量很宽的专家，不如把专家切得更细，再让一个 token 同时组合多个小专家。在总参数量和激活参数量相近的前提下，更细的专家提供了更多组合方式，也更容易形成细粒度的专业化。

第二个是 shared expert isolation。

有些知识是大量 token 都需要的。如果完全依赖 routed experts，每个专家可能都重复学习一份公共能力，既浪费参数，也挤占专业化空间。DeepSeekMoE 把一部分专家设为 shared experts，让所有 token 都经过它们；其余 routed experts 再学习更有差异的知识。

一层的输出可以粗略写成：

$$
h'_t=\sum_{i=1}^{N_s}\mathrm{FFN}^{(s)}_i(u_t)
+\sum_{i\in\mathrm{TopK}(u_t)}g_{i,t}\mathrm{FFN}^{(r)}_i(u_t)
$$

前一项是共享专家，后一项是路由选中的专家。

在 DeepSeek-V2 中，除第一层外都换成 MoE layer。每层包含 2 个 shared experts 和 160 个 routed experts，每个 token 激活其中 6 个 routed experts。最终模型总参数 236B，每 token 激活约 21B。

## 专家越细，通信问题越难躲

MoE 省 FLOPs，不代表天然省时间。专家通常分布在不同设备上，token 要通过 all-to-all 通信发给对应专家。如果 Top-K 选中的专家散落在太多设备，通信会吃掉稀疏计算省下来的收益。

DeepSeek-V2 加入 device-limited routing：先选出 affinity 最高的少数设备，再只在这些设备上的专家中做 Top-K。论文训练时将 routed experts 分布在 8 个设备上，并限制每个 token 最多发送到 3 个设备。

另一个问题是 load balance。Router 如果偏爱少数专家，会导致热门设备过载、冷门专家训练不足。论文同时使用 expert-level、device-level 和 communication-level 的 balance loss，并在训练时对超出设备容量的低 affinity token 做 token dropping。

这部分不像 MLA 那么容易被记住，却决定了 MoE 是否真的能跑快。MoE 的核心矛盾一直都是：理论激活参数很少，实际系统能不能把 token 均匀、低通信地送到这些参数上。

## 论文中的训练与后训练

DeepSeek-V2 在 8.1T token 上预训练，最大预训练序列长度为 4K，之后通过 YaRN 扩展到 128K context。语料里中文 token 数量约比英文多 12%。

后训练包括 SFT 和 RL。论文在 RL 部分使用 GRPO，目的之一是省去和 policy 同规模的 critic model。今天看 DeepSeek-V2 时，人们更容易记住 MLA；但从 DeepSeekMath 到 V2 再到 R1，GRPO 也是另一条延续下来的技术线。

不过这篇文章还是把重点留在架构上。因为论文声称的几项效率收益，主要不是 RL 带来的：

- 42.5% 的训练成本降低，主要对应稀疏 MoE 和训练系统；
- 93.3% 的 KV cache 减少，主要对应 MLA；
- 5.76 倍最大生成吞吐，则是缓存、激活计算和系统实现共同作用的结果。

这些数字是相对 DeepSeek 67B、在论文自身配置下得到的，不应简单理解为任何部署环境都能原样复现。

## 为什么 DeepSeek-V2 影响很大

MLA 和 DeepSeekMoE 的共同点，是不再把“模型有多少能力”和“每个 token 必须搬运、缓存、计算多少东西”绑定在一起。

MLA 假设完整的多头 K/V 可以通过一个小 latent 表达；DeepSeekMoE 假设模型可以拥有许多参数，但一次只调用与当前 token 相关的少部分。一个压缩状态，一个稀疏激活。

这套思路后来在 DeepSeek-V3 和 R1 中继续出现，也解释了为什么 DeepSeek-V2 不只是一代模型的技术报告。它确立了一条很明确的扩展路线：不只追求更大的总参数，而是认真优化每个 token 的经济账。

## 论文没有完全解决的问题

第一，MLA 的收益高度依赖实现。矩阵吸收、量化、并行策略和 attention kernel 没跟上时，参数量上的节省不一定等价于端到端延迟的同比下降。

第二，MoE 的专家是否真的形成了可解释的专业分工，论文没有给出很深入的分析。Fine-grained segmentation 提供了更多组合空间，但“更多专家”不自动等于“更好的专业化”。

第三，balance loss 和 token dropping 是带权衡的。更强的均衡约束可能干扰 router 按语义自由选择；dropping 则意味着一部分 token 在训练时没有走完原本选中的专家路径。

因此 DeepSeek-V2 的意义不是找到一个没有代价的免费午餐，而是把大模型成本里两块最硬的瓶颈，分别改写成了可以继续优化的结构问题。

## 参考资料

- DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- DeepSeek-AI, [DeepSeek-V2 GitHub Repository](https://github.com/deepseek-ai/DeepSeek-V2)
- Dai et al., [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)

