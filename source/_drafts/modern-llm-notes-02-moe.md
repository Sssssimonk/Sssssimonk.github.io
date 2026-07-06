---
title: "现代大模型笔记 02：MoE，为什么大模型开始变稀疏"
date: 2025-09-13 20:30:00
updated: 2025-09-13 20:30:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - 大模型
  - MoE
  - 稀疏激活
  - DeepSeek
  - Qwen
  - 笔记
math: true
category_bar: true
---

这一篇整理 MoE，也就是 mixture of experts。

现在很多前沿开源模型都会强调两个数字：total parameters 和 activated parameters。比如一个模型可能有 235B 总参数，但每个 token 只激活 22B 参数。

第一次看到这个说法时很容易觉得矛盾：到底是 235B 模型，还是 22B 模型？

MoE 的关键就在这里。它不是让所有参数每次都参与计算，而是给每个 token 选择一小部分 expert 来算。

## 1. Dense model 的问题

普通 Transformer 是 dense model。

每个 token 经过每一层时，都会使用这一层的全部参数。

如果模型变大，计算量也跟着变大。一个 dense 70B 模型和一个 dense 7B 模型相比，容量更大，但每个 token 的计算成本也明显更高。

这带来一个很现实的问题：参数规模、训练成本、推理成本几乎绑在一起。

想要更强模型，就要更大参数；参数越大，每次推理越贵。

MoE 试图拆开这件事：

- 总参数可以很大，用来提供更高容量
- 每个 token 只用一部分参数，用来控制计算成本

也就是说，MoE 想要的是“大容量”和“低激活成本”同时存在。

## 2. Expert 是什么

在很多 MoE Transformer 里，attention 层仍然是 shared 的，MoE 主要替换 FFN 层。

普通 FFN 是一个大的前馈网络：

$$
\mathrm{FFN}(x)
$$

MoE FFN 则有多个 expert：

$$
E_1(x), E_2(x), ..., E_N(x)
$$

每个 expert 本质上可以理解成一个 FFN。区别是，不是每个 token 都跑所有 expert，而是由 router 选择其中几个。

如果使用 top-2 routing，那么 token $x$ 的输出大概是：

$$
y = g_1(x)E_{i}(x) + g_2(x)E_{j}(x)
$$

这里 $E_i$ 和 $E_j$ 是被选中的两个 experts，$g_1$ 和 $g_2$ 是 router 给出的权重。

直觉上，模型在说：

> 这个 token 更适合交给这些 expert 处理。

## 3. Router：MoE 的真正难点

MoE 听起来像“多放几个专家，然后自动选择”。但真正难的是 router。

Router 通常会对每个 token 算一个分数：

$$
s = xW_r
$$

再从所有 expert 里选 top-k。

问题是，如果 router 学得不好，所有 token 都挤向少数几个 expert，其他 expert 几乎不用。这样不仅浪费参数，还会导致某些 expert 负载爆炸。

所以 MoE 训练里经常要处理 load balancing。

Switch Transformer 里用了 auxiliary loss 来鼓励 token 更均匀地分配到不同 expert。DeepSeek-V3 则强调 auxiliary-loss-free 的 load balancing，试图减少辅助损失对模型主目标的干扰。

这里有一个技术取舍：

- 加 auxiliary loss，路由更容易均衡，但可能影响语言建模目标
- 不加或弱化 auxiliary loss，主目标更干净，但路由稳定性更难保证

MoE 的难点不是“有多个 expert”，而是“怎么让 expert 被合理使用”。

## 4. 总参数和激活参数

MoE 报告里经常出现两个数字。

Total parameters 指模型所有参数，包括所有 experts。

Activated parameters 指处理一个 token 时实际参与计算的参数。

比如一个 MoE 模型有 64 个 experts，每次只选 2 个。那总参数可能很大，但每个 token 只走其中一小部分。

这就解释了为什么一个模型可以同时标注为 235B total parameters 和 22B activated parameters。

它的容量接近一个超大模型，但单 token 计算量更接近一个中等模型。

不过这句话不能简单理解成“235B 的效果，22B 的成本”。因为 MoE 还有通信、路由、负载均衡、batching 等额外问题。

## 5. MoE 为什么适合大规模预训练

预训练阶段，模型会看到海量 token。不同 token 的统计特征差异很大：代码、数学、中文、英文、表格、对话、网页文本。

Dense model 对所有 token 使用同一组 FFN 参数。

MoE 则允许不同 token 走不同 expert。理论上，不同 expert 可以学到不同类型的模式。

比如某些 expert 可能更常处理代码 token，某些 expert 可能更常处理数学符号，某些 expert 更常处理自然语言。

这里要注意，我不会把 expert 理解成非常清晰的人类职业分工。它们不一定真的变成“数学专家”“代码专家”。更准确地说，expert 是被 router 动态选择的参数子空间。

这个区别很重要。技术报告里如果没有分析 expert specialization，就不要强行给每个 expert 贴人类标签。

## 6. MoE 的通信成本

MoE 在单卡上很容易理解，但大模型训练通常跨很多 GPU。

如果 expert 分布在不同 GPU 上，token 被 router 分配后，就需要在 GPU 之间传输。

这会引入 all-to-all communication。

Dense model 的通信主要来自模型并行、数据并行、梯度同步等。MoE 额外增加了 token dispatch 和 combine 的通信。

这就是为什么 MoE 论文和技术报告会关心：

- expert parallelism
- load balancing
- token dropping
- capacity factor
- communication-computation overlap

这些听起来像工程细节，但它们决定 MoE 能不能真的跑起来。

一个 routing 很漂亮但通信开销巨大的 MoE，在真实训练里可能并不划算。

## 7. Capacity factor 和 token dropping

每个 expert 每个 batch 能处理的 token 数通常会有限制。

如果某个 expert 被分配了太多 token，超出的 token 可能被 drop，或者走 fallback 路径。

Capacity factor 控制每个 expert 的容量上限。

如果 capacity 太小，容易 drop token，训练信号受损。

如果 capacity 太大，显存和计算浪费增加。

所以 MoE 的训练不是简单调 learning rate。它多了一组系统参数，而且这些参数和硬件拓扑、batch size、sequence length 都有关。

## 8. DeepSeekMoE 的思路

DeepSeek 系列技术报告里，MoE 是核心结构之一。

DeepSeek-V3 使用 671B total parameters，但每个 token 激活约 37B 参数。它还结合了 MLA、multi-token prediction、FP8 training 等工程优化。

这里值得注意的是：DeepSeek-V3 的强并不只是因为 MoE。

MoE 提供参数容量和计算效率，MLA 降低 KV cache 成本，FP8 降低训练成本，多 token prediction 尝试提高生成效率。

所以读 DeepSeek-V3 时，不要把结论压缩成“MoE 很强”。更准确的说法是：

> DeepSeek-V3 是一组结构、训练和系统优化叠加出来的结果，MoE 是其中负责参数容量扩展的一块。

## 9. Qwen3 MoE 的位置

Qwen3 技术报告里同时有 dense 和 MoE 模型。

这很有意思，因为它说明 dense 和 MoE 不是谁完全替代谁，而是面向不同规模和部署场景的选择。

小模型用 dense 可能更简单、更稳定、更容易部署。

大模型用 MoE 可以把总容量拉高，同时控制每 token 计算。

所以模型家族通常会同时提供 dense 和 MoE 版本，让用户在延迟、成本、能力之间选。

## 10. MoE 不是免费午餐

MoE 最大的误区是：只看 activated parameters，然后以为成本就等于一个小 dense 模型。

实际不是。

MoE 至少有这些额外代价：

- Router 本身要计算
- Token dispatch 需要通信
- Expert 负载不均衡会降低硬件利用率
- 小 batch 推理时可能不如 dense 规整
- 训练稳定性更复杂
- Serving 系统更难做

这也是为什么 MoE 更像系统工程，而不只是模型结构。

如果部署环境很简单，dense model 可能反而更好。MoE 的优势通常要在足够大规模、足够好的并行系统里才能发挥。

## 11. 一个简单例子

假设一个客服模型要处理三类问题：

- 订单状态
- 退款政策
- 技术故障

Dense model 会用同一套 FFN 参数处理所有问题。

MoE model 可能学到某些 token 更常走某些 expert。和订单相关的表达经常激活一组 expert，技术故障相关的表达激活另一组 expert。

但真实 LLM 里的分工不会这么干净。一个 token 的意义受上下文影响，同一个词在代码、数学、闲聊里可能完全不同。

所以 MoE 的 router 不是按关键词查表，而是根据 hidden state 做动态选择。

## 12. 读 MoE 模型报告时看什么

我会重点看这些问题：

- 总参数和激活参数分别是多少
- 每层都有 MoE，还是部分层有 MoE
- 每次 routing 选 top-1、top-2，还是更多 expert
- 有没有 shared expert
- load balancing 怎么做
- 是否使用 auxiliary loss
- expert 数量是多少
- 训练和推理时的通信开销怎么处理
- benchmark 提升是来自 MoE 本身，还是同时叠加了数据、RL、上下文扩展

这些问题比“用了 MoE”更重要。

## 13. 小结

MoE 的核心思想很直接：

> 用稀疏激活把总参数容量和单 token 计算成本拆开。

但 MoE 的真实难点也很明确：

> 参数变稀疏以后，训练和系统复杂度会显著上升。

所以我会把 MoE 看成一种大规模模型扩展路线，而不是一个单纯提升模型能力的小技巧。

它适合的问题是：当 dense scaling 越来越贵时，能不能用更聪明的参数激活方式继续扩展模型容量。

## 参考资料

- Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- Fedus et al., [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- GLM-4.5 Team, [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471)
- Moonshot AI, [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
