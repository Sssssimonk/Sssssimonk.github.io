---
title: "后训练笔记 02：SFT 与 PEFT"
date: 2025-11-08 20:30:00
updated: 2025-11-08 20:30:00
categories:
  - 后训练笔记
tags:
  - 后训练
  - SFT
  - LoRA
  - Prefix Tuning
  - P-Tuning
  - PEFT
  - 笔记
math: true
category_bar: true
---

这一篇整理 SFT 和 PEFT。

在大模型后训练里，SFT 通常是第一步。Base model 经过大规模预训练后，已经学到语言、知识和模式，但它不一定会按指令回答。SFT 的作用就是先把模型拉到一个“能正常当助手”的状态。

PEFT则是SFT的一种解法（即parameter **efficient** fine-tuning：模型太大，全参数微调成本太高，所以只训练很少一部分参数，让模型适配新任务。


## 1. Base model 为什么不能直接用

Base model 的训练目标通常是 next token prediction：

$$
\max_\theta \sum_t \log p_\theta(x_t \mid x_{<t})
$$

它学的是文本分布。

如果给 base model 一个问题：

> 请解释一下 LoRA 是什么。

它可能回答，也可能继续补全网页文本，或者生成一段类似问答数据的续写。问题不一定是它“不知道”，而是它没有被训练成稳定的 instruction-following model。

SFT 的目标就是把输入输出形式固定下来：

$$
\max_\theta \log \pi_\theta(y \mid x)
$$

其中 $x$ 是 instruction，$y$ 是高质量回答。

### SFT 具体学到了什么

SFT 不只是“灌知识”。

它主要学这些行为：

- 遵循指令
- 按对话格式回答
- 输出更清晰的结构
- 遇到不确定问题时更谨慎
- 学会某些任务格式，比如代码解释、数学推理、摘要
- 初步形成 reasoning trace 的风格


### SFT 的局限

SFT 本质上是 imitation learning。

数据是什么样，模型就往什么样学。

如果数据里回答很模板化，模型也会模板化。

如果数据里推理过程只是表面解释，模型会学会写“看起来像推理”的文本，但不一定真的更会解题。

如果数据里有错误，模型会跟着学。

更关键的是，SFT 不知道两个正确回答哪个更好。

比如一个回答很简洁，一个回答更完整；一个回答更安全，一个回答更直接。SFT 只能模仿目标答案，不能表达偏好排序。


Full fine-tuning 的成本很大，全参数微调会更新模型所有参数。

这就是 PEFT 出现的背景：不改全部参数，只训练小模块。

## 5. PEFT 的基本想法

PEFT 是 parameter-efficient fine-tuning。

核心想法是：

> 冻结大部分预训练参数，只训练少量可学习参数，让模型适配新任务。

这样做的好处是：

- 训练显存低
- 存储成本低
- 多任务可以共用同一个 base model
- adapter 可以按需加载
- 实验迭代更快

但它也有代价。训练参数少意味着表达空间受限。如果任务和 base model 差距很大，PEFT 可能不如 full fine-tuning。

### LoRA：低秩更新

LoRA 的核心想法是：不要直接训练原始权重矩阵 $W$，而是冻结 $W$，只学习一个低秩更新：

$$
W' = W + \Delta W
$$

其中：

$$
\Delta W = BA
$$

如果 $W \in \mathbb{R}^{d \times k}$，那么可以设：

$$
B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d,k)
$$

训练参数从 $dk$ 变成 $r(d+k)$。

直觉上，LoRA 假设任务适配所需的权重变化不需要覆盖完整高维空间，而可以落在一个低秩子空间里。

这不是说模型能力只剩低秩。Base model 的原始参数仍然在，LoRA 只是学习一个附加方向。

**LoRA 插在哪里**

LoRA 常插在 Transformer 的线性层上，比如：

- attention 的 $W_Q, W_K, W_V, W_O$
- FFN 的 up projection / down projection

具体插哪里会影响效果和成本。

只插 attention 层，参数更少，但适配能力可能有限。

attention 和 FFN 都插，能力更强，但训练参数更多。

对很多任务来说，LoRA rank、target modules、learning rate、数据质量都会显著影响效果。

不要把 LoRA 理解成固定魔法。它是一个低成本适配工具，具体效果依赖配置。

### LoRA 为什么推理时方便

LoRA 的更新是：

$$
h = Wx + BAx
$$

训练时可以把 $BA$ 当成旁路。

部署时，如果只服务一个 adapter，可以把 $BA$ 合并回 $W$：

$$
W_{\mathrm{merged}} = W + BA
$$

这样推理时不一定增加额外延迟。

如果要服务多个 adapter，就可以保留 base model，按任务加载不同 LoRA 权重。

这也是 LoRA 在个人微调和多任务部署中很受欢迎的原因。

### Prefix Tuning：给模型加可学习前缀

Prefix tuning 不直接改模型权重，而是在每层 attention 里加入可学习 prefix。

可以理解成给模型提供一组虚拟的 key/value tokens，让后续 token 在 attention 时能读到这些可学习信息。

它和手写 prompt 不一样。手写 prompt 是离散文本，prefix tuning 学的是连续向量。

直觉上：

> Prefix tuning 不是告诉模型“请你这样回答”，而是在模型内部提供一组可学习上下文。

它的优势是训练参数少。

缺点是表达能力受限，而且会占用一定上下文或 attention 空间。

### Prompt Tuning 和 P-Tuning

Prompt tuning 学的是输入 embedding 前面的 soft prompt。

如果原输入是：

$$
[x_1, x_2, ..., x_n]
$$

prompt tuning 会在前面拼上一组可学习向量：

$$
[p_1, p_2, ..., p_m, x_1, x_2, ..., x_n]
$$

P-Tuning 则进一步用更复杂的方式生成或组织这些 continuous prompts，比如使用 MLP / LSTM 等 prompt encoder。

它们的共同点是：冻结模型主体，只训练 prompt-like 参数。

这类方法在早期很重要，但在当前 LLM 微调实践里，LoRA 通常更常见，因为 LoRA 对生成任务更灵活，工程生态也更成熟。

## 补充说明

**SFT 数据比方法更重要**

SFT经常被讨论成训练技术，但真正决定结果的通常是数据。

高质量 SFT 数据要满足：

- 指令清楚
- 回答准确
- 风格一致但不模板化
- 覆盖目标任务
- 没有大量幻觉
- reasoning trace 不只是装样子

如果数据质量差，LoRA 再省显存也没用。

如果数据太窄，模型会过拟合某种格式。

如果数据里充满“当然可以，下面是……”这种套话，模型会变得很像客服脚本。

所以 SFT 的核心不是把 loss 跑低，而是把模型行为拉到你想要的区域。


## 16. 小结

SFT 是后训练的入口。它让 base model 从“会续写”变成“会回答”。

PEFT 是低成本适配工具。LoRA、Prefix Tuning、P-Tuning 都是在减少可训练参数，但它们改模型的方式不同。


## 参考资料

- Wei et al., [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Li and Liang, [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- Liu et al., [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602)
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
