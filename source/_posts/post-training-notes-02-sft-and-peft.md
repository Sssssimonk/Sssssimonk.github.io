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

PEFT 则是另一类问题：模型太大，全参数微调成本太高，所以只训练很少一部分参数，让模型适配新任务。

这篇把两件事放一起看：

> SFT 解决“训练什么行为”，PEFT 解决“用多低成本训练”。

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

## 2. SFT 具体学到了什么

SFT 不只是“灌知识”。

它主要学这些行为：

- 遵循指令
- 按对话格式回答
- 输出更清晰的结构
- 遇到不确定问题时更谨慎
- 学会某些任务格式，比如代码解释、数学推理、摘要
- 初步形成 reasoning trace 的风格

比如 base model 可能知道 AdamW，但 SFT model 更可能用分点方式解释 Adam 和 AdamW 的区别。

这也是为什么 chat model、instruct model、reasoning model 往往都是从同一个 base model 派生出来，但行为差很多。

## 3. SFT 的局限

SFT 本质上是 imitation learning。

数据是什么样，模型就往什么样学。

如果数据里回答很模板化，模型也会模板化。

如果数据里推理过程只是表面解释，模型会学会写“看起来像推理”的文本，但不一定真的更会解题。

如果数据里有错误，模型会跟着学。

更关键的是，SFT 不知道两个正确回答哪个更好。

比如一个回答很简洁，一个回答更完整；一个回答更安全，一个回答更直接。SFT 只能模仿目标答案，不能表达偏好排序。

所以 SFT 通常只是后训练的起点，后面还需要 DPO、PPO、GRPO、KTO 或 OPD。

## 4. Full fine-tuning 的成本

全参数微调会更新模型所有参数。

对于小模型，这很直接。对于几十亿、几百亿参数模型，成本很高：

- optimizer states 占显存
- gradient 占显存
- checkpoint 很大
- 多个任务需要存多份模型
- 训练和部署都更重

Adam 优化器通常需要为每个参数维护一阶、二阶动量，所以显存压力不只是模型参数本身。

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

## 6. LoRA：低秩更新

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

## 7. LoRA 插在哪里

LoRA 常插在 Transformer 的线性层上，比如：

- attention 的 $W_Q, W_K, W_V, W_O$
- FFN 的 up projection / down projection

具体插哪里会影响效果和成本。

只插 attention 层，参数更少，但适配能力可能有限。

attention 和 FFN 都插，能力更强，但训练参数更多。

对很多任务来说，LoRA rank、target modules、learning rate、数据质量都会显著影响效果。

不要把 LoRA 理解成固定魔法。它是一个低成本适配工具，具体效果依赖配置。

## 8. LoRA 为什么推理时方便

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

## 9. Prefix Tuning：给模型加可学习前缀

Prefix tuning 不直接改模型权重，而是在每层 attention 里加入可学习 prefix。

可以理解成给模型提供一组虚拟的 key/value tokens，让后续 token 在 attention 时能读到这些可学习信息。

它和手写 prompt 不一样。手写 prompt 是离散文本，prefix tuning 学的是连续向量。

直觉上：

> Prefix tuning 不是告诉模型“请你这样回答”，而是在模型内部提供一组可学习上下文。

它的优势是训练参数少。

缺点是表达能力受限，而且会占用一定上下文或 attention 空间。

## 10. Prompt Tuning 和 P-Tuning

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

## 11. LoRA、Prefix Tuning、P-Tuning 怎么选

可以粗略这样看。

LoRA 适合需要改模型行为、任务风格、领域适配的场景。它改的是线性层权重的低秩增量，表达能力相对强。

Prefix tuning 适合希望用少量参数控制模型行为，同时不动主体权重的场景。

Prompt tuning / P-Tuning 更像学习软提示，参数少，但对大模型和复杂生成任务不一定总是最强。

如果是个人博客里的实践建议，我会优先写：

- 小成本实验：LoRA / QLoRA
- 需要多任务 adapter：LoRA
- 只想做轻量控制：prefix / prompt tuning
- 追求最好效果且资源够：full fine-tuning 或更大规模 post-training

## 12. QLoRA：进一步省显存

QLoRA 是 LoRA 的重要实践扩展。

它把 base model 量化，比如 4-bit 存储，同时仍然用 LoRA adapter 训练。

关键点是：base model 不参与梯度更新，只需要在前向/反向中提供计算；可训练的是 LoRA 参数。

这样可以在更小显存上微调较大模型。

但量化也可能带来数值误差。QLoRA 的效果依赖量化格式、计算精度、rank、数据质量。

## 13. SFT 数据比方法更重要

SFT 和 PEFT 经常被讨论成训练技术，但真正决定结果的通常是数据。

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

## 14. SFT 和后续 RL / DPO 的关系

SFT 给模型一个可控起点。

DPO / KTO 进一步从偏好信号中学习什么回答更好。

PPO / GRPO 进一步用 reward 或 verifier 优化结果。

OPD 则可以用 teacher 对 student 的 on-policy outputs 做 dense supervision。

如果 SFT 起点太差，后面 RL 可能要花很多成本纠正基础格式问题。

如果 SFT 起点太强但过度模板化，后面偏好优化可能很难恢复多样性。

所以 SFT 不是越多越好，而是要给后续训练留下空间。

## 15. 读 SFT / PEFT 论文时看什么

我会看这些问题：

- Base model 是什么
- SFT 数据来自哪里
- 数据规模和质量如何
- 是否包含 reasoning traces
- 是 full fine-tuning 还是 PEFT
- LoRA rank 是多少
- LoRA 插在哪些模块
- 是否使用 QLoRA
- 是否只评估 benchmark，还是也看真实对话质量
- 后面有没有接 DPO/RL/OPD

尤其要注意：很多报告里“用了 LoRA”不是主要贡献。LoRA 只是训练方式，真正重要的是数据、目标和评估。

## 16. 小结

SFT 是后训练的入口。它让 base model 从“会续写”变成“会回答”。

PEFT 是低成本适配工具。LoRA、Prefix Tuning、P-Tuning 都是在减少可训练参数，但它们改模型的方式不同。

我的理解是：

> SFT 决定模型学什么行为，PEFT 决定用多少参数去学这个行为。

后面讲 PPO、DPO、GRPO 时，要一直记住：这些算法通常不是从 raw base model 开始，而是在 SFT 模型基础上继续塑形。

## 参考资料

- Wei et al., [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Li and Liang, [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- Liu et al., [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602)
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
