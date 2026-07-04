---
title: "现代大模型笔记 04：Reasoning Model，从 CoT 到 DeepSeek-R1"
date: 2025-09-27 21:30:00
updated: 2025-09-27 21:30:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - 大模型
  - Reasoning
  - CoT
  - DeepSeek-R1
  - 强化学习
  - 笔记
math: true
category_bar: true
---

这一篇整理 reasoning model。

现在很多模型会区分 thinking mode 和 non-thinking mode，也会强调数学、代码、复杂推理能力。表面上看，这是“模型回答前多想一会”。但如果只这么理解，就太浅了。

Reasoning model 的变化至少包括三层：

- 推理时愿意花更多 token 和计算
- 训练时用更强的推理数据和偏好信号
- 后训练阶段用 RL 等方法强化可验证任务上的表现

所以这篇从 CoT 开始，逐步过渡到 DeepSeek-R1 这类模型。

## 1. 普通 next token prediction 的限制

语言模型的基本目标是预测下一个 token：

$$
\max_\theta \sum_t \log p_\theta(x_t \mid x_{<t})
$$

这个目标很强，能学到语法、知识、代码模式、常识关系。

但复杂推理不只是“最像下一个词”。很多题需要中间状态：

- 先拆条件
- 再选择公式
- 再计算
- 再检查结果

如果模型直接从问题跳到答案，中间任何一步错了都很难纠正。

所以 reasoning 的一个核心思路是：不要急着输出最终答案，先生成中间推理过程。

## 2. Chain-of-thought 的作用

Chain-of-thought（CoT）最早可以理解成一种 prompting 方法。

给模型几个带中间步骤的例子，模型在新问题上也更倾向于写出中间步骤。

比如简单问题：

> 小明有 3 个苹果，又买了 5 个，吃掉 2 个，还剩几个？

直接回答也能做。

但如果问题变成多条件、多变量、多约束，直接回答容易跳步。

CoT 的作用是把一次复杂映射拆成多步局部映射：

$$
\text{problem} \rightarrow \text{intermediate steps} \rightarrow \text{answer}
$$

这不是魔法。它更像给模型更多计算轨迹，让模型在输出空间里保留中间状态。

## 3. CoT 为什么会提升能力

我觉得可以从三个角度理解。

第一，token budget 变多了。

模型不是在一个 token 或一句话里压出答案，而是有更多 token 做中间计算。

第二，中间步骤提供了隐式 scratchpad。

模型可以把已经得到的结论写下来，后面继续引用。

第三，训练分布更匹配。

如果模型在训练或示例中见过大量“逐步解题”的文本，它就会学到这种生成模式。

但 CoT 也有问题。模型写出来的推理不一定忠实。有时它只是生成一个看起来合理的解释，而真正导致答案的内部机制未必和文本推理一致。

所以 CoT 提升表现，不等于 CoT 完全可解释。

## 4. Test-time compute：推理时多花计算

Reasoning model 的一个核心趋势是 test-time compute。

传统模型更强调训练阶段投入：预训练更多数据、更大参数。

Reasoning model 则强调推理阶段也可以多花计算。

比如：

- 生成更长 reasoning trace
- 采样多个解法再选择
- 自我检查
- 用 verifier 打分
- 分解问题再逐步求解

这背后的想法是：对于复杂任务，生成一个短答案太便宜，也太冒险。

如果一道数学题需要 20 步推导，模型只生成 2 行答案，省下的 token 可能就是错误来源。

## 5. Thinking mode 和 non-thinking mode

Qwen3 这类模型提出 thinking mode 和 non-thinking mode 的统一框架。

这很实际。

不是所有问题都需要长推理。

问“北京今天几点”这种问题，如果模型长篇 reasoning，反而浪费时间。

但问一道复杂证明题、代码 debug、实验设计，就需要更多推理 token。

所以 thinking mode 的关键不是“永远思考”，而是根据任务复杂度分配计算。

这和人也像。简单问题直接答，复杂问题先列步骤。

从系统角度看，这叫 latency-quality trade-off。更长思考可能提高质量，但一定增加延迟和成本。

## 6. DeepSeek-R1-Zero：只靠 RL 能出现什么

DeepSeek-R1 里最有意思的部分是 R1-Zero。

R1-Zero 不先做 supervised fine-tuning，而是直接对 base model 做大规模 RL，目标是增强 reasoning 能力。

报告里提到，模型在 RL 过程中自然出现了一些推理行为，比如更长的思考、更强的自我验证。

这个结果很重要，因为它说明 reasoning 行为不一定完全依赖人工标注的 CoT 数据，也可以通过可验证奖励被强化出来。

但 R1-Zero 也有问题，比如可读性差、语言混杂、输出格式不稳定。

这说明“能推理”和“能以人类喜欢的方式推理”不是一回事。

## 7. DeepSeek-R1：cold-start + multi-stage RL

DeepSeek-R1 在 R1-Zero 的基础上加入 cold-start data。

也就是先给模型一些高质量、可读性更好的推理样本，让模型有一个更好的起点，然后再做 RL。

后面再经过多阶段训练，把 reasoning 能力、通用问答能力、可读性等拉到更平衡的状态。

这体现了一个很重要的训练思路：

> RL 可以强化能力，但如果起点太乱，最终行为可能不好控制。

SFT 提供格式和初始行为，RL 提供目标导向的能力强化。

两者不是互斥，而是互补。

## 8. 可验证奖励为什么关键

Reasoning RL 最适合数学、代码这类任务，因为它们容易验证。

数学题可以看最终答案对不对。

代码题可以跑测试。

这类 reward 比“这个回答好不好”更明确。

如果 reward 很模糊，RL 容易学到投机行为。比如生成看起来更自信、更长、更像标准答案的文本，但不一定更正确。

所以 reasoning model 的突破，很大程度来自可验证任务的 reward 设计。

这也解释了为什么数学和代码经常是 reasoning model 的核心 benchmark。

## 9. Distillation：小模型学 reasoning

DeepSeek-R1 还强调 distillation，把大 reasoning model 的能力蒸馏到较小模型上。

蒸馏的直觉是：大模型生成高质量 reasoning data，小模型用这些数据做 supervised training。

这说明一个现象：

> reasoning 能力的一部分可以通过行为模仿迁移。

但蒸馏也有上限。

小模型可以学到大模型的解题风格和部分策略，但参数容量、基础知识、搜索深度都有限。它不可能在所有场景都复制大模型能力。

所以蒸馏模型适合低成本部署，但不能简单等同于原始 teacher model。

## 10. Reasoning 不是更长越好

一个常见误区是：thinking token 越多，答案越好。

实际不是。

过长 reasoning 可能带来：

- 延迟变高
- 成本变高
- 中间步骤引入错误
- 模型绕远路
- 简单问题过度分析

所以 thinking budget 很重要。

理想状态是：简单任务短思考，复杂任务长思考；模型知道什么时候该停。

这也是 Qwen3 把 thinking budget 作为机制来讲的原因。

## 11. Reasoning trace 是否可信

模型写出的 reasoning trace 可以帮助人检查，但不能完全等同于模型内部真实推理。

有时模型会先在内部形成答案，再生成一段看似合理的解释。

有时它会在文本推理中走错，但最后答案碰巧对。

有时它的中间推理看起来很漂亮，但关键假设是错的。

所以我更愿意把 reasoning trace 看成一种可检查的输出过程，而不是完整透明的内部机制。

它有用，但不能盲信。

## 12. 读 reasoning model 报告时看什么

我会看这些问题：

- Base model 是什么
- 有没有 cold-start SFT
- RL reward 怎么设计
- 任务是否可验证
- 是否使用 rejection sampling 或 verifier
- thinking token 平均多长
- 是否支持非 thinking 模式
- benchmark 提升来自训练、采样，还是更长输出
- 蒸馏模型和原模型差距多大

尤其要看成本。一个 reasoning model 如果每题多花 10 倍 token，benchmark 提升就要放在成本背景下理解。

## 13. 小结

Reasoning model 的本质不是“模型突然会思考”，而是：

> 训练目标、数据构造、RL reward 和推理阶段计算分配共同改变了模型解决复杂问题的方式。

CoT 给模型 scratchpad。

Test-time compute 让模型推理时多花计算。

RL 在可验证任务上强化正确行为。

Distillation 把昂贵推理模型的行为迁移到小模型。

这几件事合起来，才是现代 reasoning model 的主线。

## 参考资料

- Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- Wang et al., [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
