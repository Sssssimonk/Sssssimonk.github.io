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

> ”三思而后行“ 这是机器人先祖给AI的教诲

简单的说，告诉模型"Let's think step by step"效果意外的好了很多，然后就开始卷模型的reasoning了。

本篇从 CoT 开始，逐步过渡到 DeepSeek-R1 这类模型。





## 普通 next token prediction 的限制

语言模型的基本目标是预测下一个 token：

$$
\max_\theta \sum_t \log p_\theta(x_t \mid x_{<t})
$$

这个目标很强，能学到语法、知识、代码模式、常识关系。但复杂推理不只是“最像下一个词”。很多题需要中间状态：

- 先拆条件
- 再选择公式
- 再计算
- 再检查结果

如果模型直接从问题跳到答案，中间任何一步错了都很难纠正。所以 reasoning 的一个核心思路是：不要急着输出最终答案，先生成中间推理过程。

## Chain-of-thought 的作用

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

给模型更多计算轨迹，增强模型的上下文细节，让模型在输出空间里保留中间状态，最后的结果就能变好。

## CoT 为什么会提升能力

我觉得可以从三个角度理解。

第一，token budget 变多了。

模型不是在一个 token 或一句话里压出答案，而是有更多 token 做中间计算。

第二，中间步骤提供了隐式 scratchpad。

模型可以把已经得到的结论写下来，增强了后续正确答案的分布。

第三，训练分布更匹配。

如果模型在训练或示例中见过大量“逐步解题”的文本，它就会学到这种生成模式。

但 CoT 也有问题。模型写出来的推理不一定忠实。有时它只是生成一个看起来合理的解释，而真正导致答案的内部机制未必和文本推理一致。

所以 CoT 提升表现，不等于 CoT 完全可解释。

## Test-time compute：推理时多花计算

Reasoning model 的一个核心趋势是 test-time compute。

传统模型更强调训练阶段投入：预训练更多数据、更大参数。

Reasoning model 则强调推理阶段也可以多花计算。

比如：

- 生成更长 reasoning trace
- 采样多个解法再选择
- 自我检查
- 用 verifier 打分
- 分解问题再逐步求解

最简单的 self-consistency 会采样多条 reasoning trace，再对最终答案做投票：

```python
from collections import Counter

def self_consistency(model, prompt, num_samples=8):
    candidates = []

    for _ in range(num_samples):
        response = model.generate(
            prompt,
            temperature=0.8,
            do_sample=True,
        )
        answer = extract_final_answer(response)
        candidates.append((answer, response))

    answer_counts = Counter(answer for answer, _ in candidates)
    best_answer, _ = answer_counts.most_common(1)[0]

    # 返回多数答案，以及产生该答案的一条完整 trace
    best_trace = next(
        response for answer, response in candidates
        if answer == best_answer
    )
    return best_answer, best_trace
```

如果有单独的 verifier，也可以不做多数投票，而是对每条 response 打分后选择最高分。这两种方法都用更多推理计算换取更高的单题成功率。

这背后的想法是：对于复杂任务，生成一个短答案太便宜，也太冒险。

如果一道数学题需要 20 步推导，模型只生成 2 行答案，省下的 token 可能就是错误来源。

## Thinking mode 和 non-thinking mode

Qwen3 这类模型提出 thinking mode 和 non-thinking mode 的统一框架。

按照前大模型负责人lin junyang所说，他们本来的目的是为了让模型实现self-controlled thinking。也就是模型自己判断所需要的thinking长度，不过最后效果好像还是达不到。所以仍然只能显式的控制了。


## DeepSeek-R1-Zero：只靠 RL 能出现什么

DeepSeek-R1 里最有意思的部分是 R1-Zero。

R1-Zero 不先做 supervised fine-tuning，而是直接对 base model 做大规模 RL，目标是增强 reasoning 能力。

报告里提到，模型在 RL 过程中自然出现了一些推理行为，比如更长的思考、更强的自我验证。

这个结果很重要，因为它说明 reasoning 行为不一定完全依赖人工标注的 CoT 数据，也可以通过可验证奖励被强化出来。

但 R1-Zero 也有问题，比如可读性差、语言混杂、输出格式不稳定。

这说明“能推理”和“能以人类喜欢的方式推理”不是一回事。

## DeepSeek-R1：cold-start + multi-stage RL

DeepSeek-R1 在 R1-Zero 的基础上加入 cold-start data。

也就是先给模型一些高质量、可读性更好的推理样本，让模型有一个更好的起点，然后再做 RL。

后面再经过多阶段训练，把 reasoning 能力、通用问答能力、可读性等拉到更平衡的状态。

这体现了一个很重要的训练思路：

> RL 可以强化能力，但如果起点太乱，最终行为可能不好控制。

SFT 提供格式和初始行为，RL 提供目标导向的能力强化。

两者不是互斥，而是互补。

所以普遍认为结论为：进行一些高质量的sft，让模型学会基础的指令遵从。然后再用RL探索思维和推理的能力效果最好。

## RLVR为什么关键

Reasoning RL 最适合数学、代码这类任务，因为它们容易验证。

数学题可以看最终答案对不对。

代码题可以跑测试。

这类 reward 比“这个回答好不好”更明确。

如果 reward 很模糊，RL 容易学到投机行为。比如生成看起来更自信、更长、更像标准答案的文本，但不一定更正确。

所以 reasoning model 的突破，很大程度来自可验证任务的 reward 设计。

这也解释了为什么数学和代码经常是 reasoning model 的核心 benchmark。


**注意点：Reasoning trace 是否可信**

模型写出的 reasoning trace 可以帮助人检查，但不能完全等同于模型内部真实推理。

有时模型会先在内部形成答案，再生成一段看似合理的解释。

有时它会在文本推理中走错，但最后答案碰巧对。有时它的中间推理看起来很漂亮，但关键假设是错的。



## 参考资料

- Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- Wang et al., [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
