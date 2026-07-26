---
title: "后训练笔记 05：OPD，On-Policy Distillation"
date: 2025-11-29 22:00:00
updated: 2025-11-29 22:00:00
categories:
  - 后训练笔记
tags:
  - 后训练
  - OPD
  - 蒸馏
  - On-Policy
  - RLVR
  - 笔记
math: true
category_bar: true
---

这一篇整理 OPD，也就是 On-Policy Distillation。

OPD 是近期后训练里很值得关注的一条路线。它和 GRPO / RLVR 的目标有些相似：都想让模型在自己的生成分布上变强。但它给训练信号的方式不一样。

GRPO 通常是：

> student 自己生成 response，然后 verifier / reward model 给一个 scalar reward。

OPD 更像是：

> student 自己生成 response，然后 teacher 在这些 on-policy responses 上给 dense supervision。

所以 OPD 的关键词是：on-policy、teacher、distillation、dense signal。

## 普通 distillation 是什么

Knowledge distillation 的基本想法是：用强 teacher model 指导弱 student model。

最简单的 supervised distillation 是让 student 模仿 teacher 的输出：

$$
\mathcal{L}_{KD} = D_{KL}(p_{\mathrm{teacher}}(\cdot \mid x) || p_{\mathrm{student}}(\cdot \mid x))
$$

在 LLM 里，可以让 teacher 生成高质量 answers，然后 student 做 SFT。

这类方法的优点是简单。

但它通常是 off-policy 的：训练数据来自 teacher 或静态数据集，不一定来自 student 当前会生成的分布。

这会带来一个问题：student 在推理时遇到的状态，可能和训练时看到的不一样。

## 为什么需要 on-policy

语言模型生成是自回归的。

一旦前面某个 token 偏了，后面的上下文就变成 student 自己制造出来的分布。

如果训练时只看 teacher 的完美轨迹，student 没有学会在自己的错误轨迹上恢复。

这就是 distribution shift。

On-policy 的意思是：训练数据来自当前 student policy。

OPD 会让 student 先 rollout，生成它自己会生成的 responses，然后 teacher 在这些 responses 或中间 token 上提供监督。

直觉上：

> 不要只教学生看标准答案，也要在学生自己的解题过程上指出应该怎么走。

这点和 RL 类似。PPO/GRPO 也是在当前 policy 的 outputs 上学习。

区别在于，RL 用 reward；OPD 用 teacher distribution。

## RLVR 的 sparse reward 问题

以数学题为例。

GRPO 可能只知道最终答案是否正确：

$$
r(y) \in \{0,1\}
$$

如果答案错了，模型不知道哪一步错。

如果答案对了，模型也不知道中间哪些 token 真的关键。

这种 scalar reward 很稀疏。

OPD 希望用 teacher 提供更密集的信号。比如在 student rollout 的每个 token 位置，teacher 都可以给出概率分布或 log probability。

这样 student 不只是知道整段 response 好不好，而是能看到 teacher 在每个位置上更倾向什么。

## OPD 的基本流程

一个简化的 OPD 流程可以写成：

1. 从 prompt dataset 里取一批 prompts
2. student policy $\pi_\theta$ 生成 responses
3. teacher model $\pi_T$ 对这些 on-policy responses 计算 token-level supervision
4. student 用 distillation loss 更新

如果 student 生成：

$$
y = (y_1, y_2, ..., y_T)
$$

teacher 可以在每个位置给：

$$
\log \pi_T(y_t \mid x, y_{<t})
$$

student 则优化自己在这些 on-policy tokens 上的行为。

这和纯 SFT 不一样。SFT 的 response 通常来自数据集；OPD 的 response 来自 student 当前 policy。

## Reverse KL 的直觉

OPD 论文里经常会出现 KL objective。

Forward KL 和 reverse KL 的行为不一样。

Forward KL：

$$
D_{KL}(p_T || p_\theta)
$$

倾向于让 student 覆盖 teacher 的分布。

Reverse KL：

$$
D_{KL}(p_\theta || p_T)
$$

倾向于让 student 避开 teacher 认为低概率的区域。

在生成模型里，reverse KL 往往更 mode-seeking。它会鼓励 student 更集中地落到 teacher 支持的高质量区域。

这对后训练很有吸引力，因为 student 的 on-policy rollout 可能包含很多质量一般的 token。Teacher 的 dense signal 可以把它往更好的区域拉。

## OPD 和 GRPO 的核心差别

GRPO 的训练信号通常是 response-level reward。

比如：

$$
A_i = \frac{r_i - \mu}{\sigma}
$$

它告诉模型：这一组 responses 里哪个更好。

OPD 的训练信号来自 teacher 对 token distribution 的判断。

它告诉模型：在 student 自己走到的上下文里，teacher 会怎么分布概率。

所以两者的区别是：

- GRPO：用 scalar reward 做 outcome-driven learning
- OPD：用 teacher probabilities 做 dense behavior shaping

这也是 OPD 可能更高效的原因。Scalar reward 只在终点给信号，teacher distribution 可以在每一步给信号。

## OPD 和 SFT 的核心差别

SFT 是在静态数据上模仿。

OPD 是在 student 当前 rollout 上蒸馏。

如果 student 已经接近 teacher，SFT 可能足够。

如果 student 会生成很多偏离 teacher 的中间状态，OPD 更有价值，因为 teacher 可以在这些状态上给反馈。

这有点像老师批改学生自己的草稿，而不是只给学生看范文。

## Online OPD 的成本

标准 OPD 的一个问题是成本。

训练时需要 student rollout，然后 teacher 对这些 rollouts 计算 supervision。

如果 teacher 很大，就需要 live teacher inference server。

这会带来：

- 推理成本高
- 系统复杂
- 训练吞吐低
- teacher 服务和 student 训练要同步
- 多机异步时容易出现 stale samples

所以 OPD 虽然训练信号更密集，但系统上并不轻。

这也是 Lightning OPD、StableOPD、f-OPD 这些工作的背景。

## Lightning OPD：把 online teacher 变成 offline

Lightning OPD 关注一个问题：能不能不用 live teacher server？

它的思路是 offline on-policy distillation。

核心是预计算 teacher log-probabilities，然后训练时复用。

但这里有一个关键条件：teacher consistency。

如果 SFT 阶段和 OPD 阶段使用的 teacher 不一致，或者 teacher 对 student rollouts 的分布不一致，就可能引入不可消除的 gradient bias。

Lightning OPD 的一个重要结论是：只要保持 teacher consistency，offline OPD 可以接近 standard OPD 的 optimum，同时显著降低系统成本。

这对个人或学术实验很重要。因为 live teacher server 是很高的门槛。

## StableOPD：长度膨胀和重复崩溃

OPD 也有自己的失败模式。

一个问题是 length inflation。

Student on-policy rollouts 可能越来越长，训练目标又在这些 rollouts 上蒸馏，最后导致长而重复的输出变多。

如果 responses 被截断，训练数据里会出现大量 truncated trajectories。这会造成 biased gradient signal。

StableOPD 这类方法尝试用 reference-based divergence constraint 和 rollout mixture distillation 稳定训练。

直觉上，它在防止 student 被自己的坏 rollout 分布拖走。

这说明 OPD 虽然不用 scalar reward，但仍然有 policy drift 问题。

## f-OPD：异步训练里的 freshness

大规模训练通常不是完全同步的。

Rollout 生成、teacher 计算、student 更新可能是异步执行。

这会带来 stale sample 问题。

一个样本生成时对应的是旧 student policy，但真正训练时 student 已经更新过很多步。

Teacher supervision 的上下文也可能变旧。

f-OPD 把这种偏差拆成 rollout drift 和 supervision drift，并引入 freshness score 来判断样本是否还可靠。

直觉上：

> 样本不是越多越好，旧到一定程度的样本会偏离 on-policy 目标。

这对 long-horizon agentic tasks 尤其重要。任务越长，轨迹越复杂，staleness 的影响越大。

## OPD 适合什么任务

OPD 适合这些场景：

- 有强 teacher model
- Student 需要在自己的生成分布上学习
- Sparse reward 太弱
- 希望 token-level dense supervision
- 任务轨迹较长，单一 scalar reward 不够

比如数学 reasoning、代码生成、视频定位、agentic tool use，都可能适合 OPD。

但 OPD 不是所有场景都合适。

如果没有强 teacher，OPD 的上限会受限。

如果 teacher 推理成本太高，系统负担很重。

如果 student rollout 质量太差，teacher supervision 可能长期在很差的状态上补救，训练效率不一定高。

## OPD 和 RL 能不能结合

可以。

OPD 和 RLVR 不是互斥关系。

一种可能是先用 OPD 做 dense behavior shaping，让 student 接近 teacher 的高质量 reasoning 分布。

再用 RLVR 针对最终正确率做强化。

另一种可能是在 RL 过程中加入 teacher regularization，防止模型为了 reward 走向奇怪分布。

也可以把 OPD 看成一种替代 sparse reward 的中间路线：它不像 SFT 那么离线，也不像 RL 那么依赖 scalar reward。

## OPD 的风险

第一，teacher bias。

Student 会继承 teacher 的偏好和错误。

第二，teacher ceiling。

如果目标任务需要超过 teacher 的能力，单纯 OPD 很难突破 teacher。

第三，计算成本。

Teacher log-probs 对大模型来说很贵。

第四，length / repetition instability。

On-policy rollouts 可能产生越来越长、越来越重复的样本。

第五，staleness。

异步训练时，rollout 和 supervision 可能过期。

所以 OPD 不是“比 RL 更简单”的万能方法。它只是把问题从 reward design 转移到 teacher supervision 和系统效率上。

## 读 OPD 论文时看什么

我会看这些问题：

- Teacher model 是什么
- Student model 是什么
- Rollouts 来自当前 student 还是旧 policy
- Teacher supervision 是 token-level 还是 response-level
- 用 forward KL 还是 reverse KL
- 是否需要 live teacher server
- 是否 offline 预计算 teacher log-probs
- 如何处理 length inflation
- 如何处理 stale samples
- 是否和 GRPO / PPO 在同等 compute 下比较
- 推理时是否真的提升 greedy performance

尤其要注意 compute。OPD 如果性能提升但 teacher 成本巨大，实际价值要重新评估。

## 小结

OPD 可以理解成一条介于 SFT 和 RLVR 之间的后训练路线。

SFT 在静态数据上模仿。

GRPO 在 on-policy samples 上用 scalar reward 学。

OPD 在 on-policy samples 上用 teacher dense supervision 学。

我的理解是：

> OPD 的价值在于，它保留了 on-policy 学习的分布匹配，又用 teacher 分布缓解了 sparse reward 的问题。

但它的难点也很明确：teacher 成本、teacher consistency、长度膨胀、异步 staleness。

如果后面 reasoning / coding / agentic post-training 继续发展，OPD 这条线很可能会越来越重要。

## 参考资料

- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- Wu et al., [Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation](https://arxiv.org/abs/2604.13010)
- Luo et al., [Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models](https://arxiv.org/abs/2604.08527)
- Chen et al., [f-OPD: Stabilizing Long-Horizon On-Policy Distillation with Freshness-Aware Control](https://arxiv.org/abs/2605.17862)
- Li et al., [Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation](https://arxiv.org/abs/2602.02994)
