---
title: "现代大模型笔记 05：Post-training，SFT、DPO、GRPO 到 RL 推理"
date: 2025-10-04 20:00:00
updated: 2025-10-04 20:00:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - 大模型
  - Post-training
  - SFT
  - DPO
  - GRPO
  - RLHF
  - 笔记
math: true
category_bar: true
---

这一篇整理 post-training。

Pretraining 让模型学会语言、知识和模式，但一个 pretrained base model 通常不是好用的助手。它可能会续写问题、模仿网页文本、输出不稳定格式，也不一定遵循用户指令。

Post-training 的目标是把 base model 变成更符合使用需求的模型。

可以先用一句话区分：

> Pretraining 学“世界和语言的统计结构”，post-training 学“怎么作为一个模型产品来回答问题”。

这句话不完全严谨，但很有用。

## 1. Base model 为什么不够用

Base model 的训练目标通常是 next token prediction：

$$
\max_\theta \sum_t \log p_\theta(x_t \mid x_{<t})
$$

它看到的是大规模文本，目标是预测下一个 token。

所以当你问：

> 请解释一下 AdamW 和 Adam 的区别。

Base model 可能确实知道相关知识，但它未必会以你期待的方式回答。它可能继续补全一段网页，也可能生成多个问答样例，也可能格式混乱。

这不是知识缺失，而是行为没有对齐。

Post-training 就是在解决行为问题。

## 2. SFT：先学会怎么回答

SFT 是 supervised fine-tuning。

给模型一批指令和高质量回答，让它模仿这些回答。

训练目标仍然是 next token prediction，只不过数据变成了 instruction-response 格式：

$$
\max_\theta \log p_\theta(y \mid x)
$$

其中 $x$ 是用户指令，$y$ 是期望回答。

SFT 的作用很基础：

- 学会遵循指令
- 学会问答格式
- 学会拒答边界
- 学会输出更清晰的结构
- 学会一些特定任务风格

如果没有 SFT，模型可能有知识，但不好用。

## 3. SFT 的局限

SFT 本质上是模仿。

它的问题是：数据里是什么样，模型就往什么样靠。

如果 SFT 数据质量一般，模型会学到平庸回答。

如果 SFT 数据太模板化，模型会变得啰嗦、套话。

如果 SFT 数据覆盖不到复杂场景，模型遇到新任务还是不稳。

更关键的是，很多问题没有唯一标准答案。

比如两个回答都正确，但一个更简洁，一个更详细；一个更安全，一个更有帮助。SFT 很难表达这种偏好排序。

这就引出了 preference learning。

## 4. RLHF：从偏好里学习

RLHF 的经典流程大概是：

1. 先训练一个 SFT model
2. 对同一个 prompt 采样多个回答
3. 人类标注哪个回答更好
4. 训练 reward model
5. 用 PPO 等 RL 方法优化 policy

Reward model 学的是：

$$
r_\phi(x,y)
$$

也就是给定 prompt $x$ 和回答 $y$，判断这个回答有多好。

然后语言模型作为 policy，优化目标大概是让 reward 更高，同时不要偏离原模型太远。

通常会加 KL penalty：

$$
\max_\theta \mathbb{E}[r_\phi(x,y)] - \beta D_{KL}(\pi_\theta || \pi_{ref})
$$

这个 KL 项很重要。没有它，模型可能为了骗 reward model 走到奇怪分布。

## 5. PPO：经典但复杂

PPO 是 RLHF 里常见的算法。

它把语言模型看成 policy，生成回答就是 action sequence。

PPO 的优点是通用，能直接优化 reward。

但它也复杂：

- 需要 reward model
- 需要采样
- 需要 value model 或 advantage estimation
- 训练不稳定
- 超参数敏感
- 成本高

所以后来大家开始寻找更简单的 preference optimization 方法。

这里不能简单说 PPO 落后。它仍然很重要，尤其是在更复杂的 RL 场景里。但对于很多偏好对齐任务，DPO 这类方法更容易训练。

## 6. DPO：把偏好优化变简单

DPO 的核心是：不显式训练 reward model，也不跑完整 RL，而是直接用偏好数据优化模型。

偏好数据通常是三元组：prompt、chosen response、rejected response。

DPO 让模型提高 chosen 的相对概率，降低 rejected 的相对概率，同时参考一个 reference model。

它的损失可以写成类似：

$$
\mathcal{L}_{DPO} = -\log \sigma \left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)
$$

其中 $y_w$ 是 preferred response，$y_l$ 是 rejected response。

直觉上，DPO 在说：

> 相比 reference model，新模型应该更偏向好回答，而不是差回答。

DPO 的优点是稳定、简单、成本低。

但它依赖偏好数据质量，也不一定适合所有需要复杂探索的任务。

## 7. GRPO：为什么 reasoning 里会出现它

GRPO 是 DeepSeekMath 和 DeepSeek-R1 里提到的重要方法。

它可以看成 PPO 的一种变体，核心是省掉单独的 value model，用一组采样回答的相对奖励来估计 advantage。

对同一个问题，模型生成一组回答：

$$
\{y_1, y_2, ..., y_G\}
$$

每个回答得到 reward。然后用组内平均和标准差做归一化，判断某个回答相对这一组是好还是坏。

直觉上：

> 不一定要知道一个回答的绝对价值，只要知道它在同一组候选里相对更好还是更差。

这对数学、代码这类可验证任务很自然。因为 reward 可以来自答案是否正确、测试是否通过。

GRPO 的优势是减少 value model 带来的额外显存和训练复杂度。

但它仍然是 RL，不是免费的。采样多个回答、计算 reward、控制 KL、保持训练稳定，都是成本。

## 8. SFT、DPO、GRPO 的位置

可以粗略这样理解：

- SFT：教模型怎么回答
- DPO：教模型在两个回答里偏向更好的那个
- GRPO/RL：让模型通过奖励信号强化解决问题的策略

SFT 更像 imitation。

DPO 更像 preference shaping。

RL 更像 outcome-driven optimization。

这三者不是互斥关系。很多现代模型会组合使用：先 SFT，再偏好优化，再针对 reasoning/code 做 RL。

## 9. Post-training 改的是能力还是行为

这是一个容易混的点。

Post-training 有时主要改行为，比如让模型更礼貌、更清晰、更遵循指令。

但在 reasoning、coding 任务里，post-training 也可能显著提升能力。

原因是这些任务的“能力”很依赖解题策略。

Base model 可能已经有知识，但不会稳定地分解问题、检查答案、使用工具、运行测试。

通过 post-training，模型学到更有效的行为策略，于是表现看起来像能力提升。

所以能力和行为在 LLM 里不是完全分开的。

## 10. Reward hacking 和过优化

只要有 reward，就有 reward hacking 风险。

如果 reward model 偏好长回答，模型可能变啰嗦。

如果 reward 偏好自信语气，模型可能更会胡说。

如果数学 reward 只看最终答案，模型可能生成错误推理但碰巧答案对。

如果代码 reward 只看公开测试，模型可能过拟合测试。

所以 post-training 的关键不是“把 reward 拉满”，而是设计和验证 reward 是否真的代表目标。

这也是为什么现代模型训练要大量 eval、红队、安全测试、人工审查。

## 11. Post-training 和数据质量

Post-training 对数据质量非常敏感。

一批高质量 SFT 数据可能比十倍低质量数据更有用。

偏好数据也一样。标注者是否一致、偏好标准是否清晰、chosen 和 rejected 差异是否有意义，都会影响训练。

对 reasoning model 来说，高质量 reasoning traces 更关键。

如果训练数据里充满表面步骤，模型会学会“写步骤”，但不一定学会“做推理”。

所以数据构造不是辅助工作，而是 post-training 的核心。

## 12. 为什么同一个 base model 会有不同版本

一个 base model 可以派生出很多版本：

- instruct model
- chat model
- reasoning model
- code model
- tool-use model
- domain-specific model

它们底层知识可能相近，但 post-training 目标不同。

比如 code model 会更强调代码数据、测试反馈、仓库级任务。

Reasoning model 会更强调数学、逻辑、可验证 reward、长 reasoning trace。

Chat model 会更强调安全、有帮助、对话体验。

所以看到模型名时，要区分 base 和 post-trained variant。只比较参数量没有意义。

## 13. 读 post-training 报告时看什么

我会看这些问题：

- Base model 是什么
- SFT 数据规模和来源是什么
- 是否使用偏好数据
- 偏好优化用 PPO、DPO，还是其他方法
- RL reward 是人工 reward model，还是规则验证
- 是否针对 reasoning/code/agent 做专门训练
- 是否蒸馏自更强模型
- 有没有控制 KL，reference model 是什么
- eval 是否覆盖安全、幻觉、长上下文、代码、数学

特别要警惕只展示 benchmark 分数但不讲训练细节的报告。Post-training 的细节往往决定模型实际行为。

## 14. 小结

Post-training 是现代大模型从“会续写”变成“能使用”的关键阶段。

SFT 解决基本指令跟随。

RLHF/PPO 让模型从偏好中优化，但复杂且不稳定。

DPO 把偏好学习变成更简单的目标。

GRPO 适合在可验证任务上做 group-based RL，尤其和 reasoning/coding 结合紧密。

我会把 post-training 看成一套行为塑形系统。它不只是让模型更礼貌，而是在很多任务上改变模型解决问题的方式。

## 参考资料

- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- GLM-4.5 Team, [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471)
