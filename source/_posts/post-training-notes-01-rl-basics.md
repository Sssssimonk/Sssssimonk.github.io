---
title: "后训练笔记 01：强化学习基础"
date: 2025-11-01 20:00:00
updated: 2025-11-01 20:00:00
categories:
  - 后训练笔记
tags:
  - 后训练
  - 强化学习
  - RLHF
  - RLVR
  - Policy Gradient
  - 笔记
math: true
category_bar: true
---

这一篇先整理后训练里会用到的强化学习基础。

大模型后训练里的 RL 和传统 RL 不太一样。传统 RL 常见场景是 agent 在环境里反复交互，比如机器人控制、游戏、推荐系统。LLM 后训练里的“环境”往往更窄：给一个 prompt，模型生成 response，然后由人类偏好、reward model、规则 verifier、测试用例或者答案匹配器给出反馈。

所以这里不用先背完整 RL 教材。先抓住一条主线：

> 后训练里的 RL，本质上是在让模型生成的回答更容易得到高 reward，同时又不能偏离原模型太远。

这句话里有三个关键词：policy、reward、reference。

## 1. 语言模型里的 policy 是什么

在 RL 里，policy 通常写成：

$$
\pi_\theta(a \mid s)
$$

它表示在状态 $s$ 下采取动作 $a$ 的概率。

放到语言模型里，可以这样对应：

- 状态（state）：当前 prompt 加上已经生成的 tokens
- 动作（action）：下一个 token
- policy：语言模型给下一个 token 的概率分布

如果把完整回答看成一个 token 序列：

$$
y = (y_1, y_2, ..., y_T)
$$

那么模型生成这个回答的概率是：

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{<t})
$$

其中 $x$ 是 prompt。

这也是为什么很多后训练算法会用 log probability：

$$
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})
$$

因为整段 response 的概率是 token 概率连乘，直接乘容易数值很小，取 log 后就变成求和。

## 2. Token-level action 和 response-level action

理论上，LLM 每一步生成 token 都是一个 action。

但很多 reward 是 response-level 的。比如数学题最后答案对不对、代码是否通过测试、回答是否更符合人类偏好，这些通常是整段回答生成完之后才知道。

这就带来一个 credit assignment 问题：

> 如果最终回答错了，到底是哪几个 token 造成的？

PPO 里会尝试用 value model 和 advantage estimation，把整段 reward 分摊到 token-level 更新上。

GRPO 里则常见做法是对同一个 prompt 采样多个 responses，再用组内 reward 相对大小得到 advantage。

DPO 更进一步，不显式做在线 RL，而是直接比较 chosen response 和 rejected response 的 log probability。

这些算法看起来差很多，但都绕不开同一个问题：训练信号怎样从“整段回答好不好”传回到模型参数。

## 3. Reward：后训练真正优化的目标

Reward 可以来自很多地方。

在人类偏好场景里，reward 可能来自 reward model：

$$
r_\phi(x, y)
$$

它输入 prompt 和 response，输出一个标量分数。

在 reasoning / coding 场景里，reward 经常是 verifiable reward：

- 数学题最终答案是否匹配
- 代码是否通过单元测试
- 格式是否满足要求
- 工具调用是否成功

这类方法常被叫做 RLVR，也就是 reinforcement learning from verifiable rewards。

RLVR 的优势是 reward 更客观，不需要每个样本都人工标注偏好。

但它也有明显局限：reward 很稀疏。一个数学题只有最终答案对/错，模型并不知道中间哪一步推理值得奖励。

这也是为什么 reasoning model 训练里经常会关心：

- 如何采样更多候选
- 如何过滤高质量 reasoning traces
- 如何避免只学会猜答案
- 如何让 reward 不只看最终答案

## 4. Return、value 和 advantage

在一般 RL 里，return 是从当前时刻往后能拿到的累计 reward：

$$
G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k
$$

LLM 后训练里，很多任务只有最终 reward，所以可以粗略理解为整段回答的得分。

Value function 估计的是某个状态下未来能拿到多少 reward：

$$
V^\pi(s) = \mathbb{E}_{\pi}[G_t \mid s_t=s]
$$

Advantage 衡量的是某个 action 比平均水平好多少：

$$
A_t = G_t - V^\pi(s_t)
$$

这个量很重要。

如果一个回答得分是 0.8，不一定说明它很值得加强。也许这个 prompt 很简单，随便生成都能 0.8。

如果另一个回答得分是 0.6，也不一定差。也许这个 prompt 很难，大多数回答都是 0.1。

Advantage 想表达的是：相对于当前状态的预期表现，这个 action / response 到底更好还是更差。

## 5. Policy gradient 的直觉

Policy gradient 的经典形式可以写成：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a \mid s) A(s,a)]
$$

放到 LLM 里，可以粗略理解成：

> 如果某个 response 的 advantage 为正，就提高它的生成概率；如果 advantage 为负，就降低它的生成概率。

但这里有两个细节。

第一，提高的是整段 response 里 tokens 的 log probability，不是直接给最终答案加分。

第二，更新不能太大。语言模型是高维分布，一次更新如果把概率分布拉得太远，可能会破坏原来的语言能力。

这就引出 KL constraint。

## 6. KL constraint：为什么需要 reference model

后训练里通常会保留一个 reference model，记作 $\pi_{\mathrm{ref}}$。

它通常是 SFT 后的模型，或者训练开始前的 policy snapshot。

更新 policy 时，会惩罚当前模型和 reference model 的差异：

$$
D_{KL}(\pi_\theta || \pi_{\mathrm{ref}})
$$

直觉上，KL constraint 在说：

> 可以朝高 reward 方向走，但不要走到完全不像原来的模型。

没有 KL 约束会有什么问题？

模型可能为了骗 reward model 生成奇怪模式。比如 reward model 偏好长回答，模型就越来越啰嗦；reward model 偏好自信语气，模型就更容易胡说；verifier 只看最终答案，模型可能学会投机格式。

所以后训练不是简单最大化 reward，而是：

$$
\max_\theta \mathbb{E}[r(x,y)] - \beta D_{KL}(\pi_\theta || \pi_{\mathrm{ref}})
$$

这里 $\beta$ 控制 reward 优化和保持原模型分布之间的平衡。

## 7. On-policy 和 off-policy

On-policy 指训练数据来自当前 policy。

Off-policy 指训练数据来自旧 policy、其他模型，或者离线数据集。

LLM 后训练里，这个区别非常关键。

如果模型训练时看到的是自己当前会生成的回答，那训练分布和推理分布更接近。这是 on-policy 的优势。

但 on-policy 代价高，因为训练过程中要不断采样新 responses。

Off-policy 成本低，可以直接用已有数据训练。但如果数据分布和当前模型差很多，模型学到的更新可能不稳定。

这就是为什么 PPO / GRPO / OPD 都强调 on-policy 或接近 on-policy。它们希望模型在自己的生成分布上学习，而不是只模仿静态数据。

## 8. RLHF、RLVR 和 preference optimization

这几个词容易混。

RLHF 通常指 reinforcement learning from human feedback。经典流程是：人类偏好数据 -> reward model -> PPO 优化 policy。

RLVR 是 reinforcement learning from verifiable rewards。它不一定需要人类偏好，而是用规则、答案、测试来给 reward。

Preference optimization 更宽，包括 DPO、IPO、KTO 等直接从偏好或二元反馈训练的算法。它们不一定显式训练 reward model，也不一定跑在线 RL。

可以这样记：

- RLHF：人类偏好先变 reward model，再做 RL
- RLVR：可验证任务直接给 reward
- DPO/KTO：从偏好数据直接优化 policy，不显式训练 reward model
- OPD：用 teacher 的 dense supervision 替代 sparse reward

这些路线都属于后训练，但训练信号来源不一样。

## 9. 为什么 reasoning model 更依赖 RL

SFT 可以让模型学会解题格式，但它本质上是模仿。

如果 SFT 数据里有高质量 reasoning traces，模型会学到“像这样推理”。但它没有直接被奖励“答案正确”。

RLVR 可以直接优化最终正确率。比如同一个数学题，模型生成多个答案，只有正确答案拿到 reward。经过训练，模型会提高能得到正确答案的生成轨迹概率。

这也是 DeepSeek-R1 这类 reasoning model 重要的地方：它不是只靠人工写好的 CoT，而是通过可验证 reward 强化推理行为。

不过这也带来风险。

如果 reward 只看最终答案，模型可能学到短路策略。比如中间推理不可靠，但答案格式正确；或者生成更长 reasoning 来提高偶然命中率。

所以 reasoning RL 的关键不是“用了 RL”，而是 reward、采样、过滤、KL、长度控制一起设计。

## 10. 后训练里的几个常见失败模式

第一，reward hacking。

模型学会利用 reward 的漏洞，而不是真的变好。

第二，mode collapse。

模型输出变得单一，缺少多样性。

第三，length inflation。

模型生成越来越长，因为长回答更容易看起来努力，或者更容易覆盖答案。

第四，over-optimization。

训练集 reward 越来越高，但真实评估变差。

第五，distribution drift。

policy 离 reference model 太远，语言能力或安全边界受损。

这些问题后面讲 PPO、GRPO、OPD 时都会反复出现。

## 11. 读后训练论文时先看什么

我会先看这几个问题：

- 训练信号来自哪里：人类偏好、reward model、verifier、teacher model？
- 数据是 on-policy 还是 off-policy？
- reward 是 token-level、step-level，还是 response-level？
- 有没有 reference model 和 KL constraint？
- 是否需要 value model？
- 每个 prompt 采样多少 responses？
- 如何处理全对/全错样本？
- 如何控制长度膨胀？
- benchmark 提升是否只是来自更多采样或更长输出？

这些问题比算法名字本身更重要。

## 12. 小结

后训练里的 RL 可以压缩成一句话：

> 让模型在自己的生成分布上尝试回答，用 reward 或偏好信号判断好坏，再提高好回答的概率，同时用 reference model 限制分布漂移。

PPO、GRPO、DPO、KTO、OPD 的区别，主要在于训练信号怎么来、advantage 怎么算、是否需要在线采样、是否需要额外模型。

所以这一章先把 RL 语言打通。后面再看具体算法时，就不会只是在背缩写。

## 参考资料

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
