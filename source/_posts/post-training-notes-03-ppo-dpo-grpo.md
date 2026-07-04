---
title: "后训练笔记 03：PPO、DPO 与 GRPO"
date: 2025-11-15 21:00:00
updated: 2025-11-15 21:00:00
categories:
  - 后训练笔记
tags:
  - 后训练
  - PPO
  - DPO
  - GRPO
  - RLHF
  - RLVR
  - 笔记
math: true
category_bar: true
---

这一篇整理 PPO、DPO 和 GRPO。

这三个方法经常一起出现，但它们解决的问题不完全一样。

PPO 是经典 RLHF 路线里的 policy optimization 方法。

DPO 是直接从偏好对里学习，不显式训练 reward model，也不跑在线 RL。

GRPO 是 reasoning model 里很常见的 RLVR 方法，用同一个 prompt 下的一组 responses 做相对比较，省掉 value model。

可以先抓住一句话：

> PPO、DPO、GRPO 的核心区别，不是公式长得像不像，而是训练信号从哪里来。

## 1. 三种训练信号

PPO 的训练信号来自 reward model 或 verifier，再加 value model 估计 advantage。

DPO 的训练信号来自 preference pair，也就是 chosen response 和 rejected response。

GRPO 的训练信号来自 group relative reward：同一个 prompt 采样多个 responses，看谁在组内更好。

这三者的结构可以粗略写成：

- PPO：prompt -> policy rollout -> reward/value/reference -> policy update
- DPO：prompt + chosen/rejected -> policy/reference log-ratio -> preference loss
- GRPO：prompt -> sample group -> rewards -> group advantage -> policy update

如果只看名字，很容易觉得都是“对齐算法”。但从工程角度看，它们需要的模型、数据、采样方式完全不同。

## 2. PPO：经典 RLHF 路线

PPO 原本是通用强化学习算法，后来被用在 RLHF 里。

在 LLM RLHF 中，通常会有四个模型角色：

- Policy model：正在训练的模型
- Reference model：冻结的参考模型，用来限制 policy drift
- Reward model：给 response 打分
- Value model：估计 baseline / value

流程大概是：

1. policy 根据 prompt 生成 response
2. reward model 给 response 打分
3. reference model 计算 KL penalty
4. value model 估计 baseline
5. 用 PPO 更新 policy

PPO 的目标不是简单让 reward 最大，而是在 reward 和分布约束之间平衡。

## 3. PPO clipping 的直觉

PPO 的核心是限制 policy 更新幅度。

定义 probability ratio：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

如果 $r_t(\theta)$ 太大，说明新 policy 比旧 policy 更倾向于采取这个 action。

PPO 会把 ratio clip 到一个范围里：

$$
\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)
$$

目标函数可以写成：

$$
\mathcal{L}_{PPO} = \mathbb{E}_t[\min(r_t(\theta)A_t, \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t)]
$$

直觉上，PPO 在说：

> 如果一个 action 好，可以提高它的概率，但不要一次提高太多；如果一个 action 差，可以降低它的概率，但也不要一次降太多。

语言模型是很脆弱的分布。更新过猛，模型可能学到 reward model 的漏洞，或者输出风格突然崩掉。

## 4. PPO 为什么复杂

PPO 的问题不是理论上不能用，而是系统上很重。

它通常需要：

- 在线采样 responses
- reward model 推理
- reference model 推理
- value model 训练或推理
- advantage estimation
- KL 控制
- 多轮 policy update

这对显存、吞吐、工程稳定性都不友好。

尤其是大模型场景下，policy 本身已经很大，再加 reward/value/reference，训练链路会非常重。

所以后面出现了很多“简化 PPO”的路线：DPO 直接避开 RL；GRPO 去掉 value model；OPD 用 teacher dense supervision 替代 sparse reward。

## 5. DPO：直接从偏好对学习

DPO 的出发点很漂亮。

经典 RLHF 里，人类偏好先训练 reward model，再用 PPO 优化 policy。

DPO 认为：如果 reward model 和最优 policy 之间存在对应关系，那能不能跳过 reward model，直接从偏好数据推导 policy loss？

DPO 数据通常是：

- prompt $x$
- preferred response $y_w$
- rejected response $y_l$

DPO 不需要在线采样。它直接用这些 preference pairs 训练 policy。

## 6. DPO 的 log-ratio

DPO 会比较 policy model 和 reference model 对 chosen/rejected 的相对偏好。

核心量是：

$$
\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
$$

如果这个值变大，说明相对于 reference，当前 policy 更偏向 chosen 而不是 rejected。

DPO loss 可以写成：

$$
\mathcal{L}_{DPO} = -\log \sigma \left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\right]\right)
$$

这个公式看起来复杂，其实直觉很简单：

> 好回答的概率应该相对变高，差回答的概率应该相对变低，但变化要以 reference model 为锚。

Reference model 在这里也很重要。它防止 policy 只是盲目拉高 chosen 的绝对概率，而是关注相对 reference 的偏好变化。

## 7. DPO 的优点和局限

DPO 的优点很明显：

- 不需要训练 reward model
- 不需要在线 RL rollout
- 不需要 value model
- 训练稳定性通常比 PPO 简单
- 工程实现更接近 supervised fine-tuning

但 DPO 也有局限。

第一，它依赖 preference pair。真实产品里经常只有 thumbs up / thumbs down，而不是成对比较。

第二，它是离线训练。数据分布如果和当前 policy 差很多，可能不如 on-policy 方法。

第三，它优化的是偏好排序，不一定直接优化可验证正确率。

第四，chosen/rejected 的质量很关键。如果 rejected 只是很差的回答，模型学到的边界可能很粗；如果 chosen 和 rejected 差异太小，训练信号也可能不稳定。

所以 DPO 很适合偏好对齐，但不是所有 post-training 问题的终点。

## 8. GRPO：从 PPO 到 group relative advantage

GRPO 是 Group Relative Policy Optimization。

它可以看成 PPO 在 reasoning / RLVR 场景下的一种简化。

PPO 需要 value model 来估计 baseline。GRPO 不单独训练 value model，而是对同一个 prompt 采样一组 responses，用组内 reward 的相对关系估计 advantage。

对一个 prompt $x$，采样 $G$ 个 responses：

$$
\{y_1, y_2, ..., y_G\}
$$

每个 response 得到 reward：

$$
\{r_1, r_2, ..., r_G\}
$$

然后用组内均值和标准差归一化：

$$
A_i = \frac{r_i - \mathrm{mean}(r_1,...,r_G)}{\mathrm{std}(r_1,...,r_G)}
$$

这就是 group relative 的意思。

## 9. GRPO 为什么适合 reasoning

Reasoning 任务里，同一个问题可以采样多个解法。

比如一道数学题，模型生成 8 个 responses，其中 3 个答案对，5 个答案错。

正确的 responses 得到更高 reward，错误的 responses 得到更低 reward。组内比较后，正确 responses 的 advantage 为正，错误 responses 的 advantage 为负。

这样就不需要 value model 去预测这个题本身有多难。组内样本直接给出了相对基线。

这对数学、代码这种可验证任务很自然：

- 数学题可以匹配答案
- 代码题可以跑测试
- 格式题可以规则检查

所以 GRPO 和 RLVR 很搭。

## 10. GRPO 的核心代价

GRPO 省掉了 value model，但它不是免费。

它需要对每个 prompt 采样多个 responses。

如果 group size 是 8，训练时生成成本就明显增加。

更麻烦的是，如果一组 responses 全对或全错，组内 reward 没有差异，advantage 可能接近 0。

这意味着没有有效学习信号。

比如某个题太简单，8 个 response 全对；或者太难，8 个 response 全错。它们都不能很好地区分哪些轨迹更值得加强。

这也是后面 GRPO 变体要解决的问题之一：怎样让采样、reward、advantage 更有效。

## 11. PPO、DPO、GRPO 的关键差别

可以从四个维度比较。

第一，是否在线采样。

PPO 和 GRPO 通常需要在线采样；DPO 不需要。

第二，是否需要 reward model。

PPO 通常需要 reward model；GRPO 可以用 verifier 或 reward model；DPO 不显式需要 reward model。

第三，是否需要 value model。

PPO 需要；DPO 不需要；GRPO 不需要。

第四，训练信号粒度。

PPO 用 reward + value 得到 advantage。

DPO 用 chosen/rejected 的偏好差。

GRPO 用同一 prompt 下多个 responses 的相对 reward。

这几个差别决定了工程成本和适用场景。

## 12. 三者分别适合什么

PPO 适合经典 RLHF 场景：已经有 reward model，希望在线优化 policy，并且有足够工程资源。

DPO 适合偏好数据充足的场景：有 chosen/rejected pairs，希望稳定、低成本地做对齐。

GRPO 适合可验证 reasoning 场景：数学、代码、逻辑推理，能自动给 reward，并且愿意为每个 prompt 采样多个 responses。

如果从个人学习角度看，我会这样排序：

先理解 DPO，因为它最简洁，能帮助理解 reference model 和 log-ratio。

再理解 PPO，因为它是 RLHF 的经典路线。

最后理解 GRPO，因为它建立在 RL 直觉上，同时更贴近 DeepSeek-R1 这类 reasoning model。

## 13. 为什么 DPO 不是 RL，但又和 RLHF 有关系

DPO 经常被放在 RLHF 的替代方法里，但它本身不是在线 RL。

它不采样当前 policy 的新 outputs 来计算 reward，也不做环境交互。

但 DPO 和 RLHF 有理论联系：DPO 从 reward model 的隐式形式出发，把偏好优化转成 policy loss。

所以 DPO 更像是：

> 把“先学 reward，再优化 policy”的两步合成一个监督式目标。

这也是为什么 DPO 训练起来像 SFT，但作用却更接近偏好对齐。

## 14. 为什么 GRPO 不是简单版 PPO

GRPO 省掉 value model，但它引入了 group sampling。

它不是“PPO 少一个模型”这么简单。

PPO 的 baseline 来自 value model。

GRPO 的 baseline 来自同一 prompt 下其他 samples。

这意味着 GRPO 的训练效果很依赖采样分布。如果 samples 没有差异，学习信号就弱。如果 samples 很混乱，advantage 也可能噪声很大。

所以 GRPO 的关键不只是公式，而是：

- group size 怎么选
- sampling temperature 怎么设
- reward 怎么设计
- 是否过滤无效 groups
- 如何控制 response length
- KL penalty 怎么加

这些细节决定了 GRPO 是否稳定。

## 15. 读 PPO / DPO / GRPO 论文时看什么

我会看这些问题：

- 数据是离线 preference pair，还是 online rollout？
- Reward 来自人类偏好、reward model，还是 verifier？
- 是否需要 value model？
- Reference model 是哪个 checkpoint？
- KL 系数怎么设？
- 每个 prompt 采样几个 responses？
- 是否使用 length penalty？
- 是否报告训练 token 成本？
- 提升是 greedy decoding 下提升，还是采样多次后提升？
- 有没有看 reward hacking 或长度膨胀？

很多方法 benchmark 看起来提升，但可能是采样更多、输出更长、筛选更强带来的。读的时候要把训练方法和推理策略分开看。

## 16. 小结

PPO、DPO、GRPO 是三条不同的后训练路线。

PPO 是经典 RLHF：强大但重。

DPO 是直接偏好优化：简单稳定，但依赖 preference pairs。

GRPO 是 reasoning RL 常用路线：省掉 value model，但依赖 group sampling 和可验证 reward。

我的理解是：

> PPO 学 reward，DPO 学偏好差，GRPO 学组内相对好坏。

后面讲 GRPO 变体时，很多改动都围绕一个问题展开：怎么让这个“组内相对好坏”的训练信号更稳定、更有效。

## 参考资料

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
