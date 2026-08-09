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
  - 笔记
math: true
category_bar: true
---

大模型后训练里的 RL 和传统 RL 不太一样。传统 RL 常见场景是 agent 在环境里反复交互，比如机器人控制、游戏、推荐系统。LLM 后训练里的“环境”往往更窄：给一个 prompt，模型生成 response，然后由人类偏好、reward model、规则 verifier、测试用例或者答案匹配器给出反馈。

有一说一，传统RL学起来真有点绕脑子的😓

<!-- more -->

## RL的基本概念
为了搞清楚 RL，先把 terminology 理清楚很重要。

RL 的一部分要从随机过程讲起。更准确地说，RL 的目标是在不确定的交互过程中，通过学习 policy 来最大化期望累计回报。
在公式中，大写字母 $X$ 表示随机变量，小写字母 $x$ 表示观测到的具体值，比如 $x_0 = 1$、$x_1 = 0$。

![强化学习中的 agent-environment 交互](/img/post-training/rl-agent-environment.png)

然后引入一些定义
agent：实际选择 action 的主体，与 environment 交互，目标是最大化 expected return。
environment：agent 用来交互的环境。
action：agent 可以采取的行为，比如在马里奥游戏中向左、向右或向前移动。
state：描述当前环境状况的信息；agent 执行 action 后，environment 会转移到新的 state。

state space 和 action space 分别是所有可能 state 和 action 的集合。

trajectory：一条完整的交互轨迹可以写成：

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_{T-1}, a_{T-1}, r_{T-1}, s_T)
$$

**policy function**：一个概率函数，用来输出给定当前 state 时执行某个 action 的概率。

$$
\pi_\theta(a_t \mid s_t) = P(A_t = a_t \mid S_t = s_t)
$$

**reward & return**
reward：与环境交互后获得的奖励
return：cumulative future reward，表示从当前时刻开始累计得到的 reward。

PDF 里用大写 $R_t$ 表示随机 reward，用大写 $U_t$ 表示随机 return。先不考虑折扣时：

$$
U_t = \sum_{k=t}^{T-1} R_k
$$

当一条具体轨迹采样完成后，$R_k$ 变成观测到的 reward $r_k$，对应的 return 可以写成 $u_t$。很多 RL 文献也会直接用 $G_t$ 表示这条轨迹上的 return。

$$
G_t \equiv u_t = \sum_{k=t}^{T-1} r_k
$$

discounted return：由于即时奖励和之后的奖励不能一概而论（比如现在拿 100 和一年后拿 100），所以引入折扣率 $\gamma$ 来控制未来 reward 的影响程度。

$$
U_t = \sum_{k=t}^{T-1} \gamma^{k-t} R_k
\qquad\text{and}\qquad
G_t \equiv u_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k
$$

如果不区分“随机 return”和“采样后的具体 return”，也可以把它们都简写成 $G_t = U_t$。这里保留这个区分，是为了和 PDF 中的记号保持一致。

**state transition function**
一个旧的state转移到新的state的概率函数

RL中的随机性主要来源于两个部分
1. policy选择动作时的随机性
2. 状态转移函数带来的随机性

### Value function
**Action-value function**：$Q_{\pi}(s_t, a_t)$ 用来表示当前的state，policy下，使用action a_t所能得到的return。这样就可以衡量一个action的好坏。不过由于state一般认为不变，一个更好的policy 也可以提升action的return。

$$
Q^\pi(s_t, a_t) = \mathbb{E}_\pi\left[U_t \mid S_t = s_t, A_t = a_t\right]
$$

这里的期望是在固定当前 $(s_t, a_t)$ 后，对后续 state 和 action 的随机性求平均。这样就能比较同一个 state 下不同 action 的长期价值。

**State-value function**
状态价值函数进一步对当前 action 取期望：

$$
V^\pi(s_t) = \mathbb{E}_{A_t \sim \pi(\cdot \mid s_t)}\left[Q^\pi(s_t, A_t)\right]
    = \sum_a \pi(a \mid s_t) Q^\pi(s_t, a)
$$

这里进一步对当前 action 取期望，把 action 选择本身的随机性也纳入 value function。

### Return 的估计：Monte Carlo 与 Bellman equation
前面定义的 $U_t$ 是一个随机变量，实际训练时需要用采样到的 trajectory 去估计它，以及估计 $V^\pi(s)$ 和 $Q^\pi(s,a)$。最直接的两条路线是 Monte Carlo 和 Bellman recursion。

#### 蒙特卡洛（Monte Carlo）：等 episode 结束后再算

Monte Carlo（MC）的思路很直接：让 agent 按照当前 policy 跑完一整条 trajectory，等所有 reward 都拿到之后，再从后往前计算每个时刻的 return。

假设一条 trajectory 上的 reward 是：

```text
r_0 = 0,  r_1 = 1,  r_2 = 2
```

取 $\gamma = 0.9$，那么：

$$
G_2 = r_2 = 2
$$

$$
G_1 = r_1 + \gamma r_2 = 1 + 0.9 \times 2 = 2.8
$$

$$
G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 = 0 + 0.9 \times 1 + 0.9^2 \times 2 = 2.52
$$

如果同一个 state $s$ 在不同 trajectory 中被访问了多次，就可以用这些 trajectory return 的平均值估计 state value：

$$
V^\pi(s) \approx \frac{1}{N(s)}\sum_{i=1}^{N(s)} G_t^{(i)}
$$

同理，也可以只统计在 $(s,a)$ 处采取 action $a$ 后得到的 return，估计 action value：

$$
Q^\pi(s,a) \approx \frac{1}{N(s,a)}\sum_{i=1}^{N(s,a)} G_t^{(i)}
$$

MC 的优点是目标值来自完整 trajectory，不需要提前知道环境的 transition function，也不会用当前 value estimate 去构造自己的训练目标。只要 trajectory 是按照 policy $\pi$ 采样的，样本数量足够多，平均 return 就能收敛到真实的 value。

它的问题也很明显：必须等 episode 结束才能计算 return。对于游戏还好，但对于没有明确终点的环境，或者 LLM 生成很长 response 的场景，等待成本会比较高。而且一条 trajectory 中任何一个 reward 的随机性都会传到前面所有时刻，所以 MC 的 variance 往往比较大。

#### 贝尔曼方程（Bellman equation）：把长期回报拆成一步

Bellman equation 的核心是把 return 拆成当前 reward 和下一时刻的 return：

$$
U_t = R_t + \gamma U_{t+1}
$$

对当前 state 和 policy 取期望，就得到 Bellman expectation equation：

$$
V^\pi(s) = \sum_a \pi(a \mid s)\sum_{s'}p(s' \mid s,a)
\left[r(s,a,s') + \gamma V^\pi(s')\right]
$$

它表达的是：当前 state 的价值，等于当前 action 得到的 immediate reward，加上下一 state 的价值，再对 policy 和 state transition 的随机性取平均。

对应的 action-value 形式是：

$$
Q^\pi(s,a) = \sum_{s'}p(s' \mid s,a)
\left[r(s,a,s') + \gamma\sum_{a'}\pi(a' \mid s')Q^\pi(s',a')\right]
$$

这两个式子用于 policy evaluation：policy $\pi$ 已经给定，目标是计算它的 $V^\pi$ 或 $Q^\pi$。

如果目标不是评估一个固定 policy，而是直接寻找最优策略，就得到 Bellman optimality equation：

$$
V^*(s) = \max_a \sum_{s'}p(s' \mid s,a)
\left[r(s,a,s') + \gamma V^*(s')\right]
$$

$$
Q^*(s,a) = \sum_{s'}p(s' \mid s,a)
\left[r(s,a,s') + \gamma\max_{a'}Q^*(s',a')\right]
$$

这里的 $\max$ 体现了最优 action 选择，而 Bellman expectation equation 里的 $\pi(a \mid s)$ 体现的是按照给定 policy 采样 action。

#### 单步 TD：走一步就能更新

Bellman equation 不需要等整条 trajectory 结束。单步 TD 只使用当前 transition：

1. 执行 action，得到即时 reward $r_t$ 和下一状态 $s_{t+1}$。
2. 用当前 value function 估计未来价值。
3. 立刻计算 TD error，并更新 critic，甚至可以用它更新 actor。

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

这里的 $V(s_{t+1})$ 不是完整的真实 return，而是当前 value function 对未来的估计，所以单步 TD 可以在线更新，但会引入 bootstrap bias。它只看一步真实 reward，后面的结果完全依赖 value estimate，方差通常较低，但偏差可能比较大。

#### 广义优势估计（GAE）：在 MC 和 TD 之间取平衡

PPO 中常用的 GAE（Generalized Advantage Estimation）并不是另一个 value function，而是一种估计 advantage 的方法。它先计算每一步的 TD error，再把未来多个时间步的 TD error 做指数加权：

$$
A_t^{\mathrm{GAE}(\gamma,\lambda)}
= \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V
$$

其中：

- $\gamma$ 控制未来 reward 的折扣；
- $\lambda$ 控制 GAE 向多步 return 延伸的程度；
- $\delta_t^V$ 是当前时刻的 TD error。

在有限长度的 rollout 中，实际计算会在轨迹片段末尾截断：

$$
\hat A_t = \delta_t^V + \gamma\lambda\hat A_{t+1}
$$

从最后一个时间步开始反向递推即可。若最后状态不是 terminal state，就用 value function 对下一状态进行 bootstrap：

$$
\delta_{H-1}^V
= r_{H-1} + \gamma V(s_H) - V(s_{H-1})
$$

如果最后状态确实是 terminal state，则终止状态之后没有未来 reward，通常令 bootstrap value 为 0。

实际实现中通常用 $d_t$ 区分 terminal 和 rollout truncation：真正结束时 $d_t=1$，只是达到固定采样长度时 $d_t=0$。于是 TD error 可以统一写成：

$$
\delta_t^V = r_t + \gamma(1-d_t)V(s_{t+1}) - V(s_t)
$$

这样，固定长度截断仍然保留 $V(s_{t+1})$ 的 bootstrap；只有 episode 真正结束时才移除未来价值。

GAE 的定位可以这样理解：

- 当 $\lambda=0$ 时，只保留当前一步 TD error，接近单步 TD；
- 当 $\lambda$ 接近 1 时，会融合更长范围的 TD error，接近 Monte Carlo；
- 中间的 $\lambda$ 用来平衡 bias 和 variance。

所以 GAE 不能做到“刚走一步就算出完整的 $A_t$”。$A_t$ 依赖后面多个时间步的 TD error，至少要等一段 rollout 收集完之后才能反向计算。但它也不需要等完整 episode 结束：只要收集固定长度的轨迹片段，就可以在片段末尾用 $V(s_H)$ 自举。

实际实现会从 rollout 的最后一步反向递推。`values` 比 `rewards` 多一个元素，其中最后的 `values[-1]` 是 rollout 被截断时使用的 bootstrap value：

```python
import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    rewards: [T]
    values:  [T + 1]，包含最后状态的 bootstrap value
    dones:   [T]，真正 terminal 的位置为 1，普通 rollout 截断为 0
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = 0.0

    for t in reversed(range(len(rewards))):
        not_terminal = 1.0 - dones[t]
        delta = (
            rewards[t]
            + gamma * not_terminal * values[t + 1]
            - values[t]
        )
        next_advantage = (
            delta
            + gamma * gae_lambda * not_terminal * next_advantage
        )
        advantages[t] = next_advantage

    returns = advantages + values[:-1]
    return advantages, returns
```

如果最后一步只是达到固定 rollout 长度，`dones[-1]=0`，代码会保留 `values[-1]`；如果 episode 真正结束，`dones[-1]=1`，未来 value 和 advantage 都会在这里截断。

#### PPO 中的实际流程：批次采样后统一计算优势

工业界通常不是走一步就更新一次，而是采用 rollout batch 的方式：

1. 用当前 policy 与 environment 交互，收集一批固定长度的 trajectory。
2. 保存每一步的 state、action、reward、old log probability 和 value estimate。
3. 数据收集完成后，从后往前用 GAE 计算每一步的 advantage。
4. 用这批数据更新 actor 和 critic 多个 epoch。
5. 丢弃旧 batch，再用更新后的 policy 重新采样。

因此，GAE 需要未来若干步信息这件事，在 PPO 的批次训练范式中并不是问题。它换来了更稳定的 advantage，以及比纯 MC 更低的等待成本和更好的样本利用方式。

| 方法 | target 从哪里来 | 是否需要等 episode 结束 | 典型特点 |
| --- | --- | --- | --- |
| Monte Carlo | 完整 trajectory 的实际 return $G_t$ | 需要 | 无 bootstrap，方差较大 |
| 单步 TD | $r_t + \gamma V(s_{t+1})$ | 不需要 | 可以在线更新，但 bootstrap bias 较大 |
| GAE | 多个 TD error 的指数加权和 | 不需要等完整 episode，但要等一段 rollout | 用 $\lambda$ 平衡 bias 和 variance |

可以把三者记成一句话：单步 TD 只相信下一步，Monte Carlo 相信完整 trajectory，GAE 则在两者之间做平滑融合。Actor-critic 方法通常正是用 critic 估计 value，再用 GAE 产生 advantage，给 actor 提供更新方向。


## 语言模型里的 policy 是什么

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

## Token-level action 和 response-level action

理论上，LLM 每一步生成 token 都是一个 action。

但很多 reward 是 response-level 的。比如数学题最后答案对不对、代码是否通过测试、回答是否更符合人类偏好，这些通常是整段回答生成完之后才知道。

这就带来一个 credit assignment 问题：

> 如果最终回答错了，到底是哪几个 token 造成的？

PPO 里会尝试用 value model 和 advantage estimation，把整段 reward 分摊到 token-level 更新上。

GRPO 里则常见做法是对同一个 prompt 采样多个 responses，再用组内 reward 相对大小得到 advantage。

DPO 更进一步，不显式做在线 RL，而是直接比较 chosen response 和 rejected response 的 log probability。

这些算法看起来差很多，但都绕不开同一个问题：训练信号怎样从“整段回答好不好”传回到模型参数。

## Reward：后训练真正优化的目标

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

## Return、value 和 advantage

在一般 RL 里，$G_t$ 通常表示一条已采样轨迹上，从当前时刻往后得到的 discounted return：

$$
G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k
$$

LLM 后训练里，很多任务只有最终 reward，所以可以粗略理解为整段回答的得分。

Value function 估计的是某个状态下未来能拿到多少 reward：

$$
V^\pi(s) = \mathbb{E}_{\pi}[U_t \mid s_t=s]
$$

Advantage 衡量的是某个 action 比平均水平好多少：

$$
A_t = G_t - V^\pi(s_t)
$$

这个量很重要。

如果一个回答得分是 0.8，不一定说明它很值得加强。也许这个 prompt 很简单，随便生成都能 0.8。

如果另一个回答得分是 0.6，也不一定差。也许这个 prompt 很难，大多数回答都是 0.1。

Advantage 想表达的是：相对于当前状态的预期表现，这个 action / response 到底更好还是更差。

## Policy gradient 的直觉

Policy gradient 的经典形式可以写成：

$$
\nabla_\theta J(\theta) = \underset{\pi_\theta}{\mathbb{E}}[\nabla_\theta \log \pi_\theta(a \mid s) A(s,a)]
$$

放到 LLM 里，可以粗略理解成：

> 如果某个 response 的 advantage 为正，就提高它的生成概率；如果 advantage 为负，就降低它的生成概率。

但这里有两个细节点。

第一，提高的是整段 response 里 tokens 的 log probability，不是直接给最终答案加分。

第二，更新不能太大。语言模型是高维分布，一次更新如果把概率分布拉得太远，可能会破坏原来的语言能力。

这就引出 KL constraint。

## KL constraint：防止模型跑远了

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

## On-policy 和 off-policy

On-policy 指训练数据来自当前 policy。

Off-policy 指训练数据来自旧 policy、其他模型，或者离线数据集。

LLM 后训练里，这个区别非常关键。

如果模型训练时看到的是自己当前会生成的回答，那训练分布和推理分布更接近。这是 on-policy 的优势。

但 on-policy 代价高，因为训练过程中要不断采样新 responses。

Off-policy 成本低，可以直接用已有数据训练。但如果数据分布和当前模型差很多，模型学到的更新可能不稳定。

这就是为什么 PPO / GRPO / OPD 都强调 on-policy 或接近 on-policy。它们希望模型在自己的生成分布上学习，而不是只模仿静态数据。

## 为什么 reasoning model 更依赖 RL：RL 和 SFT 的本质区别

我觉得应该这么理解，虽然SFT 可以让模型学会解题格式，但它本质上是模仿。
而在reasoning问题上，潜在的解题方式有很多，比如1+1=2可以通过一堆不一样的复杂公式来给出同样的结果，这种情况下的action space太大了，只靠SFT是模仿不完的。所以需要用RL来自己探索。

比如 RLVR 可以直接优化最终正确率。同一个数学题，模型生成多个答案，只有正确答案拿到 reward。经过训练，模型会提高能够得到正确答案的生成轨迹概率。只要合理设置 reward（比如答案是否正确、输出长度、步骤数等），就可以让模型自己探索解题空间。

这也是 DeepSeek-R1 这类 reasoning model 重要的地方：它不是只靠人工写好的 CoT，而是通过可验证 reward 强化推理行为。

不过这也带来风险。

如果 reward 只看最终答案，模型可能学到短路策略。比如中间推理不可靠，但答案格式正确；或者生成更长 reasoning 来提高偶然命中率。


**后训练里的几个常见失败模式**

第一，reward hacking。

典中典问题，模型学会利用 reward 的漏洞，靠作弊来拿分。

"Learning to Summarize with Human Feedback" (OpenAI)
在 RLHF 训练摘要模型时，明确观察到模型学会利用 reward model 的偏差。

第二，mode collapse。

模型输出变得单一，缺少多样性。

第三，length inflation。

模型生成越来越长，因为长回答更容易看起来努力，或者更容易覆盖答案。

第四，over-optimization。

训练集 reward 越来越高，但真实评估变差。

"Scaling Laws for Reward Model Overoptimization" (Anthropic)
定量研究了 reward model score 持续升高而真实人类偏好评分反而下降的现象，提出了 over-optimization 的标度律。

> Goodhart's Law: "当某个度量变成目标时，它就不再是一个好的度量。"

第五，distribution drift。

policy 离 reference model 太远，语言能力或安全边界受损。


## 参考资料

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
