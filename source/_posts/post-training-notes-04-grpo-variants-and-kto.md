---
title: "后训练笔记 04：GRPO 变体，从归一化、ratio 到 group-aware reward"
date: 2025-11-22 21:30:00
updated: 2026-08-23 22:30:00
categories:
  - 后训练笔记
tags:
  - 后训练
  - GRPO
  - DAPO
  - Dr.GRPO
  - CISPO
  - GSPO
  - SAPO
  - GAPO
  - 笔记
math: true
category_bar: true
---

GRPO 之后出现了很多相关的改进算法。

这篇文章围绕四个问题整理 GRPO 变体：

1. reward 如何变成 advantage；
2. token loss 如何聚合与归一化；
3. importance ratio 在什么粒度上计算，surrogate 又如何约束 policy shift；
4. 哪些 prompt、response 和 token 真正进入梯度。

按照这个坐标系，本文精读六项代表性工作：DAPO、Dr.GRPO、CISPO、GSPO、SAPO 和 GAPO。

<!-- more -->

## 统一坐标：GRPO baseline

给定 prompt $q$，旧策略对它采样 $G$ 个回答：

$$
o_1,\ldots,o_G \sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid q).
$$

Verifier 为每个回答给出 reward $R_i$。标准 GRPO 用组内均值和标准差构造 advantage：

$$
\hat A_i
=
\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)}.
$$

同一回答里的 token 通常共享这个 response-level advantage。对第 $i$ 个回答的第 $t$ 个 token，importance ratio 为：

$$
r_{i,t}(\theta)
=
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
}.
$$

标准 clipped surrogate 可以记为：

$$
\ell_{i,t}(\theta)
=
\min\left(
r_{i,t}(\theta)\hat A_i,
\operatorname{clip}\left(r_{i,t}(\theta),1-\epsilon,1+\epsilon\right)\hat A_i
\right).
$$

最后还要决定怎样把所有 token 的 loss 聚合成一个标量。原始 GRPO 常见写法是先在每条回答内部平均，再对回答平均：

$$
\mathcal L_{\mathrm{GRPO}}
=
-\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\ell_{i,t}(\theta).
$$

这四步正好对应后续变体的主要分歧：

```text
group rewards
    ↓
advantage 构造与缩放
    ↓
token / sequence importance ratio
    ↓
hard clip / clipped weight / soft gate
    ↓
mask 与 loss normalization
    ↓
policy update
```



## 第一组：归一化、优势尺度与长度偏置

### DAPO：它首先是一套长 CoT RL recipe

[DAPO](https://arxiv.org/abs/2503.14476) 很容易被简化成“把 sample-level loss 改成 token-level loss”。论文的消融恰好说明，这种理解太窄。DAPO 真正处理的是长链路 reasoning RL 中同时出现的多个训练症状。

#### DAPO 在修什么

1. **Entropy collapse**：policy 很快变尖，同组回答趋于相似，探索 token 难以被继续提升；
2. **零梯度 prompt**：同组回答全对或全错时，组内 advantage 全为零；
3. **长回答中的 token 被稀释**：先对每条回答按长度平均，会让长回答中单个 token 的权重更小；
4. **截断样本的 reward noise**：一个推理方向可能是对的，只是超过 generation budget，却被直接当成错误回答惩罚。

#### DAPO 的四个官方组件

**1. Clip-Higher：给正向探索更宽的上界**

DAPO 将上下 clipping boundary 解耦：

$$
r_{i,t}\in[1-\epsilon_{\mathrm{low}},\ 1+\epsilon_{\mathrm{high}}],
\qquad
\epsilon_{\mathrm{high}}>\epsilon_{\mathrm{low}}.
$$

论文使用的直觉是：低概率 exploration token 即使 ratio 增加相同倍数，绝对概率仍然很低。对它使用和高概率 token 相同的上界，会过早阻止这些稀有路径继续增大概率。提高上界主要是在正 advantage 方向保留探索空间，而不是无条件放大所有更新。

**2. Dynamic Sampling：保证 batch 中有足够的有效 prompt**

对于二元 verifier reward，如果某个 prompt 的 $G$ 个回答全对或全错，则：

$$
R_1=\cdots=R_G
\quad\Longrightarrow\quad
\hat A_1=\cdots=\hat A_G=0.
$$

这些 group 占用 rollout 和训练 batch，却不产生 policy gradient。DAPO 持续过采样并过滤组内准确率为 $0$ 或 $1$ 的 prompt，直到有效 batch 被填满：

$$
0 < \sum_{i=1}^{G}\mathbb 1[R_i=1] < G.
$$

它改善的是 batch 的有效梯度密度，不是发明新的 advantage 公式。代价是需要额外 rollout；收益是减少零梯度 group 对 batch gradient 的稀释。

**3. Token-Level Policy Gradient Loss：让 token 而不是 response 获得等权重**

DAPO 将 loss 改为在整个有效 token 集合上归一化：

$$
\mathcal L_{\mathrm{DAPO}}
=
-\frac{1}{\sum_{i=1}^{G}|o_i|}
\sum_{i=1}^{G}
\sum_{t=1}^{|o_i|}
\ell_{i,t}(\theta).
$$

原始 GRPO 是“每条回答等权”，DAPO 是“每个有效 token 等权”。因此长回答在总梯度里会有更高权重，但长回答中的每个 token 不再因为回答更长而被额外稀释。

**4. Overlong Reward Shaping：区分截断和真正错误**

DAPO 先给出 Overlong Filtering：截断 completion 不参与 loss，而不是直接给负 reward。对应到工程实现，就是让 truncated token mask 为零。

论文又加入 Soft Overlong Punishment。在接近最大长度的缓冲区间内，长度惩罚连续增加；只有真正越过上限时才达到最大惩罚。概念上可以写成：

$$
R_{\mathrm{length}}(o)=
\begin{cases}
0, & |o|\le L_{\max}-L_{\mathrm{cache}},\\
-\dfrac{|o|-(L_{\max}-L_{\mathrm{cache}})}{L_{\mathrm{cache}}},
& L_{\max}-L_{\mathrm{cache}}<|o|\le L_{\max},\\
-1, & |o|>L_{\max}.
\end{cases}
$$

所以 `mask_truncated_completions=True` 和 soft overlong punishment 都属于 Overlong Reward Shaping 的实现，但它们不是 token-level loss 的替代品。


### Dr.GRPO：去掉两个看似自然的归一化项

[Dr.GRPO](https://arxiv.org/abs/2503.20783) 来自论文 *Understanding R1-Zero-Like Training: A Critical Perspective*。它的核心贡献不是增加新模块，而是指出标准 GRPO 的两个分母会隐式改变样本权重。

#### 偏置一：response-level length bias

原始 GRPO 对每条回答除以自身长度 $|o_i|$。这会产生不对称行为：

- 对正 advantage，短的正确回答获得更大的单 token 更新；
- 对负 advantage，长的错误回答因为除数更大而被惩罚得更轻。

结果不是单纯“模型偏好短回答”，而是更具体的：**正确回答被推向更短，错误回答却可能被推向更长。** 论文观察到错误 response 的长度增长尤其明显。

Dr.GRPO 用全局常数 $L$ 代替每条回答自己的长度，通常取最大 completion length：

$$
\mathcal L_{\mathrm{Dr.GRPO}}
=
-\frac{1}{GL}
\sum_{i=1}^{G}
\sum_{t=1}^{|o_i|}
\ell_{i,t}(\theta).
$$

因为 $L$ 对所有回答相同，回答长度不再偷偷进入 sample weight。

#### 偏置二：question-level difficulty bias

标准 GRPO 还会用每个 group 自己的 reward std 做缩放：

$$
\hat A_i=\frac{R_i-\bar R}{\operatorname{std}(R_1,\ldots,R_G)}.
$$

不同 prompt 的 std 不同，相当于给题目施加不同权重。Dr.GRPO 认为，这不是无害的方差归一化，而是 question-level reweighting。它建议只保留中心化：

$$
\hat A_i=R_i-\bar R.
$$

#### Dr.GRPO 的方法边界

Dr.GRPO 和 DAPO 都讨论长度，但它们不相同：

- DAPO 用全 batch 的有效 token 数做分母，使每个 token 等权；
- Dr.GRPO 用固定 generation budget 做分母，目标是彻底移除由实际回答长度引起的权重变化；
- DAPO 是多组件 recipe；Dr.GRPO 是对 loss 和 reward scaling 的最小 bias correction。

Dr.GRPO 的限制也很清楚：关闭 std scaling 后，reward 的原始尺度和 batch composition 会直接影响梯度大小。二元 verifier reward 比较适合这种设计；如果混合多个量纲不同的 reward，需要先处理 reward calibration。

#### Dr.GRPO 在 TRL 中的改动

```python
dr_grpo_args = GRPOConfig(
    loss_type="dr_grpo",
    scale_rewards=False,       # 只做组内中心化，不除以 group std
    max_completion_length=8192,
    beta=0.0,
)
```

这两个开关要一起看：`loss_type="dr_grpo"` 去掉 response-length denominator，`scale_rewards=False` 去掉 group-std denominator。

## 第二组：importance ratio 与 surrogate 形状

### CISPO：clip importance weight，但保留 token 梯度

[CISPO](https://arxiv.org/abs/2506.13585) 是 MiniMax-M1 技术报告中的 RL 算法。它关心的不是 reward 怎样归一化，而是 hard clipping 在多轮 off-policy update 中会丢掉什么。


#### CISPO 的关键改动

MiniMax 的实验观察到，一些反思或转折 token，例如 “However”“Wait”“Recheck”，在 base model 中概率很低。一次更新后，它们的 ratio 可能快速越过 clipping boundary。标准 PPO/GRPO surrogate 会让这些 token 在后续 minibatch update 中失去梯度。

问题在长 CoT 中更明显：稀有 token 可能正好代表推理路径的分叉点，但 hard clip 把它们最先丢掉。


CISPO 先裁剪 importance sampling weight：

$$
\hat r_{i,t}(\theta)
=
\operatorname{clip}\left(
r_{i,t}(\theta),
1-\epsilon_{\mathrm{IS}}^{\mathrm{low}},
1+\epsilon_{\mathrm{IS}}^{\mathrm{high}}
\right),
$$

然后对这个 weight 使用 stop-gradient，并保留 log-policy gradient：

$$
\mathcal J_{\mathrm{CISPO}}
=
\frac{1}{\sum_i|o_i|}
\sum_{i,t}
\operatorname{sg}\!\left(\hat r_{i,t}(\theta)\right)
\hat A_i
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

标准 clipped surrogate 在越界区域可能直接给零梯度；CISPO 则限制这个 token 对梯度的权重，却继续保留它的学习方向。

论文实现还叠加了 DAPO 的 Dynamic Sampling 和 length penalty。因此 MiniMax-M1 的整体收益不能全部归因于 CISPO 的单一公式。

### GSPO：让 ratio 的粒度与 reward 对齐

[GSPO](https://arxiv.org/abs/2507.18071) 的出发点更激进：它认为 **token-level importance ratio**在语言生成里没有完成预期的 distribution correction，反而把单样本噪声沿长序列累积起来。

#### GSPO 的关键改动

Reasoning reward 通常属于整段回答：答案是否正确、代码是否通过测试，都是 sequence-level 结果。GRPO 却对每个 token 单独计算 ratio 和 clipping condition。

GSPO 提出一个原则：

> optimization unit 应该与 reward unit 对齐。

于是GSPO 将一整条回答的 token log-ratio 取平均，再指数化：

$$
s_i(\theta)
=
\exp\left(
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\log r_{i,t}(\theta)
\right).
$$

这等价于 token ratio 的几何平均，也是做过长度归一化的 sequence likelihood ratio。随后，一整条回答共享同一个 clipping decision：

$$
\mathcal L_{\mathrm{GSPO}}
=
-\frac{1}{G}
\sum_{i=1}^{G}
\min\left(
s_i(\theta)\hat A_i,
\operatorname{clip}(s_i(\theta),1-\epsilon,1+\epsilon)\hat A_i
\right).
$$

这里最重要的不是“把 token 求平均”本身，而是 clipping 的对象变了：GRPO 决定哪些 token 越界，GSPO 决定哪条 response 整体越界。

- GSPO 为什么对 MoE的贡献

论文在 Qwen3-30B-A3B MoE 上比较 GSPO 和 GRPO。GRPO 需要 Routing Replay 才能正常收敛，GSPO 在不使用该稳定化机制时仍保持训练稳定，并在 AIME 2024、LiveCodeBench 和 CodeForces 上显示更好的训练效率。

论文还报告 GSPO 的平均 clipping fraction 约为 $0.15$，GRPO 约为 $0.0013$。GSPO 虽然一次排除更多 token，却训练得更好；作者据此认为 GRPO 保留下来的 token-level gradient 本身含有较强噪声。

这个结论目前主要由 Qwen3 MoE 长序列训练支持。对较短回答、dense model 或接近完全 on-policy 的训练，sequence-level ratio 的优势是否同样大，还需要单独验证。

#### GSPO 在 TRL 中的改动

```python
gspo_args = GRPOConfig(
    importance_sampling_level="sequence",
    loss_type="grpo",
    epsilon=3e-4,
    epsilon_high=4e-4,
    beta=0.0,
)
```

`importance_sampling_level="sequence"` 是决定 ratio 粒度的关键开关。GSPO 的 ratio 与 token ratio 数值尺度不同，不能直接沿用 GRPO 常见的 `0.2` clipping range。

### SAPO：把 hard clip 改成 soft gate

[SAPO](https://arxiv.org/abs/2511.20347) 接受 token-level adaptivity，但不接受 hard clipping 的全有或全无行为。

#### SAPO 在修什么

GRPO 的 hard clip 在边界处不连续：ratio 还在 trust region 内时保留完整梯度，一旦越界，梯度可能突然变为零。GSPO 把这个决定提升到 sequence-level 后，还会出现另一个问题：少数极端 off-policy token 可能让整条序列被裁掉，连同其中大量 near-on-policy token 的有效信号一起消失。

#### SAPO 的两个核心改动

**1. 用连续 soft gate 替代 hard clipping**

SAPO 定义：

$$
f_{i,t}(x)
=
\sigma\left(\tau_{i,t}(x-1)\right)
\frac{4}{\tau_{i,t}},
$$

并使用：

$$
\mathcal J_{\mathrm{SAPO}}
=
\frac{1}{G}
\sum_i\frac{1}{|o_i|}
\sum_t
f_{i,t}\!\left(r_{i,t}(\theta)\right)\hat A_i.
$$

对它求导后，gradient weight 在 $r_{i,t}=1$ 时达到最大值 $1$，ratio 离 $1$ 越远，权重越平滑地衰减。它不会像 hard clip 那样在某条边界外突然归零。

**2. 对正、负 advantage 使用不同温度**

$$
\tau_{i,t}=
\begin{cases}
\tau_{\mathrm{pos}}, & \hat A_i>0,\\
\tau_{\mathrm{neg}}, & \hat A_i\le 0.
\end{cases}
$$

SAPO 建议 $\tau_{\mathrm{neg}}>\tau_{\mathrm{pos}}$，也就是让负 advantage token 的梯度衰减更快。论文给出的解释是：负向更新会降低 sampled token，同时相对抬高大词表中大量未采样 token，更容易在 off-policy 条件下引入不稳定。

#### SAPO 与 CISPO、GSPO 的边界

- CISPO 裁的是 importance weight，并通过 stop-gradient 保留 log-policy gradient；
- GSPO 仍然 hard clip，但 clipping unit 是整个 sequence；
- SAPO 仍保留 token-level ratio，却用连续 gate 逐 token 衰减梯度。

在 Qwen3-30B-A3B 的数学 reasoning 实验中，SAPO 比 GSPO 和带 Routing Replay 的 GRPO-R2 更稳定；温度消融也显示 $\tau_{\mathrm{neg}}>\tau_{\mathrm{pos}}$ 最稳定。论文还在 Qwen3-VL 的 dense/MoE、多模态和文本任务上报告一致提升。

#### SAPO 在 TRL 中的实现

```python
sapo_args = GRPOConfig(
    loss_type="sapo",
    sapo_temperature_pos=1.0,
    sapo_temperature_neg=1.05,
    beta=0.0,
)
```

SAPO 的温度不是普通 sampling temperature，而是控制 off-policy gradient 随 ratio 偏移衰减多快的 trust-region 参数。

## 第三组：reward-level 方法

### GAPO：reward 的定义对象从单个回答变成整组回答

[GAPO](https://arxiv.org/abs/2511.12596) 不是纯粹的 GRPO loss 变体。它不改 advantage normalization、ratio 或 clipping，而是改变 reward function 的输入。

#### GAPO 在修什么

标准 GRPO 默认先独立给每个回答打分：

$$
R_i=R(q,o_i).
$$

这适合正确性等单样本属性，却难以直接表达“整组回答是否覆盖了多种有效模式”。如果 prompt 有多个同样有效的答案，独立 reward 可能让最常见的答案不断获得强化，最终出现 mode collapse。

GAPO 改成：

$$
(\widetilde R_1,\ldots,\widetilde R_G)
=
\widetilde R(q,o_1,\ldots,o_G).
$$

每个 rollout 的 reward 可以依赖同组其他 rollout。

#### frequency-aware reward

论文研究一个有明确有效集合 $\mathcal V$ 的任务。对组内有效答案 $v$，先计算经验频率：

$$
f_v(\mathbf o)
=
\frac{\sum_{i=1}^{G}\mathbb 1[o_i=v]}
{\sum_{i=1}^{G}\mathbb 1[o_i\in\mathcal V]}.
$$

如果目标是对 $L=|\mathcal V|$ 个答案均匀采样，则：

$$
\widetilde R_i=
\begin{cases}
1-\left(f_{o_i}(\mathbf o)-\dfrac{1}{L}\right), & o_i\in\mathcal V,\\
-1, & o_i\notin\mathcal V.
\end{cases}
$$

低频但有效的答案获得更高 reward，高频答案获得较低 reward，无效答案仍被惩罚。得到的 group-aware reward vector 再交给普通 GRPO 计算 advantage 和 loss。

#### GAPO 的证据与边界

在固定列表选择任务中，GAPO 让输出分布更接近均匀分布，平均 Jensen-Shannon divergence 从基线大于 $0.3$ 降到小于 $0.1$。在开放类别任务中，GAPO 微调后的 Qwen2.5-32B 在 500 次采样中平均给出 147 个不同答案，微调前为 24 个。

创意写作实验也显示 semantic distance 和 $1-\mathrm{SelfBLEU}$ 提升。但标准 benchmark 结果不是每项都上升，例如 GSM8K exact match 和 MMLU-Pro 略降，而 flexible/verify 指标与 HumanEval 略升。因此更谨慎的结论是：GAPO 显著改善多样性，并在论文测试中大体维持能力，而不是“多样性完全没有代价”。

它还有一个重要限制：frequency-aware reward 假设有效答案集合已知，并且目标分布接近均匀。开放式任务中什么算“有效”、不同模式应该占多少比例，仍然需要额外 verifier 或 reward model。

#### GAPO 在 TRL 中的最小改动

GAPO 不需要新的 `loss_type`，而需要一个能按 group 读取 completions 的 custom reward function：

```python
from collections import Counter


def gapo_rewards_for_group(outputs, valid_outputs):
    valid = [x for x in outputs if x in valid_outputs]
    counts = Counter(valid)
    total_valid = max(len(valid), 1)
    target = 1.0 / len(valid_outputs)

    rewards = []
    for output in outputs:
        if output not in valid_outputs:
            rewards.append(-1.0)
            continue

        frequency = counts[output] / total_valid
        rewards.append(1.0 - (frequency - target))
    return rewards
```

接入 `GRPOTrainer` 时，要保证同一 prompt 的 `num_generations` 个 completion 在 reward function 中被还原为一组，再返回与原 flattened completion 顺序一致的 reward list。

## 方法地图：这些论文分别改了什么

| 方法 | 主要修改层 | 想修复的症状 | 一句话概括 |
| --- | --- | --- | --- |
| DAPO | 训练 recipe，跨越 sampling、clip、loss 和 reward | 熵塌陷、零梯度 prompt、长回答梯度稀释、截断噪声 | 把长 CoT RL 中几个耦合问题拆开处理 |
| Dr.GRPO | advantage scaling 与 loss normalization | 回答长度偏置、题目难度偏置 | 去掉 response length 和 group std 带来的隐式重加权 |
| CISPO | importance weight 与 surrogate | hard clipping 丢掉稀有但关键 token 的梯度 | clip IS weight，但不把 token 梯度直接置零 |
| GSPO | importance ratio 粒度 | token-level ratio 噪声累积，尤其影响长序列和 MoE | reward 是 sequence-level，ratio 和 clipping 也提升到 sequence-level |
| SAPO | surrogate 的 trust region | hard clip 不连续，越界后梯度突然消失 | 用带正负温度的 soft gate 平滑衰减梯度 |
| GAPO | reward 的定义对象 | 单回答 reward 无法直接优化整组输出的多样性 | reward 从单样本函数变成 group-aware 函数 |

严格来说，这些方法不是同一条单线演化路线。DAPO 是系统 recipe，Dr.GRPO 是 bias correction，CISPO、GSPO、SAPO 主要改 optimizer，GAPO 则位于 optimizer 之前的 reward construction 层。


## 参考资料

- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- Liu et al., [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- MiniMax-AI, [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)
- Zheng et al., [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- Qwen Team, [Soft Adaptive Policy Optimization](https://arxiv.org/abs/2511.20347)
- Anschel et al., [Group-Aware Reinforcement Learning for Output Diversity in Large Language Models](https://arxiv.org/abs/2511.12596)
- Hugging Face, [TRL GRPO Trainer Documentation](https://huggingface.co/docs/trl/grpo_trainer)
- wenzhaoabc, [LLM TAP RL: GRPO Variants Notebook](https://github.com/wenzhaoabc/llm-tap-rl/blob/main/9.GRPO-Variants.ipynb)
