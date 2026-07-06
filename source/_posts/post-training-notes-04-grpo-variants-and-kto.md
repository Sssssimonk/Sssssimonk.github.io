---
title: "后训练笔记 04：GRPO变体与DPO变体"
date: 2025-11-22 21:30:00
updated: 2025-11-22 21:30:00
categories:
  - 后训练笔记
tags:
  - 后训练
  - GRPO
  - DAPO
  - GSPO
  - KTO
  - 偏好优化
  - 笔记
math: true
category_bar: true
---

这一篇整理 GRPO 的变体，以及DPO的变体。



所以这一篇分成两条线：

- GRPO 变体：围绕 RLVR 里的 group sampling、advantage、clipping、sequence-level 更新做改造
- KTO：围绕 preference data 的形式做改造，从 pairwise preference 变成 binary feedback

## 1. GRPO 的基本问题

GRPO 的基础想法很清楚：同一个prompt采样多个 responses，用组内 reward 相对值估计 advantage。

但真实训练里会遇到几个问题。

第一，组内 reward 没有差异。

如果一组全对或全错，标准差接近 0，训练信号很弱。

第二，reward 太稀疏。

数学题只有最终答案对错，代码题只有测试通过与否。中间推理过程没有细粒度反馈。

第三，token-level clipping 可能浪费训练信号。

有些 token 的 ratio 被 clip 后，梯度贡献变小甚至消失。

第四，长度会膨胀。

模型可能生成越来越长的 reasoning，因为长输出更容易覆盖更多尝试，也可能更容易骗过某些 reward。

第五，MoE 模型 RL 更不稳定。

RL 更新会影响 routing、专家激活和序列分布，sequence-level 稳定性变得更重要。

GRPO 变体基本都在修这些问题。

## 2. Dr.GRPO：重新看 normalization

GRPO 的 advantage 常写成：

$$
A_i = \frac{r_i - \mu}{\sigma}
$$

其中 $\mu$ 和 $\sigma$ 是同一组 responses 的 reward 均值和标准差。

这个标准化看起来很自然，但它也会改变训练信号强度。

如果组内 reward 差异很小，$\sigma$ 很小，更新可能被放大或不稳定。

如果组内全同，$\sigma$ 接近 0，训练信号又会失效。

Dr.GRPO 这类方法的核心思路，是重新审视 group normalization 是否真的合理，尤其是在二元 reward 场景里。

二元 reward 下，一组 responses 的 reward 只可能是 0 或 1。此时标准差本质上反映的是这个 prompt 下样本是否有分歧。

如果全对或全错，没有分歧，就没有可学习的相对信号。

如果一半对一半错，分歧最大，训练信号最强。

这说明 GRPO 的学习重点其实被 group disagreement 控制了。

## 3. DAPO：dynamic sampling 和 clipping

DAPO 可以理解成对 GRPO 训练细节的进一步修补。

它关注两个问题：

- 哪些 prompts / groups 值得训练
- policy ratio 的 clipping 怎样更合理

如果一个 group 全对或全错，继续训练价值低。Dynamic sampling 会更关注那些能产生有效差异的样本。

这和人刷题有点像。

太简单的题已经都会，太难的题全不会，最有训练价值的是“有时会、有时不会”的题。

对于 LLM RLVR，也是这样。能让同一个 prompt 下的 samples 产生分歧，才有足够的相对学习信号。

Clipping 的问题则是：固定 clip range 可能不适合所有 token、所有阶段。模型训练初期和后期需要的更新幅度不同，高置信 token 和低置信 token 也不同。

所以 DAPO 这类方法会尝试让 sampling 和 clipping 更动态。

## 4. GSPO：从 token-level 到 sequence-level

GSPO 是 Group Sequence Policy Optimization。

它的一个重要变化是把优化从 token-level ratio 推向 sequence-level ratio。

在语言模型里，reward 很多时候是 response-level 的。数学答案对不对，是整段 response 的结果；代码能不能过测试，也是整段 response 的结果。

如果 reward 是 sequence-level，但优化时过度依赖 token-level ratio，就可能出现不匹配。

GSPO 的想法是：既然 reward 是整段序列的结果，就直接在 sequence level 处理 importance ratio、clipping 和优化。

这对 MoE 模型尤其重要。因为 MoE 的 token-level routing 更复杂，token-level 更新可能带来不稳定。Sequence-level objective 更贴近最终任务评价，也可能更稳定。

可以这样理解：

> GRPO 问的是“这组回答里哪个更好”；GSPO 进一步问“整段回答作为一个整体，应该怎样被加强或削弱”。

## 5. DCPO：dynamic clipping

DCPO 关注的是 GRPO 里的 zero gradient 和 clipping 问题。

在 PPO/GRPO 类目标里，clipping 是为了防止 policy 更新过猛。

但 clipping 太强会损失训练信号。

如果很多 token 的 ratio 都被 clip，模型即使生成了有价值的 response，也不能有效学习。

DCPO 的思路是动态调整 clipping 策略，让不同 token 或不同训练阶段有更合适的更新空间。

它还关注 reward standardization 的问题，希望提升 response-level 样本利用率。

核心不是“clip 越少越好”，而是：

> 该限制的地方限制，该学习的地方不要过早截断。

## 6. DRA-GRPO：把 diversity 纳入 reward

GRPO 采样一组 responses，但如果这些 responses 高度相似，组内比较的价值有限。

DRA-GRPO 这类方法关注 diversity。

它希望不同 reasoning paths 不要因为最终 reward 相同就被完全等价对待。

比如两条解法都错，但一条接近正确思路，另一条完全胡乱推理；或者两条都对，但一条更简洁，另一条绕远路。单纯 0/1 reward 分不出来。

Diversity-aware reward adjustment 试图让训练信号不仅看正确性，也看候选之间的语义差异和探索价值。

这对 reasoning 很重要。因为模型不是只需要记住一个答案，而是要学习更可靠的解题路径。

## 7. GRPO 变体的共同主线

这些方法名字很多，但不要被缩写带跑。

它们基本在改四件事。

第一，采样。

哪些 prompts 要采，采几个 responses，如何过滤全对/全错。

第二，advantage。

组内 reward 怎么归一化，是否使用标准差，是否引入 difficulty / diversity。

第三，clipping。

policy ratio 怎么限制，token-level 还是 sequence-level，固定还是动态。

第四，reward。

只看最终答案，还是加入格式、长度、过程、diversity、verifier confidence。

理解这四个维度，比记住每个方法缩写更有用。

## 8. KTO：为什么不是 DPO 的简单替代

DPO 需要 preference pair：

$$
(x, y_w, y_l)
$$

也就是同一个 prompt 下，一个 chosen response，一个 rejected response。

但真实产品数据经常不是这样。

用户可能只是点了赞或踩：

$$
(x, y, \mathrm{desirable})
$$

或者：

$$
(x, y, \mathrm{undesirable})
$$

没有配对 rejected response。

KTO 的动机就是：能不能直接用这种 binary feedback 做 alignment？

KTO 全称是 Kahneman-Tversky Optimization，借用了 prospect theory 里的损失厌恶、参考点等想法。

## 9. KTO 的 implied reward

KTO 也使用 policy 和 reference 的 log-ratio 来定义 implied reward：

$$
r_\theta(x,y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
$$

这个量表示：相对于 reference model，当前 policy 对这个 response 的偏好提高了多少。

如果 $y$ 是 desirable，应该提高它的 implied reward。

如果 $y$ 是 undesirable，应该降低它的 implied reward。

和 DPO 不同，KTO 不需要同一个 prompt 下同时有 chosen 和 rejected。它只需要知道某个 response 是好还是坏。

这在真实数据里更容易获得。

## 10. KTO 的参考点

KTO 不只是把好样本推高、坏样本压低。

它引入 reference point，类似 prospect theory 里人们不是看绝对收益，而是看相对某个参考点的 gain/loss。

可以粗略理解成：

> 对 desirable response，希望它比参考点更好；对 undesirable response，希望它比参考点更差。

这种设计让 KTO 更适合不成对的二元反馈。

但它也带来新问题：desirable 和 undesirable 数据分布是否匹配？好样本和坏样本是否来自相同类型 prompt？类别比例是否平衡？

如果点赞数据和点踩数据分布差很多，模型可能学到数据分布偏差，而不是偏好本身。

## 11. DPO 和 KTO 的区别

DPO 学的是 pairwise preference。

它看的是：

> 对同一个 prompt，$y_w$ 应该比 $y_l$ 更可能。

KTO 学的是 binary desirability。

它看的是：

> 这个 prompt-response pair 是 desirable 还是 undesirable。

DPO 的信号更强，因为同一个 prompt 下直接比较两个 responses。

KTO 的数据更容易收集，因为不需要配对。

所以两者不是谁完全替代谁，而是数据形态不同。

如果有高质量 pairwise preference，DPO 很自然。

如果只有线上 thumbs up / thumbs down，KTO 更合适。

## 12. KTO 和 GRPO 的区别

KTO 不是 RLVR。

它不需要对同一个 prompt 采样一组 responses，也不依赖 verifier 给 reward。

它更接近 DPO 的 direct alignment。

GRPO 适合数学、代码这种可验证任务。

KTO 适合真实用户反馈、偏好稀疏、难以构造 pairwise comparison 的场景。

所以不要把 KTO 放进 GRPO 变体里。它应该放在 direct preference optimization 这条线。

## 13. 读 GRPO 变体论文时看什么

我会看这些问题：

- 它改的是 sampling、advantage、clipping，还是 reward？
- 是否仍然需要 group sampling？
- 是否处理全对/全错 group？
- 是否从 token-level 改到 sequence-level？
- 是否控制 length inflation？
- 是否在 greedy decoding 下提升，而不是只在多采样下提升？
- 对 MoE 模型是否稳定？
- 训练成本是否增加？

很多变体会在 benchmark 上提升，但可能用了更复杂的采样、筛选、reward shaping。要看清楚收益来自算法本身，还是额外工程。

## 14. 读 KTO 论文时看什么

我会看这些问题：

- 数据是 pairwise preference 还是 binary feedback？
- desirable / undesirable 是否平衡？
- 好坏样本是否来自同一 prompt 分布？
- Reference model 是什么？
- 是否和 DPO 在同等数据条件下比较？
- 是否对真实线上反馈更有优势？
- 是否出现过度压制某类回答的问题？

KTO 的价值不只是 loss 公式，而是它放宽了数据要求。

在真实系统里，二元反馈通常比 pairwise preference 更容易拿到。

## 15. 小结

GRPO 变体主要在修 RLVR 训练信号的问题：采样、advantage、clipping、sequence-level 稳定性。

KTO 则是在修偏好数据形态的问题：从成对偏好变成二元反馈。

我的理解是：

> GRPO 变体关心“怎么从一组 sampled responses 里学得更稳”；KTO 关心“没有 chosen/rejected pair 时，怎么从好/坏反馈里学”。

这两条线都属于后训练，但不要混成一类。

## 参考资料

- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- Zheng et al., [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- Yang et al., [DCPO: Dynamic Clipping Policy Optimization](https://arxiv.org/abs/2509.02333)
- Chen et al., [DRA-GRPO: Exploring Diversity-Aware Reward Adjustment](https://arxiv.org/abs/2505.09655)
- Ethayarajh et al., [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
