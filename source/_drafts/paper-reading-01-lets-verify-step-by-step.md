---
title: "论文精读 01：Let’s Verify Step by Step，过程监督到底好在哪里"
date: 2026-08-20 21:00:00
updated: 2026-08-23 23:30:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - 论文精读
  - Reward Model
  - Process Supervision
  - PRM
  - Reasoning
math: true
category_bar: true
---

《Let’s Verify Step by Step》是 OpenAI 在 2023 年放出的一篇工作。现在回头看，它讨论的问题几乎成了 reasoning model 的基础设施：一条很长的推理，到底应该只检查最终答案，还是逐步检查中间过程？

论文的答案很明确：在 MATH 上，过程监督训练出的 reward model 明显强于结果监督。但真正值得看的不只是 78.2% 这个数字，而是这篇论文把“过程监督为什么有用”拆成了一个可以做实验的问题。

<!-- more -->

## 先说清楚：这篇论文没有训练一个新的 reasoning policy

这点很容易被今天的语境带偏。

论文固定了一个 generator，让它对每道题采样很多条解答；研究者训练的是 reward model，然后用 reward model 从这些候选里挑一条最好的。整个评测过程是：

$$
x \xrightarrow{\text{generator}} \{y_1,\ldots,y_N\}
\xrightarrow{\text{reward model}} y_{\text{best}}
$$

最后再检查 $y_{\text{best}}$ 的答案是否正确。

所以这里比较的不是“ORM-RL 和 PRM-RL 谁更强”，而是：给定同一个生成模型和同一批候选答案，哪一种 reward model 更会挑。

这个边界很重要。它让实验避开了 RL 本身的不稳定性，把问题收缩到了 verifier 的质量。

## ORM 和 PRM 的区别，不只是标签数量不同

Outcome-supervised Reward Model（ORM）只看整条解答的最终结果。假设一条解答最后答对了，它就得到正标签；答错了，就得到负标签。

Process-supervised Reward Model（PRM）则在每个推理步骤后预测该步骤是否正确。人工标注有 positive、negative 和 neutral 三类，训练时模型学习在步骤边界输出对应判断。

对于一条包含 $T$ 个步骤的解答，论文把整条解答的分数定义成每一步正确概率的乘积：

$$
s_{\mathrm{PRM}}(y)=\prod_{t=1}^{T}P(\text{step}_t\text{ is correct})
$$

这个聚合方式非常严格。一条长解答里只要有一步的正确概率很低，整条解答的分数就会被拉下来。

看起来 PRM 只是多了更细的标签，但它实际改变了 credit assignment。

假设一条 15 步的解答最后错了。ORM 只告诉模型：“这 15 步里至少有哪里不对。”至于前 12 步是不是都正确、错误第一次出现在哪里，它都要自己猜。题目越难，负样本越多，这种负标签提供的新信息反而越少。

PRM 给出的信号更直接：前面这些步骤可以保留，错误从这里开始。它没有让 reward model 变得更聪明，而是把原本很难的归因任务从训练目标里拿掉了一部分。

## PRM800K 是怎么来的

论文收集了 PRM800K：约 80 万个 step-level 人工标签，覆盖 7.5 万条解答和 1.2 万道题。

标注员不是一定要把错误后的所有步骤都看完。论文只监督到第一个错误步骤为止。这样做有两个好处：

- 人工成本更可控；
- ORM 和 PRM 的信息差更干净。

对于错误解答，ORM 知道“至少有一个错误”；PRM 在此基础上多知道“第一个错误在哪里”。如果连错误后的每一步也继续标，PRM 的信息优势会更大，比较就没那么克制了。

不过数据集还有一个经常被略过的细节：PRM800K 的训练数据用了 4500 道 MATH test 问题，因此最终只在剩余 500 道题上评测。论文将它们称为具有代表性的子集，但这不是完整的 MATH test set。看到 78.2% 时，不能把它直接当成标准 MATH 全集成绩。

## 核心结果：候选越多，PRM 的优势越明显

大规模实验中的 generator 和 reward model 都从 GPT-4 base 继续训练而来。每道测试题共生成 1860 条候选解答，再比较 Best-of-N 搜索能力。

在 Best-of-1860 下：

| 方法 | 解题率 |
|---|---:|
| Majority voting | 69.6% |
| ORM | 72.4% |
| PRM | **78.2%** |

更有意思的是曲线形状。$N$ 增大时，PRM 和 ORM 的差距也在扩大。

这说明采样更多并不会自动带来更好的答案。候选池越大，里面不仅有更多正确解，也有更多“写得很像对的错误解”。如果 verifier 分辨不出来，额外的 test-time compute 只是生产了更多迷惑项。

因此 PRM 的价值和 Best-of-N 是耦合的：generator 负责提高候选解的覆盖率，PRM 负责不被那些有迷惑性的错误过程骗过。

## 为什么只看最终答案会有 false positive

数学题的最终答案通常可以自动检查，这让 outcome supervision 很便宜。但它有一个结构性问题：过程错了，答案也可能碰巧对。

比如中间把两个负号都写错，最后结果反而相消；或者用了不成立的推导，但猜中了选项。自动判题仍会给出正标签。对 ORM 来说，这些样本在教它认可一条错误推理。

论文用大 PRM 给小模型提供 synthetic supervision，做了一组更可控的比较：

- 用大 PRM 的 step label 训练小 PRM；
- 用大 PRM 对整条解答的判断训练小 ORM；
- 用最终答案检查训练小 ORM。

三者使用相同的解答集合，区别只在监督信号。结果仍然是过程监督最好；而由大 PRM 提供 outcome label 的 ORM，又比只看最终答案的 ORM 更好。

这说明 ORM 的问题有两层。一层是结果检查存在 false positive；另一层是即使结果标签本身更可靠，整条解答一个标签仍然让 credit assignment 太难。过程监督的优势不能完全归结为“自动判题不准”。

## Active Learning：不要把人工花在一眼就错的答案上

PRM800K 没有均匀抽样所有解答，而是优先标注 convincing wrong answers：最终答案错误，但当前 PRM 给分很高的解答。

这类样本明确暴露了 reward model 的盲点。一个从第一步就胡说的答案，虽然也是负样本，却几乎教不了当前 PRM 新东西。

论文的小规模模拟实验里，active learning 大约带来了 $2.6\times$ 的数据效率提升。大致流程是：

1. 先训练一个 selector PRM；
2. 每道题采样大量解答；
3. 找出 selector 高分但最终答案错误的解答；
4. 把有限标注预算优先给这些 hard negatives；
5. 用新数据继续训练 PRM。

这个思路比“多标一点”更重要。Reward model 的数据不应该只覆盖任务分布，还应该覆盖 reward model 当前最容易被骗的区域。

论文也报告了一个不那么漂亮的结果：迭代重训 selector 的实验出现不稳定，并没有得到进一步收益。作者认为迭代 active learning 应该有用，但没有用实验把这个猜想坐实。这种没有被整理掉的失败结果反而很有价值。

## OOD 结果说明了什么，又没有说明什么

论文还在较新的 AP Calculus、AP Chemistry、AP Physics 和 AMC 题目上做了分布外测试。Best-of-100 的汇总成绩里，PRM 为 72.9%，ORM 为 63.8%，多数投票为 61.3%。

它至少说明 PRM 学到的不只是 MATH 题目的固定答案模式，在相近的 STEM 推理分布上仍然能工作。

但范围也只到这里。数学步骤通常边界清楚，正确性也相对客观。在开放式研究、写作、规划或真实 agent 任务中，“这一步正确吗”可能根本没有唯一标签。把数学 PRM 的结论直接推广到所有 chain-of-thought，跨度太大。

## 我觉得这篇论文真正留下了三件事

第一，verifier 的训练粒度决定了 test-time compute 能不能兑现。采样规模上去以后，瓶颈会从“有没有生成正确答案”转向“能不能把它找出来”。

第二，过程监督的主要收益可以用 credit assignment 来解释。它不是神秘地鼓励了“更好的思维”，而是把错误位置明确告诉模型。

第三，reward model 的数据选择本身也是算法的一部分。专门收集高分错解，本质上是在做针对 verifier 的 adversarial data collection。

今天的 reasoning 系统已经不只用 Best-of-N：PRM 可以参与搜索、reranking、训练数据过滤，也可以给 RL 提供更密的 reward。但这些后续用法都建立在同一个问题上：我们究竟是在奖励一个结果，还是在奖励通向结果的过程？

《Let’s Verify Step by Step》没有终结这个问题，但它把讨论从直觉推进到了一个很扎实的实验基线。

## 参考资料

- Lightman et al., [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)
- OpenAI, [PRM800K Dataset](https://github.com/openai/prm800k)
- Uesato et al., [Solving Math Word Problems With Process- and Outcome-Based Feedback](https://arxiv.org/abs/2211.14275)

