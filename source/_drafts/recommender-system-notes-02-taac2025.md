---
title: "推荐系统笔记 02：TAAC2025 解读，生成式推荐与我的上分复盘"
date: 2025-09-18 20:00:00
updated: 2025-09-18 20:00:00
categories:
  - 推荐系统笔记
tags:
  - 推荐系统
  - 腾讯广告算法大赛
  - 生成式推荐
  - 笔记
math: true
category_bar: true
---

这一篇整理 TAAC2025。

TAAC2025 最有意思的地方在于，它不是传统 CTR / CVR 排序题，而是一个生成式推荐题。

传统广告推荐里，模型常见输入是：

```text
user features + item features + context features
-> score
```

模型输出一个点击率、转化率或者排序分数。

但 TAAC2025 更接近：

```text
user behavior sequence
-> generate next item / semantic id
```

也就是说，它不是先给你一堆候选 item 再让你排序，而是让模型直接从用户历史行为里生成目标。这个变化很大，因为它把推荐问题从“打分”推向了“序列生成”。

## 1. 赛题到底在考什么

官方论文把 TAAC2025 定位为 all-modality generative recommendation。

我理解这里有三个关键词：all-modality、sequence、generative。

**All-modality** 说明 item 不只有一个 ID，还可能有多模态内容表示、协同信号、广告侧信息等。模型不能只靠一个 item_id embedding 记忆交互。

**Sequence** 说明用户行为历史是核心输入。用户不是一个静态画像，而是一串随时间变化的行为。

**Generative** 说明目标不是对候选 item 打分，而是生成目标 item 的离散表示。

官方数据里有 TencentGR-1M 和 TencentGR-10M 两个规模。前者提供百万级用户序列，后者扩展到千万级用户，并且更明确地区分 click 和 conversion。这个设计很像在提醒参赛者：

> 推荐系统不能只学“点了什么”，还要学“什么行为更有价值”。

这和普通序列推荐数据集不一样。MovieLens 这种数据更多是用户-item 评分或交互；TAAC2025 的广告场景里，click、conversion、多模态表示、weighted evaluation 都会影响模型该学什么。

## 2. 为什么 baseline 会选 SASRec

SASRec 是一个很自然的 baseline。

它的基本想法是：用 self-attention 建模用户行为序列，根据历史行为预测下一个 item。

它介于 Markov Chain 和 RNN 之间：

- Markov Chain 更偏最近几步行为，简单但表达能力有限。
- RNN 能建模长程信息，但训练和并行效率不如 Transformer。
- SASRec 用 self-attention 从历史里挑相关行为，既能看长期信息，也能更关注关键行为。

在推荐里，这个假设很合理。

比如一个用户历史里有很多行为：

- 最近看了登山鞋
- 上周看了冲锋衣
- 三个月前看了键盘
- 半年前买过耳机

如果现在要预测下一个 outdoor item，模型不应该平均看所有历史。它应该更关注登山鞋、冲锋衣这类相关行为。

SASRec 的强点是结构直接，训练稳定，能作为生成式推荐 baseline 的序列编码器。

但它也有明显限制：

- 序列长了以后，标准 attention 成本会变高。
- 它主要建模 item 序列，对多模态 item 表示和 semantic ID 的处理不是原生设计。
- 它更像“序列预测模型”，不是完整的 generative recommender framework。

这就是后面 TIGER / HSTU 这类工作有价值的地方。

## 3. TIGER：为什么要用 Semantic ID

生成式推荐有个核心问题：

> item 到底应该怎么被模型生成？

最直接的做法是把每个 item 当成一个 token。

但这会遇到几个问题：

- item 数量巨大，词表很大。
- 新 item 冷启动困难。
- item_id 本身没有语义，相似 item 的 ID 可能完全不相关。
- 直接生成 item_id 很像在记忆，而不是在泛化。

TIGER / Generative Retrieval 这条线的关键思想是 Semantic ID。

它不直接生成原始 item_id，而是先把 item 映射成一串离散 code：

```text
item
-> semantic representation
-> discrete codewords
-> semantic id sequence
```

然后模型生成 semantic id sequence。

这样做的好处是：相似 item 可能拥有相近的 code 路径，模型更容易泛化到冷启动或低频 item。

举个简单例子。

如果两个运动鞋 item 是完全不同的原始 ID，模型很难知道它们相似。但如果它们的 semantic id 前几级都落在“运动鞋 / 户外 / 男款”附近，模型就能利用这种结构。

当然，Semantic ID 也不是白送的。

它会引入新的问题：

- tokenizer 怎么训练？
- 离散 code 是否真的反映推荐语义？
- 不同 item 是否会撞到同一个 semantic id？
- tokenizer 的目标和推荐目标是否一致？

后面很多 SID 论文其实都在围绕这些问题展开。

## 4. SID 的几个坑

Semantic ID 看起来像是生成式推荐的关键接口，但这个接口很容易出问题。

第一个坑是 **ID collision**。

如果多个 item 被映射到同一个 SID，那么模型生成这个 SID 时，到底推荐的是哪个 item？如果评估时只看 SID 命中，就可能高估真实性能。

第二个坑是 **tokenizer objective mismatch**。

很多 SID 是先用内容 embedding 或多模态 embedding 训练 tokenizer，再拿生成模型学推荐。问题是 tokenizer 可能优化的是内容重建，不一定优化推荐效果。

一个商品从内容上看很像另一个商品，但用户行为上可能完全不同。比如两个外观相似的广告素材，一个转化很好，一个转化很差。如果 SID 只看内容，它可能把这两个 item 放得太近。

第三个坑是 **codebook collapse**。

如果离散 code 使用不均匀，模型可能只使用少数 code，SID 空间就没有真正表达 item 差异。

所以写 TAAC2025 不能只说“SID 很好”。更准确的理解是：

> SID 把 item 生成问题变得可学，但它同时把推荐系统的难点转移到了 item tokenization 上。

## 5. HSTU：生成式推荐为什么要 scale

HSTU 来自 Meta 的 Generative Recommenders 路线。

这篇工作对我理解 TAAC2025 很重要，因为它把推荐系统从“特征工程 + 小模型排序”的视角，推到了“长序列生成模型能不能像 LLM 一样 scale”的视角。

HSTU 的出发点是：工业推荐数据有几个特点：

- 用户行为序列很长。
- item / action 高基数。
- 分布非平稳，今天的兴趣和明天可能不同。
- 数据量极大，但传统 DLRM 不一定能随 compute 稳定提升。

HSTU 尝试把推荐问题改写成 sequential transduction，并设计适合推荐场景的结构。论文里很强调两个点：

- 长序列建模要高效。
- 推荐模型也可能存在 scaling law。

对比赛来说，HSTU 的价值不只是“换一个 backbone”。更重要的是它提醒我：

> 这个任务里，序列建模能力和模型容量可能真的能带来上限，而不是只靠调 loss 或特征工程。

这也对应了我后面上分过程里 HSTU + scale up 的跃迁。

## 6. 我最开始踩的坑：InfoNCE 写错了

我一开始并不是直接靠 HSTU 上分。

最早 baseline 大概在 0.22。这个阶段主要是确认数据处理、模型输入、训练流程和提交格式能跑通。

完整的上分路径可以先放在这里：

| 阶段 | 分数记录 | 主要变化 |
| --- | ---: | --- |
| baseline | 0.22 | 跑通官方 baseline 和提交流程 |
| 错误的 InfoNCE | 0.29 | contrastive 方向有信号，但目标实现不对 |
| 错误的 InfoNCE + triplet loss | 0.32 | 强行拉开正负距离，有收益但不干净 |
| 正确的 InfoNCE | 0.49 | 负样本、mask、目标定义对齐后明显提升 |
| 加入时间特征 | 0.58 | 序列里的 recency 信息开始发挥作用 |
| 更多时间特征处理 | 0.64 | 时间不只是 bucket，而是兴趣状态的一部分 |
| HSTU + scale up | 0.78 | 结构容量和长序列建模开始带来上限 |
| temperature 降到 0.03 | 0.82 | contrastive 分布更尖锐，区分更强 |
| 混合精度 + hidden 维度到 256 | 0.88 | 显存释放后扩大 hidden 表达能力 |
| 加入 SENet | 0.92 | 特征通道重标定带来收益 |
| embedding 增加到 512 | 0.95 | item / SID / 行为表示容量继续提高 |
| 最后一发：batch size scale up + 冷启动优化 | 预计 0.105+ | 更大的负样本比较空间，加上长尾 item 泛化修正 |

最后两项的数字我先按原始记录保留。正式发布前最好再核一下小数位，避免读者误解。

之后我尝试 InfoNCE。

InfoNCE 的直觉很适合这个任务：让用户序列表征靠近正确目标，远离负样本。

可以粗略理解成：

$$
L = -\log \frac{\exp(s(q, k^+) / \tau)}{\exp(s(q, k^+) / \tau) + \sum_j \exp(s(q, k_j^-) / \tau)}
$$

其中 $q$ 是用户侧表示，$k^+$ 是正样本 item 表示，$k_j^-$ 是负样本，$\tau$ 是 temperature。

但这里最容易错的就是：正负样本到底怎么定义，logits 的维度到底对应什么，mask 有没有写对，batch 内负样本是否真的合法。

我一开始的错误 InfoNCE 也能涨到 0.29，说明 contrastive 方向有信号，但目标没有完全对齐。

再加 triplet loss 到 0.32，说明 margin-based 拉开正负距离也有一点帮助。但这个阶段的问题是：loss 越加越复杂，不代表任务理解更正确。

真正的跳跃来自把 InfoNCE 写对。

正确 InfoNCE 到 0.49，说明这道题最重要的不是“堆一个更复杂的损失”，而是先让训练目标和评估目标对齐。

这个经验很像推荐系统里的很多坑：指标上不去时，不一定是模型太弱，可能是样本构造和目标定义错了。

## 7. 时间特征为什么这么关键

加入时间特征后，分数从 0.49 到 0.58。

更多时间特征处理后，又到 0.64。

这说明在 TAAC2025 里，时间不是一个边缘特征，而是用户兴趣状态的一部分。

一个行为发生在 5 分钟前，和发生在 3 个月前，含义完全不一样。

比如用户历史里都有“看过某类广告”：

- 如果是刚刚看过，可能代表当前意图。
- 如果是很久以前看过，可能只是长期偏好。
- 如果某类行为突然密集出现，可能代表一个 session 或短期需求。

所以时间特征至少有三层：

- event-level time：每个历史行为距离当前多久。
- sequence-level rhythm：用户近期行为是否密集，是否有 burst。
- request-level time：当前请求发生在一天/一周/训练窗口中的什么位置。

一开始我只是把时间当成额外 embedding。后来处理得更细以后，模型才更容易区分“长期兴趣”和“当前意图”。

这也解释了为什么 HSTU / TiSASRec / DSIN 这类工作会强调 time interval、session、long sequence。推荐系统里的序列不是普通 token 序列，它有真实时间间隔。

## 8. HSTU + scale up 的跃迁

HSTU + scale up 到 0.78，是这条路径里第二个明显跃迁。

这一步说明前面的时间特征和 contrastive objective 只是把训练信号对齐了，但模型本身还需要足够强的序列建模能力。

SASRec baseline 可以作为起点，但如果用户历史很长、行为模式复杂，普通 self-attention 或浅层结构很难吃下所有信息。

HSTU 的价值在于：

- 更适合长行为序列。
- 更强调推荐数据的非平稳性。
- 能在更大模型容量下继续获益。

但 scale up 不是简单把 hidden dim 变大。

推荐系统里 scale up 很容易遇到：

- embedding 过拟合。
- 负样本分布不稳定。
- batch size 太小导致 contrastive 学不到足够比较。
- temperature 不合适导致分布太软或太尖。
- 冷启动 item 被高频 item 压制。

所以 HSTU + scale up 更像是打开了容量上限，但后面还要靠训练细节把容量用起来。

## 9. Temperature 为什么能从 0.78 到 0.82

把 temperature 降到 0.03 后到 0.82。

这一步看起来只是一个超参，但其实反映了 contrastive learning 的核心。

temperature 控制 softmax 分布的锐度。

temperature 大时，模型对正负样本的区分更平滑；temperature 小时，模型会更强烈地关注最难区分的负样本。

在推荐任务里，如果 item 数量大、负样本多，分布太软可能会导致模型没有足够动力把相似但错误的 item 拉开。

但 temperature 也不能无限小。太小会让训练不稳定，尤其是 batch 内存在 false negative 时，模型会过度惩罚本来可能相关的 item。

所以 0.03 有效，说明当时模型已经有足够强的表示能力，可以承受更尖锐的对比学习目标。

## 10. 容量、混合精度和 SENet

混合精度让 hidden 维度加到 256 后，分数继续上升。

这一步很现实：很多时候模型不是不想 scale，而是显存不允许。混合精度释放了显存，让 hidden dim、batch size 或 sequence length 有空间往上走。

增加 hidden dim 的收益说明表征容量仍然不足。

加入 SENet 到 0.92，则说明特征重标定有价值。

SENet 的直觉是：不同特征通道的重要性不一样，模型应该根据当前样本动态调整通道权重。

在推荐里这很自然。

同一个用户序列中，有些 item 或特征对当前预测很重要，有些只是噪声。SENet 这类 gating / recalibration 模块能帮模型压低无关信号，突出有效特征。

这一步让我感觉，推荐系统里的“模型结构”不一定总是大 Transformer。有些轻量模块只要插在合适位置，也能明显改善特征利用。

## 11. Embedding 到 512 和最后一发

增加 embedding 到 512 后继续涨，说明 item / SID / 行为表示本身的容量仍然重要。

但 embedding 变大也会带来风险：

- 高频 item 更容易被记住。
- 低频 item 更容易学不好。
- 冷启动 item 更容易被已有交互强的 item 压制。
- 多 epoch 后 embedding 可能比 backbone 更先过拟合。

最后一发的方向是 scale up batch size + 冷启动优化。

我觉得这两个点是连在一起的。

batch size 变大以后，InfoNCE 里的负样本更多，模型能看到更丰富的比较关系；冷启动优化则是为了避免模型只会推荐训练里出现频繁、交互充分的 item。

在生成式推荐里，冷启动尤其重要。因为如果 SID 或 item embedding 只依赖历史交互，低频 item 很难有好表示。all-modality 信息和 semantic ID 的意义就在这里：它们给冷启动 item 提供了除交互以外的语义入口。

这也是 TAAC2025 最像研究问题的地方：

> 模型不仅要记住历史行为，还要学会在长尾 item 和新 item 上泛化。

## 12. 这次比赛给我的几个判断

第一，生成式推荐不是简单把推荐系统改成 LLM。

它真正改变的是 item 表示和目标形式。item 要变成可生成的离散 token，用户行为要变成序列上下文，训练目标要能区分正确 target 和大量相似负样本。

第二，Semantic ID 是关键接口，但也是风险接口。

SID 好，模型更容易泛化；SID 差，模型会被错误的 item 分组拖累。特别是 collision、tokenizer objective mismatch、codebook 使用不均匀，都可能让离线指标虚高或泛化变差。

第三，时间特征不是锦上添花。

推荐里的序列和 NLP token 序列不同。行为之间有真实时间间隔，用户兴趣会衰减、跳转、爆发。没有时间，模型很容易把“长期偏好”和“当前意图”混在一起。

第四，scale up 有用，但前提是目标、负样本和训练稳定性先对齐。

如果 InfoNCE 写错，模型再大也只是在优化错误目标。如果 temperature 不合适，表示空间也学不出足够清晰的边界。如果冷启动没处理，模型会更偏向高频 item。

所以我的理解是：

> TAAC2025 的上分不是某一个 trick，而是把生成式推荐里的四件事逐步对齐：目标、时间、结构、容量。

后面看 TAAC2026 时，问题会变得不一样。TAAC2026 更偏 PCVR / 工业排序模型，重点从“生成目标 item”转向“如何把 user、item、sequence、time 这些异构特征编码成一个强排序模型”。

## 参考

- [Tencent Advertising Algorithm Challenge 2025: All-Modality Generative Recommendation](https://arxiv.org/abs/2604.04976)
- [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)
