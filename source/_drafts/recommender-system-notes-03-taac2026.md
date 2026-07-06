---
title: "推荐系统笔记 03：TAAC2026 解读，baseline 到工业排序模型"
date: 2026-05-19 20:00:00
updated: 2026-05-19 20:00:00
categories:
  - 推荐系统笔记
tags:
  - 推荐系统
  - TAAC2026
  - PCVR
  - RankMixer
  - HyFormer
  - TokenMixer
  - 时间特征
  - 笔记
math: true
category_bar: true
---

TAAC2026赛后思考，内容穿插了我比赛时记录的一些笔记和思考。

和 TAAC2025 不一样，TAAC2026 给我的感觉更像一个工业广告排序题。它不是让模型生成下一个 item，而是要在复杂的 user / item / sequence 特征上预测转化相关目标。

所以这篇的主线不是生成式推荐，而是：

> 怎么把异构特征、长行为序列、时间信息和工业排序模型结构结合起来。


## 1. 赛题数据：不是一张普通表

TAAC2026 的数据不是简单的 `(user, item, label)` 三元组。

从 baseline 数据部分看，每一行大致对应一个样本，里面有三类特征：

- user feature
- item feature
- sequence domain feature

sequence 又分成 A/B/C/D 四个 domain。

每个 domain 内部不是一个单独字段，而是一组按时间排列的行为序列。比如 `domain_a_seq_38`、`domain_a_seq_39`、`domain_a_seq_40` 这些字段共享同一个序列长度，可以理解为：

```text
domain A:
  time step 1: side info 38, 39, 40, ...
  time step 2: side info 38, 39, 40, ...
  ...
```

这里最重要的细节是：序列是倒序的。

也就是说，靠前的位置距离当前样本更近，靠后的位置更久远。

这点很容易影响建模。如果你把它当成普通正序序列，然后随手加 causal mask，可能就会把时间方向搞反。

## 2. schema.json：特征不是只看列名

baseline 里 `schema.json` 记录了每类特征的结构。

一个 int 特征不一定只有一维。有些特征是 variable-length list，需要 padding。

比如某个 item int 特征可能是：

```text
item_int_feats_11:
  vocab_size = ...
  dim = 20
```

这说明它不是单个 category，而是最多长度为 20 的 list feature。缺失或不足长度的部分需要补 0。

schema 的作用主要是：

- 知道有哪些特征列。
- 知道每个 sparse 特征的 vocab size。
- 知道每个特征是 scalar 还是 list。
- 知道拼接后 offset 在哪里。
- 根据 vocab size 创建 embedding。

这类比赛里，schema 往往不只是配置文件。它本身就告诉你数据的结构假设。

如果一个特征是 list，那么它可能代表多值属性、历史集合、兴趣标签或某种 side information。直接把它当成普通 scalar feature，会丢掉不少信息。

## 3. parquet row group 和本地验证

baseline 读取 parquet 时，会先拿到 row group 信息。

row group 有两个作用：

- 划分 train / valid。
- 多 worker 读取时分配数据块。

一个容易忽视的问题是：baseline 的 valid 划分是按 row group 顺序来的。

这意味着大家默认拿到的验证集可能很一致，但它未必和线上分布完全对齐。

如果本地 valid 和线上 test 的时间段、用户分布、item 分布不一致，那么本地 AUC 或 loss 的提升不一定能转成线上提升。

这也是推荐比赛里很常见的问题：

> 本地验证不是越复杂越好，关键是它能不能模拟线上分布。

如果验证集切分方式和真实测试分布差很多，模型可能会朝错误方向调参。

## 4. sequence 截断本身就是上分点

baseline 里并没有把所有序列都用上，而是通过类似 `seq_max_lens` 的配置截断每个 domain。

比如：

```text
seq_a: 256
seq_b: 256
seq_c: 512
seq_d: 512
```

这样做很现实。完整长序列太占显存，也会让训练速度变慢。

但这同时也说明一个上分点：

> 如何更有效地利用更长历史，而不是简单截断。

直接拉长 sequence length 不一定有效，因为长序列里噪声也更多。

更合理的问题是：

- 最近行为是否更重要？
- 哪些 domain 的长历史更有用？
- 能不能先压缩长序列，再让模型读取？
- 能不能用 query 从长序列里挑相关片段？
- time gap 能不能帮助模型区分近期兴趣和长期偏好？

这也是为什么 baseline 模型里会出现 sequence encoder、cross attention 和 query token。

## 5. baseline 数据处理的核心流程

baseline 的 dataset 不是普通 `getitem` 一条条读，而是更偏 batch-level 处理。

原因很简单：parquet 数据一行行读太慢。

它会：

- 找到 parquet 文件和 row group。
- 用 `iter_batches` 一次读一个 batch。
- 根据 schema 创建 numpy buffer。
- 把 user / item / sequence 特征填到固定 shape。
- 对 OOB 值做 clip 或置 0。
- 对 sequence timestamp 做 time bucket。
- 最后返回可以输入模型的 tensor dict。

这里面有两个细节值得注意。

第一，variable-length int / float feature 的 padding 很关键。list feature 如果处理错，embedding pooling 就会错。

第二，time bucket 是数据侧就算好的。模型看到的不只是 sequence content token，还会看到每个历史行为距离当前样本的时间差分桶。

这说明 TAAC2026 的 baseline 已经默认承认：

> 时间是序列建模的一部分，不是可有可无的额外特征。

## 6. 为什么需要 NS Tokenizer

模型部分第一个关键模块是 NS Tokenizer。

NS 是 non-sequence，也就是 user / item 这类非序列特征。

这些特征是异构的：

- user int 可能是用户属性、兴趣标签、历史统计。
- item int 可能是广告、行业、类目、素材属性。
- user dense 可能是预训练 embedding 或统计向量。
- item dense 在当前 schema 里可能为空或较少。

如果把这些特征全部 concat 后直接丢进一个 MLP，强特征可能会压制弱特征，不同语义也会混在一起。

所以 baseline 会先把它们 token 化。

Token 在这里不是语言模型里的词，而是一个固定维度向量：

```text
一组特征
-> embedding / pooling / projection
-> 一个 d_model 维 token
```

这样模型后面就可以在 token 维度上做交互。

## 7. GroupTokenizer 和 RankMixer Tokenizer

baseline 里有两种思路。

**GroupTokenizer** 是按语义组切分。

比如把 user 特征分成 7 组，每组变成一个 token；item 特征分成 4 组，每组变成一个 token。

它的好处是可解释，token 语义更清楚。

问题是分组是人为固定的。分错了就会限制模型，token 数量也受 feature group 限制。

**RankMixer Tokenizer** 更像自动切分。

它先把所有 sparse feature embedding 拼起来，然后按指定 token 数切 chunk，每个 chunk 再投影到 `d_model`。

例如：

```text
all embeddings concat -> 768 dim
split into 64 chunks -> each chunk 12 dim
chunk projection -> 64 NS tokens
```

这个方法的优点是 token 数可控，也更适配后面的 TokenMixing。

缺点是 chunk 可能切开原始 feature embedding，token 语义不一定清楚。

所以这里有一个 trade-off：

> GroupTokenizer 更可解释，RankMixer Tokenizer 更灵活、更适合 scaling。

## 8. Seq Tokenizer：序列每个时间步变成 token

sequence feature 的 tokenization 更直接。

每个 domain 的每个时间步都有多个 side info。baseline 会给每个 side info 建 embedding，然后拼接或投影成一个 sequence token。

可以理解为：

```text
domain d, step t:
  side info embeddings
  + time bucket embedding
  -> seq token
```

这样每个 domain 会得到一个 `[L, D]` 的 token 序列。

这里有一个问题：不同 side info 的角色其实并不一样。

有些像 item id，有些像 action type，有些像统计值，有些像类目或标签。如果全部 embedding 后简单相加或拼接，模型要自己学会区分角色。

这也是后面相关论文里 role-aware sequence embedding、semantic sequence embedder 这些设计有价值的地方。

## 9. Query Generator：用 NS token 去问序列

baseline 里有一个关键模块：SeqQueryGenerator。

它的作用是生成 q token。

我理解它是在做一件事：

> 根据当前 user / item / context，生成几个“问题”，然后去历史序列里找答案。

它通常会把 NS tokens 展平，再和 sequence 的 pooled summary 拼接，然后生成每个 domain 的 query tokens。

比如：

```text
NS tokens + sequence summary
-> q tokens for domain A
-> q tokens for domain B
-> q tokens for domain C
-> q tokens for domain D
```

为什么不直接 mean pool sequence？

因为历史行为里不是所有 token 都重要。query token 的作用就是让模型有机会做 target-aware / context-aware 的读取。

但 baseline 的 query generation 也有弱点：如果 sequence summary 只是 mean pooling，那么 query 一开始拿到的信息已经很粗。

这也是一个很自然的优化点：

- mean + recency pooling
- recent top-k pooling
- domain-specific summary
- time-aware summary
- target-aware summary

## 10. Cross Attention：q token 从序列里读信息

有了 q token 后，模型会用 cross attention 读取 sequence。

形式上是：

```text
Q: q tokens
K/V: encoded sequence tokens
-> decoded q tokens
```

直觉是：每个 q token 从历史序列里抽取一个兴趣子空间。

比如：

- 一个 query 关注近期高意图行为。
- 一个 query 关注长期偏好。
- 一个 query 关注某个 domain 里的强相关 side info。

这比把 sequence 压成一个向量更灵活。

但 cross attention 的效果取决于两个东西：

- q token 是否生成得好。
- sequence token 是否编码得好。

如果 query 很粗，或者 sequence token 里时间、side info、domain 角色都混在一起，那么 cross attention 也只能在噪声里找信号。

## 11. TokenMixing：便宜的跨 token 交互

cross attention 之后，baseline 会把 decoded q tokens 和 NS tokens 拼起来，再做 RankMixer-style TokenMixing。

TokenMixing 的核心操作是无参的。

假设输入是 `[B, T, D]`，其中 `T` 是 token 数，`D` 是 hidden size。它会把 `D` 切成和 token 数相关的子空间，然后交换 token 维和子空间维，让不同 token 的信息流动。

直觉上：

```text
原始 tokens:
  token_1 = [a1, a2, a3, ...]
  token_2 = [b1, b2, b3, ...]
  token_3 = [c1, c2, c3, ...]

mix 后:
  mixed_1 = [a1, b1, c1, ...]
  mixed_2 = [a2, b2, c2, ...]
  mixed_3 = [a3, b3, c3, ...]
```

这个操作几乎不增加参数，比 self-attention 便宜很多。

但它也有两个明显问题。

第一，它要求维度和 token 数配合，比如 `d_model % T = 0`。这会限制 token 数和 hidden size 的组合。

第二，mix 后的 token 语义可能和原 token 不对齐。原来的第一个 token 可能是 user token，但 mix 后第一个 token 已经混入所有 token 的第一个子空间。

如果后面直接 residual add，就会有语义错位风险。

这也是 TokenMixer-Large 要提出 mixing-and-reverting、pre-norm、interval residual 的原因。

## 12. baseline 总结

把 baseline 模型抽象一下，大概是：

```text
user / item non-sequence features
-> NS Tokenizer

sequence domains A/B/C/D
-> Seq Tokenizer + time bucket
-> Seq Encoder

NS tokens + sequence summaries
-> Query Generator
-> q tokens

q tokens cross-attend sequence tokens
-> decoded q tokens

decoded q tokens + NS tokens
-> TokenMixing
-> prediction head
```

所以它不是一个普通 MLP 排序模型，而是一个 hybrid structure：

- NS tokenization 处理异构非序列特征。
- Sequence encoder 处理行为序列。
- Query / cross attention 让 NS 信息去读取序列。
- TokenMixing 做便宜的跨 token 交互。

这套结构和 HyFormer、RankMixer、TokenMixer 的思路都能对应起来。

## 13. RankMixer：为什么推荐排序开始 token 化

RankMixer 的核心目标是把工业排序模型变得更 GPU-friendly、更容易 scale。

传统排序模型里有很多手工特征交叉模块，比如 cross network、attention、expert、各种业务特征组合。

这些模块不一定适合现代 GPU 高吞吐训练。

RankMixer 的思路是：

- 先把 sparse / dense / sequence / cross feature 变成少量 feature tokens。
- 再用 cheap token mixing 交换 token 信息。
- 用 per-token FFN 给不同 token 子空间建模。

它背后的直觉是：

> 推荐特征太异构，先 token 化，再让 token 之间交换信息，比直接拼接成一个大向量更适合 scaling。

TAAC2026 baseline 的 NS Tokenizer 和 TokenMixing 很明显借鉴了这条线。

但 baseline 不是完整 RankMixer。它更多是把 RankMixer 的 tokenizer / mixing 思想接入到一个 sequence-aware backbone 里。

## 14. TokenMixer-Large：为什么固定 mixing 可能不够

TokenMixer-Large 对 RankMixer / TokenMixer 做了进一步反思。

它指出原始 TokenMixer 有几个问题：

- mixing 后 token 语义变了，但 residual 还直接加回原 token。
- token 数和 hidden size 有结构约束。
- post-norm 在深层模型里可能不稳定。
- 模型加深后梯度传播不一定好。
- shared FFN 可能不足以处理异构 token。

其中最值得注意的是 semantic misalignment。

如果 mixed token 已经不再对应原 token，那么：

```text
original token + mixed branch
```

这个 residual 的语义就不完全干净。

TokenMixer-Large 用 mixing-and-reverting 试图把 mixed representation 映射回原 token 语义空间，再做 residual。

这对 TAAC2026 很有启发。

如果 baseline scale up 后本地看起来变好，但线上不稳定，可能不是“模型大了没用”，而是 residual、norm、token mixing 这些结构细节没有支撑住 scale。

## 15. HyFormer：序列建模和特征交互不能太晚融合

HyFormer 的问题意识很接近 TAAC2026。

传统 pipeline 往往是：

```text
sequence encoder
-> compressed sequence vector
-> concat NS features
-> feature interaction
-> prediction
```

这个 late fusion 有一个问题：NS 特征很晚才影响 sequence 表示。

HyFormer 则把过程改成循环迭代：

```text
Query Generation
-> Query Decoding
-> Query Boosting
-> repeat
```

Query Decoding 是让 query 去读序列，Query Boosting 是让 decoded query 和 NS tokens 做交互。

这和 TAAC2026 baseline 的结构很像：

- QueryGenerator 生成 q tokens。
- CrossAttention 让 q tokens 读取 sequence。
- TokenMixing 让 q tokens 和 NS tokens 交互。

所以我觉得 TAAC2026 baseline 可以看成 HyFormer 思想的一个比赛版实现。

它的关键不是“用了 attention”，而是：

> sequence modeling 和 feature interaction 不应该完全分开做。

## 16. OneTrans 和 MixFormer：更统一的融合路线

OneTrans 的思路更激进。

它把 sequence tokens 和 non-sequence tokens 放进一个统一 Transformer 里，用 causal mask 控制信息流。

这解决的是 late fusion 问题：NS tokens 可以直接看完整 sequence history，sequence 和 NS 的交互在同一个 backbone 中发生。

但 OneTrans 的代价也更高。完整统一 Transformer 对长序列和多候选场景都不便宜，所以论文里会强调 pyramid stack、KV cache、mixed parameterization。

MixFormer 则更像 TAAC2026 baseline 的近亲。

它用 non-sequence features 生成多个 query heads，再让这些 heads cross-attend sequence，最后做 per-head fusion。

这个思路对比赛更现实：

- 不必把所有 tokens 放进一个大 Transformer。
- query 可以带着 user / item / context 信息读取 sequence。
- per-head / per-query FFN 可以避免不同语义子空间互相干扰。

所以如果基于 baseline 做实验，我会优先考虑 MixFormer / HyFormer 的轻量改法，而不是直接复刻完整 OneTrans。

## 17. UniMixer：固定 permutation 不是终点

UniMixer 给 RankMixer / TokenMixer 提供了一个更统一的解释。

它把 TokenMixing 看成一个固定 permutation matrix。

也就是说，原本看起来只是 reshape + transpose 的操作，本质上是一个固定的 mixing pattern。

固定 pattern 的好处是便宜、稳定、GPU-friendly。

坏处是它不根据数据自适应。

推荐特征非常异构。user token、item token、dense token、sequence query token 之间的关系不一定适合固定交换。

所以 UniMixer 想做的是：

> 保留 TokenMixer 的高效先验，但让 mixing pattern 变得可学习。

对 TAAC2026 来说，短期不一定要完整实现 UniMixer。更现实的方向是：

- 给 fixed mixing 加 residual gate。
- 在 token 维加一个小的 learnable mixing matrix。
- 用 identity 初始化，让模型自己决定要不要偏离原始 TokenMixing。

这比一上来换完整 backbone 风险小。

## 18. 时间特征：不能只当普通 embedding

TAAC2026 的 sequence 是按时间排列的，且倒序。

baseline 已经有 time bucket，但时间信息还有很多层。

第一层是 event-level time：

> 每个历史行为距离当前样本多久？

第二层是 sequence rhythm：

> 最近是否有密集行为？是否有 session burst？

第三层是 request-level time：

> 当前样本发生在一天/一周/训练窗口的什么位置？

如果把这些时间信息全部当普通 token 或普通 embedding，可能会有两个问题：

- 时间信号被主干 token mixing 稀释。
- 时间先验太强时污染 user / item 表示。

所以我更倾向于把时间做成一个 control signal：

```text
time raw features
-> TimeContextEncoder
-> z_t

z_t affects:
  query generation
  domain weighting
  final logit residual
```

这比简单加一个 time token 更符合工业推荐里的时间建模。

## 19. Embedding reinit：推荐模型的过拟合经常先发生在 embedding

CTR / CVR 模型里，多 epoch 训练很容易遇到 embedding overfitting。

原因是 sparse embedding table 高基数、长尾、更新稀疏。某些 ID row 被更新次数很少，重复训练时容易记住训练集局部模式。

相关论文里有两条思路：

- MEDA：每个 epoch 重置 embedding，保留 MLP / backbone。
- Pinterest Ads：更谨慎，强调 frequency-adaptive learning rate，不要对所有 embedding row 一视同仁。

放到 TAAC2026，我觉得不能简单“所有 embedding 每轮都重置”。

更合理的是区分：

- 高基数、长尾、行为序列 ID：更容易过拟合，可考虑 reinit / 降 LR / 强正则。
- 稳定 user / item side info：不一定该重置。
- time bucket / gap bucket：更像结构性特征，不应该随便重置。

这类细节看起来不像模型结构，但很可能影响泛化。

推荐比赛里经常有这种现象：loss 更好看，不代表线上更好。embedding 记忆训练集后，loss 会下降，但 test 不一定涨。

## 20. 我会怎么继续看 TAAC2026

如果只按 baseline 改，我觉得优先级可以这样排。

第一，先把数据和验证集搞清楚。

本地 valid 如果和线上不一致，后面所有实验判断都会变形。

第二，改 sequence summary 和 query generation。

baseline 里 mean pooling 太粗。可以尝试 recent pooling、recency-weighted pooling、domain summary、time-aware summary。

第三，改时间特征注入方式。

不要只把时间作为普通 embedding 加到 seq token 上，可以考虑独立 TimeContextEncoder，让时间影响 query、domain gate 和输出 residual。

第四，改 TokenMixing 的 residual 和 FFN。

可以先做：

- residual gate
- mix-and-revert
- pre-norm
- per-token / grouped FFN

这些都比直接换大模型更稳。

第五，再考虑更大的 backbone。

比如 MixFormer-style per-query fusion，HyFormer block 加深，或者 learnable mixing。

但 scale up 要建立在前面几步稳定之后。否则只是把 baseline 的结构问题放大。

## 21. 这篇的总结

TAAC2026 baseline 看起来代码很复杂，但抽象出来就是几件事：

- 把 user / item 异构特征 token 化。
- 把四个 domain 的行为序列编码成 sequence tokens。
- 生成 query tokens 去读取序列。
- 用 TokenMixing 做 q tokens 和 NS tokens 的交互。
- 输出 PCVR 相关预测。

我觉得它最值得学习的地方不是某个具体模块，而是它把工业推荐里的几个问题放在了一起：

- 异构特征怎么统一表示？
- 长序列怎么不爆显存地使用？
- 当前 item / user context 怎么影响历史行为读取？
- 时间信息怎么进入模型？
- cheap token mixing 能不能支撑 scale up？

TAAC2025 像是在看生成式推荐的 item 表示和目标设计。

TAAC2026 则更像是在看工业排序模型的结构设计。

这两年题目放在一起看，会发现推荐系统正在从传统的“特征工程 + 小模型排序”走向两个方向：

- 一边是生成式推荐，把 item 变成可生成的 semantic ID。
- 一边是工业排序大模型，把异构特征和长序列组织成可 scale 的 token backbone。

这也是我觉得推荐系统值得继续跟的原因：它不是一个已经固定下来的工程套路，而是正在吸收 Transformer、生成模型、scaling law、长序列建模这些更前沿的东西。

## 参考

- [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)
- [TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2602.06563)
- [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
- [OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender](https://arxiv.org/abs/2510.26104)
- [MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)
- [UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems](https://arxiv.org/abs/2604.00590)
- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)
