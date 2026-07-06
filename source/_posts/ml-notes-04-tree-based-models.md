---
title: "机器学习笔记 04：树模型"
date: 2024-06-29 21:30:00
updated: 2024-06-29 21:30:00
categories:
  - 机器学习笔记
tags:
  - 机器学习
  - 决策树
  - 随机森林
  - XGBoost
  - 笔记
math: true
category_bar: true
---

这一篇整理 tree-based models。树模型的核心思想很直观：不断用特征把数据切开，让每个子节点里的样本越来越“纯”。单棵树容易 overfit，所以实际中经常用 ensemble tree models，比如 Random Forest、GBDT、XGBoost。

## 1. 决策树（decision tree）的基本想法

Decision Tree 每一步都在问一个问题：


> 这个 feature 按某个 threshold 切开后，数据会不会更容易预测？


比如判断一个用户是否会购买会员，可以有这样的分裂：

```text
访问次数 > 10 ?
    yes -> 停留时间 > 5 分钟 ?
    no  -> 预测不购买
```

每个内部节点是一个 split rule，每个叶子节点给出最终预测。

分类树希望叶子节点里的类别尽量单一；回归树希望叶子节点里的目标值尽量接近。

## 2. 如何切分数据信息，什么叫节点更“纯”

假设一个节点里有 10 个样本：

| 类别 | 数量 |
|---|---:|
| 正类 | 5 |
| 负类 | 5 |

这个节点很不纯，因为正负各一半。

如果一次 split 后变成：

| 子节点 | 正类 | 负类 |
|---|---:|---:|
| left | 5 | 1 |
| right | 0 | 4 |

那这个 split 就不错，因为两个子节点都更容易判断。

Decision Tree 的 splitting metric，本质上就是量化“split 前后到底变纯了多少”。

在这里使用**熵（entropy）** 衡量不确定性：
熵越低，节点越“纯”

$$
H(S)=-\sum_{c}p_c\log_2(p_c)
$$

其中，$p_c$ 是类别 $c$ 在节点中的比例。

二分类时，如果正负样本各一半，entropy 最大；如果节点里全是同一类，entropy 为 0。

例：

| 节点情况 | Entropy |
|---|---|
| 5 正 / 5 负 | 1.0 |
| 9 正 / 1 负 | 约0.47 |
| 10 正 / 0 负 | 0 |

Entropy 越低，节点越纯。

## 3. Splitting metrics

**信息增益（information gain）**

Information Gain 衡量 split 后 entropy 降低了多少：

$$
IG(S,A)=H(S)-\sum_{v \in Values(A)}\frac{|S_v|}{|S|}H(S_v)
$$

前半部分是 split 前的不确定性，后半部分是 split 后各子节点不确定性的加权平均。

如果 split 后子节点变得很纯，后半部分就小，information gain 就大。

ID3 使用 information gain 来作为split指标。


**增益率（gain ratio）**

Information Gain 有一个问题：它偏好取值很多的特征。

比如用户 ID 这种特征，每个用户几乎都是唯一值。按用户 ID 切，训练集上可以变得很纯，但完全没有泛化意义。

Gain Ratio 会对这种情况做惩罚：

$$
GainRatio(S,A)=\frac{InformationGain(S,A)}{SplitInfo(S,A)}
$$

C4.5 使用 gain ratio。直觉上，它不只看 split 后纯不纯，也看这个 split 是不是把数据切得过于碎。

**基尼指数（Gini index）**

CART 分类树常用 Gini Index：

$$
Gini(S)=1-\sum_c p_c^2
$$

Gini 也衡量节点不纯度。节点越纯，Gini 越小。

二分类时：

| 正类比例 | Gini |
|---:|---:|
| 0.5 | 0.5 |
| 0.9 | 0.18 |
| 1.0 | 0 |

Gini 和 entropy 的目标很接近：都希望 split 后子节点更纯。实际使用中，Gini 计算更简单一些。

## 4. ID3、C4.5、CART

三类树的区别：

| 算法 | Split metric | 任务 |
|---|---|---|
| ID3 | Information Gain | Classification |
| C4.5 | Gain Ratio | Classification |
| CART | Gini / Squared Error | Classification / Regression |

CART 是二叉树，每次 split 生成两个子节点。分类时常用 Gini，回归时常用 squared error 或 variance reduction。

回归树的直觉也很简单：如果一次 split 能让两个子节点内部的目标值更接近，这个 split 就有价值。

比如预测房价时，按“是否靠近地铁”切分后，如果靠近地铁的一组房价普遍更高，不靠近的一组更低，那么这个 split 就能降低每个子节点内部的误差。

**剪枝（pruning）：**

单棵树很容易 overfit。如果不限制深度，它可以不断 split，直到叶子节点里只剩很少样本，甚至每个叶子只对应一个训练样本。

这在训练集上很好，但对新数据很差。

常见限制方式：

- `max_depth`：限制树深度
- `min_samples_split`：节点样本太少就不再切
- `min_samples_leaf`：叶子节点至少保留一定样本
- `max_leaf_nodes`：限制叶子节点数

Pruning 分两类：

**Pre-pruning**：树还没长完就提前停止，比如限制最大深度。

**Post-pruning**：先长出一棵比较完整的树，再从底部往上剪掉泛化收益不大的分支。

简单说，剪枝就是不让树把训练集记得太细。

## 5.多棵树的组合

**Bagging：随机森林（random forest）**

Random Forest 属于 Bagging 方法。它训练很多棵 decision tree，然后让它们投票或取平均。

Random Forest 的随机性主要来自两点：

1. 每棵树使用 bootstrap sample，也就是有放回地抽训练样本。
2. 每次 split 时，只从部分特征中选择最佳 split。

这样每棵树都不太一样，错误也不会完全一致。

单棵树 variance 很高，Random Forest 通过多棵树平均来降低 variance。

一个直觉例子：一个人判断可能很偏，但如果很多个相对独立的人投票，最终结果通常更稳定。Random Forest 也是类似的思路。

**Boosting：GBDT，XGBoost，lightGBM**

Boosting 和 Random Forest 不一样。Random Forest 里多棵树大致是并行、独立训练的；Boosting 是一棵接一棵训练，后面的树重点修正前面模型的错误。

**GBDT** 的思路是：

```text
initial prediction
-> compute residual / negative gradient
-> train a new tree to fit the residual
-> add this tree to the ensemble
-> repeat
```

如果第一棵树预测房价总是偏低，下一棵树就会学习这个偏差，把预测往正确方向拉。

**XGBoost**可以看作更工程化、更正则化、更高效的 gradient boosting tree 实现。它的特点包括：

- **二阶泰勒展开**：使用一阶和二阶梯度（GBDT只用一阶），使损失函数逼近更精准。
- **正则化**：目标函数加入L1/L2正则项（控制叶子节点权重和数量），防止过拟合。
- **并行化**：特征预排序（Block结构）支持多线程查找最佳分裂点，但树仍是串行生成的。
- **缺失值处理**：自动学习缺失值的默认分裂方向。

这也是为什么 XGBoost 在传统 tabular data 上经常很强。

**LightGBM**

| **特性** | **XGBoost** | **LightGBM** |
| --- | --- | --- |
| **生长策略** | Level-wise（按层分裂） | Leaf-wise（按叶子分裂） |
| **速度** | 较慢（预排序特征） | 更快（直方图算法） |
| **内存占用** | 较高（存储预排序数据） | 较低（直方图压缩） |
| **正则化** | 支持L1/L2 | 支持L1/L2 |
| **类别特征处理** | 需独热编码 | 直接支持（无需编码） |
| **并行优化** | 特征并行 | 特征并行+数据并行 |

**LightGBM**：大数据场景的默认选择，速度快、内存友好。

**XGBoost**：**小数据**或高精度需求时更优，调参空间大。

## 备注

树模型不需要像 KNN 那样强依赖 feature scaling。因为树只关心某个特征是否大于 threshold，不关心距离。

树模型擅长处理非线性关系和特征交互。比如“访问次数高并且最近 7 天活跃”这种规则，树可以比较自然地表达，而且树模型很适合可视化的。

单棵树容易 overfit。实际里更常用 Random Forest、GBDT、XGBoost 这类 ensemble。

一般来说，Random Forest 更偏降低 variance，Boosting 更偏逐步降低 bias，

## 参考资料

- ucsd dsc40B课程
- [An Introduction to Statistical Learning](https://www.statlearning.com/)

