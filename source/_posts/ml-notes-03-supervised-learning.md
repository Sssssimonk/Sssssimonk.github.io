---
title: "机器学习笔记 03：有监督学习"
date: 2024-06-22 21:00:00
updated: 2024-06-22 21:00:00
categories:
  - 机器学习笔记
tags:
  - 机器学习
  - 线性回归
  - 逻辑回归
  - KNN
  - 笔记
math: true
category_bar: true
---


## 1. 线性回归（linear regression）：用线性函数做预测

Linear regression 用来预测连续值。最简单的一维形式是：

$$
\hat{y} = wx + b
$$

多维特征时写成：

$$
\hat{y} = w_1x_1 + w_2x_2 + ... + w_dx_d + b
$$

比如房价预测中，$x_1$ 是面积，$x_2$ 是房龄，$x_3$ 是距离地铁的距离。模型会学习每个特征对应的权重 $w_i$。如果 $w_1$ 很大，说明面积对预测房价影响更大；如果 $w_2$ 是负数，说明房龄增加可能让房价下降。

Linear regression 常用 MSE 作为 loss：

$$
J(w,b)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

训练目标就是找到一组 $w,b$，让预测值和真实值之间的平方误差尽量小。

一个小例子：

| 面积 | 真实房价 | 模型预测 |
|---:|---:|---:|
| 60 | 300 | 320 |
| 80 | 450 | 430 |
| 100 | 600 | 610 |

如果模型预测整体偏高，更新参数时就会降低某些权重或 bias；如果预测整体偏低，就会反过来调大。具体怎么改，由 loss 对参数的 gradient 决定。

## 2. 线性模型的假设

Linear model 最大的假设是：特征和目标之间可以用线性关系近似。

这不代表真实世界一定是线性的，而是说在当前特征空间里，线性函数已经足够表达主要趋势。比如房价和面积大致正相关，这种关系线性模型就能抓住。但如果目标和特征之间存在复杂的非线性关系，普通线性模型就会 underfit。

一个典型例子是广告点击率。用户年龄对点击率的影响可能不是单调线性的：太年轻和太年长的用户都可能低，中间某个年龄段高。直接用年龄做线性特征可能不够，需要特征变换、交叉特征，或者换更强的模型。

线性模型的优点是简单、稳定、可解释。缺点是表达能力有限，对 feature engineering 比较依赖。

## 3. 逻辑回归（logistic regression）：名字叫 regression，但做分类（classification）

Logistic regression 通常用于二分类。它先计算一个线性打分：

$$
z = w^Tx + b
$$

然后用 sigmoid 把这个打分压到 $[0,1]$：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

输出可以理解成正类概率：

$$
P(y=1|x)=\sigma(w^Tx+b)
$$

比如垃圾邮件分类中，如果输出 $0.92$，可以理解为模型认为这封邮件是垃圾邮件的概率很高。如果输出 $0.08$，则更倾向于正常邮件。

最后再通过 threshold 得到类别：

```text
if P(y = 1 | x) >= 0.5:
    predict 1
else:
    predict 0
```

threshold 不一定非要是 0.5。比如疾病筛查里更怕漏诊，可以把 threshold 调低，让模型更容易预测为阳性，从而提高 recall。

## 4. 为什么逻辑回归（logistic regression）适合分类

Linear regression 直接输出连续值，不适合表示概率。Logistic regression 的 sigmoid 输出天然在 $[0,1]$ 范围内，所以可以解释为概率。

二分类常用 Binary Cross Entropy：

$$
L(y,\hat{y})=-[y\log(\hat{y})+(1-y)\log(1-\hat{y})]
$$

如果真实标签是 1，但模型只给正确类别 0.1 的概率，loss 会很大；如果给 0.9，loss 就很小。

这也是 logistic regression 和 linear regression 的关键区别：

| 模型 | 输出 | 常见任务 | 常见 loss |
|---|---|---|---|
| Linear Regression | 连续值 | Regression | MSE |
| Logistic Regression | 概率 | Binary Classification | Binary Cross Entropy |

虽然 logistic regression 名字里有 regression，但它本质上是在做分类概率建模。


## 6. KNN：没有显式训练过程的模型

KNN 全称是 K-Nearest Neighbors。它的思想很直接：要预测一个新样本，就看训练集中离它最近的 $K$ 个样本，然后让它们投票。

分类任务里，流程是：

```text
for a new sample:
    compute distance to all training samples
    choose K nearest samples
    count their labels
    return the majority label
```

比如 $K=5$，最近的 5 个邻居里有 3 个是类别 A，2 个是类别 B，那么新样本预测为 A。

KNN 是 non-parametric model。它没有像 linear regression 那样显式学出一组 $w,b$。训练阶段基本就是存下训练数据，预测阶段才真正计算距离。

这也带来一个问题：训练很快，但预测可能很慢。数据量很大时，每预测一个样本都要和大量训练样本算距离。

## 7. 距离和相似度很关键

KNN 的效果很依赖距离度量。常见距离包括 Euclidean distance、Manhattan distance、cosine similarity 等。

Euclidean distance：

$$
d(x,z)=\sqrt{\sum_i (x_i-z_i)^2}
$$

Cosine similarity：

$$
\cos(\theta)=\frac{x \cdot z}{\|x\|\|z\|}
$$

如果特征 scale 差异很大，KNN 会被大尺度特征主导。比如一个特征是年龄 $[0,100]$，另一个特征是收入 $[0,100000]$，不做 normalization 时，收入几乎会决定距离。

所以 KNN 前通常需要做 feature scaling。

## 8. 点积（dot product）和余弦相似度（cosine similarity）

这部分和推荐系统、embedding retrieval、RAG 都有关。

Dot product 是：

$$
x \cdot z = \sum_i x_i z_i
$$

Cosine similarity 是归一化后的方向相似度：

$$
\frac{x \cdot z}{\|x\|\|z\|}
$$

差别在于：dot product 同时受方向和向量长度影响；cosine similarity 主要看方向。

如果 embedding 已经归一化，也就是 $\|x\|=\|z\|=1$，那么：

$$
x \cdot z = \cos(\theta)
$$

推荐系统里有时会保留 embedding norm，因为热门物品可能被训练出更大的向量长度，dot product 会把这种 popularity signal 也算进去。检索场景里如果只关心语义方向，则常用 cosine similarity 或 normalized embedding dot product。

## 9. 几个点

Logistic regression 做分类，但它的 decision boundary 仍然是线性的。sigmoid 只是把线性打分转成概率，并没有让边界变复杂。

KNN 对特征 scale 很敏感。写 KNN 前先想 normalization，不然距离可能没有意义。

KNN 的 $K$ 太小容易受噪声影响，$K$ 太大又可能把局部结构抹平。

Linear model 不是“不强”，而是依赖特征表达。如果 feature engineering 做得好，线性模型在很多工业场景里仍然很有用。

## 参考资料

- [Stanford CS229: Machine Learning](https://cs229.stanford.edu/)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
