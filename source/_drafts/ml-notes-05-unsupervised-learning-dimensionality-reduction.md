---
title: "ML Notes 05: Unsupervised Learning 与 Dimensionality Reduction"
date: 2026-07-03 22:00:00
updated: 2026-07-03 22:00:00
categories:
  - Machine Learning Notes
tags:
  - Machine Learning
  - Unsupervised Learning
  - Clustering
  - PCA
  - Notes
math: true
category_bar: true
---

这一篇整理无监督学习和降维。监督学习有 label，模型知道自己要预测什么；无监督学习没有明确答案，目标更多是发现数据结构。最常见的两类任务是 clustering 和 dimensionality reduction。

## 1. 什么是 Unsupervised Learning

无监督学习只有输入 $x$，没有标签 $y$。

比如有一批用户行为数据：

| 用户 | 访问次数 | 平均停留时间 | 购买次数 |
|---|---:|---:|---:|
| A | 50 | 8 分钟 | 5 |
| B | 3 | 1 分钟 | 0 |
| C | 45 | 7 分钟 | 4 |
| D | 5 | 2 分钟 | 0 |

没有人告诉模型谁是“高价值用户”，但模型可以发现 A 和 C 比较像，B 和 D 比较像。这就是 clustering 的典型使用场景。

无监督学习不是为了直接预测 label，而是为了整理数据：哪些样本相似，数据有没有低维结构，有没有异常点。

## 2. Clustering：把相似样本放在一起

Clustering 的目标是把样本分成若干组，让同一组内部尽量相似，不同组之间尽量不同。

但这里有一个麻烦：没有 label，所以“分得好不好”不像分类任务那样容易判断。

不同 clustering 方法对 cluster 形状有不同假设。KMeans 偏向球形 cluster，DBSCAN 偏向密度连通区域，hierarchical clustering 更强调层级关系。

## 3. KMeans

KMeans 是最经典的 clustering 方法之一。它需要先指定 cluster 数量 $K$。

基本流程：

```text
choose K initial centers
assign each point to the nearest center
update each center by averaging assigned points
repeat until centers stop changing
```

目标可以理解为最小化簇内平方距离：

$$
\sum_{k=1}^{K}\sum_{x_i \in C_k}\|x_i-\mu_k\|^2
$$

其中，$C_k$ 是第 $k$ 个 cluster，$\mu_k$ 是这个 cluster 的中心。

一个简单例子：如果用户按“访问次数”和“购买次数”分布明显，一群是高访问高购买，另一群是低访问低购买，KMeans 很容易把它们分开。

KMeans 的问题也很明显：

- 需要提前指定 $K$
- 对初始中心敏感
- 对 outlier 敏感
- 更适合接近球形的 cluster
- 特征 scale 会影响距离，通常要先 normalization

## 4. 怎么选 K

常见方法是 elbow method 和 silhouette score。

**Elbow method** 看不同 $K$ 下的簇内误差。随着 $K$ 增大，误差一定会下降，因为 cluster 越多，样本离中心越近。但下降幅度会逐渐变小。拐点附近的 $K$ 通常是一个候选。

**Silhouette score** 同时考虑样本离自己 cluster 的距离，以及离其他 cluster 的距离。分数越高，说明样本更像自己 cluster，而不像其他 cluster。

实际使用时，不要只机械看指标。比如用户分群最后要服务业务策略，那 cluster 数量还要考虑是否可解释、是否方便运营。

## 5. DBSCAN：基于密度的 clustering

DBSCAN 不需要提前指定 cluster 数量。它通过密度来定义 cluster。

两个重要参数：

- `eps`：邻域半径
- `min_samples`：成为核心点需要的邻居数量

DBSCAN 会把点分成三类：

| 类型 | 含义 |
|---|---|
| Core point | eps 范围内有足够多邻居 |
| Border point | 不满足 core 条件，但在某个 core point 邻域内 |
| Noise point | 不属于任何密度区域 |

DBSCAN 的优点是能发现非球形 cluster，也能识别 outlier。

比如二维空间里有两个弯月形 cluster，KMeans 可能分不好，因为它偏向球形；DBSCAN 更可能按密度连通性分出来。

缺点是参数敏感。如果 `eps` 太小，很多点会被当成 noise；如果太大，不同 cluster 可能被连在一起。

## 6. Hierarchical Clustering

Hierarchical clustering 关注层级结构。最常见的是 agglomerative clustering，也就是自底向上合并。

流程：

```text
start with each point as its own cluster
merge the two closest clusters
repeat until only one cluster remains or target number reached
```

它的结果可以用 dendrogram 表示。虽然这里不画图，但可以把它想成一棵树：底部是单个样本，越往上 cluster 越大。

关键在于怎么定义 cluster 之间的距离：

- single linkage：两个 cluster 中最近点的距离
- complete linkage：两个 cluster 中最远点的距离
- average linkage：所有点对距离的平均
- Ward linkage：合并后簇内方差增加最小

Hierarchical clustering 的优点是能看到层级关系，不一定要一开始就指定 cluster 数。缺点是计算成本高，大数据上不一定适合。

## 7. Clustering 方法对比

| 方法 | 核心假设 | 优点 | 缺点 |
|---|---|---|---|
| KMeans | cluster 接近球形 | 简单、快 | 要指定 K，对 outlier 敏感 |
| DBSCAN | cluster 是密度区域 | 能识别噪声，支持非球形 | 参数敏感，高维效果可能差 |
| Hierarchical | 数据有层级结构 | 结果可解释 | 大数据成本高 |

KMeans 更像是在找中心点，DBSCAN 更像是在找密度连通区域，hierarchical clustering 更像是在整理样本之间的层级关系。

## 8. Dimensionality Reduction：为什么要降维

降维的目标是把高维数据变成低维表示，同时尽量保留重要信息。

常见动机：

- 可视化：把高维 embedding 投到 2D/3D
- 去噪：去掉不重要的方向
- 压缩：减少特征数量
- 缓解维度灾难：高维空间中距离可能变得不稳定

比如一个用户有 1000 个行为特征，很多特征之间高度相关。降维可以把它们压成几十个综合特征，保留主要变化方向。

## 9. PCA

PCA 的核心思想是：找到数据方差最大的方向，然后把数据投影到这些方向上。

基本流程：

```text
standardize features
compute covariance matrix
compute eigenvectors and eigenvalues
choose top principal components
project data to lower-dimensional space
```

如果一个二维数据集大致沿着一条斜线分布，PCA 的第一主成分就是这条斜线方向。把数据投影到这条线，就能用一维保留大部分变化。

数学上，PCA 会找一组正交方向，第一主成分解释最多 variance，第二主成分在和第一主成分正交的条件下解释剩余最多 variance。

## 10. PCA 的几个注意点

PCA 对 scale 敏感。一个特征范围是 $[0,100000]$，另一个是 $[0,1]$，如果不 standardize，大范围特征会主导 principal components。

PCA 是线性降维。如果数据本身是复杂非线性结构，PCA 可能表达不好。

PCA 的主成分不一定好解释。原始特征可能有明确含义，但主成分是多个特征的线性组合，解释起来会更抽象。

## 11. t-SNE 和 UMAP

t-SNE 和 UMAP 常用于可视化高维 embedding，比如把文本 embedding 或图像 embedding 投到二维。

它们更关注局部邻域结构，适合看样本是否聚在一起。但这类方法主要用于 visualization，不太适合直接当成严肃的全局距离解释。

一个常见误区是看见二维图上两个 cluster 离得远，就断言原始高维空间里它们也有同样的全局距离关系。这个不一定成立。

## 12. 几个点

Clustering 没有 label，所以评估本来就更难。指标只能辅助，最后还要看任务目的。

KMeans 前基本要做 scaling，否则距离会被大尺度特征主导。

DBSCAN 适合找异常点和非球形 cluster，但参数 `eps` 很关键。

PCA 保留的是最大 variance 方向，不等于保留对下游任务最有用的方向。

降维后的图很好看，不代表结论一定可靠。可视化是探索工具，不是最终证明。

## References

- [scikit-learn User Guide: Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn User Guide: Decomposition](https://scikit-learn.org/stable/modules/decomposition.html)
- [scikit-learn User Guide: Manifold Learning](https://scikit-learn.org/stable/modules/manifold.html)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
