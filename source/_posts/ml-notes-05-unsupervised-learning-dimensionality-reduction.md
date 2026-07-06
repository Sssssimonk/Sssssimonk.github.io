---
title: "机器学习笔记 05：无监督学习与降维"
date: 2024-07-06 22:00:00
updated: 2024-07-06 22:00:00
categories:
  - 机器学习笔记
tags:
  - 机器学习
  - 无监督学习
  - 聚类
  - PCA
  - 笔记
math: true
category_bar: true
---

这一篇整理无监督学习和降维。监督学习有 label，模型知道自己要预测什么；无监督学习没有明确答案，目标更多是发现数据结构。最常见的两类任务是 clustering 和 dimensionality reduction。

## 1. 什么是无监督学习（unsupervised learning）

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

## 2. 聚类（clustering）：把相似样本放在一起

Clustering 的目标是把样本分成若干组，让同一组内部尽量相似，不同组之间尽量不同。

但这里有一个麻烦：没有 label，所以“分得好不好”不像分类任务那样容易判断。

不同 clustering 方法对 cluster 形状有不同假设。KMeans 偏向球形 cluster，DBSCAN 偏向密度连通区域，hierarchical clustering 更强调层级关系。

### KMeans：最经典的聚类方法

KMeans 是最经典的 clustering 方法之一。它需要先指定 cluster 数量 $K$。

基本流程：

```text
1. 选择k个centroids
2. 计算每个点与centroids的距离
3. 将每个点分配给距离最近的centroid
4. 更新centroid位置
5. 重复3、4直到手链
```

目标可以理解为最小化簇内平方距离：

$$
\sum_{k=1}^{K}\sum_{x_i \in C_k}\|x_i-\mu_k\|^2
$$

其中，$C_k$ 是第 $k$ 个 cluster，$\mu_k$ 是这个 cluster 的中心。

优点：实现简单

KMeans 的问题：

- 需要提前指定 $K$
- 对初始中心敏感
- 对 outlier 敏感
- 更适合接近球形的 cluster
- 特征 scale 会影响距离，通常要先 normalization


**关于怎么选 K**

常见方法是 elbow method 和 silhouette score。

**Elbow method** 看不同 $K$ 下的簇内误差。随着 $K$ 增大，误差一定会下降，因为 cluster 越多，样本离中心越近。但下降幅度会逐渐变小。拐点附近的 $K$ 通常是一个候选。

**Silhouette score** 同时考虑样本离自己 cluster 的距离，以及离其他 cluster 的距离。分数越高，说明样本更像自己 cluster，而不像其他 cluster。

对于kmeans的改进方法有：kmeans++，sequential kmeans等，这里我就懒得写了，感兴趣可以搜一下。

### DBSCAN：基于密度的聚类

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

以下是pseudo code
```text
DBSCAN(D, eps, minPts):
    所有点标记为未访问，cluster_id = 0

    for p in D:
        if p 已访问: continue
        标记 p 已访问
        neighbors = 距离 p ≤ eps 的所有点

        if |neighbors| < minPts:
            标记 p 为噪声
        else:
            cluster_id += 1
            把 p 归入 cluster_id
            把 neighbors 加入队列
            while 队列非空:
                q = 队列出队
                if q 未访问:
                    标记 q 已访问
                    qNeighbors = 距离 q ≤ eps 的所有点
                    if |qNeighbors| >= minPts:
                        队列加入 qNeighbors
                if q 尚未归入任何簇:
                    把 q 归入 cluster_id
```
### 层次聚类（hierarchical clustering）

Hierarchical clustering 关注层级结构。最常见的是 agglomerative clustering，也就是自底向上合并。

流程：


1. 找最近的两个点/cluster合并成新cluster。
2. 不断合并直到收敛


它的结果可以用 dendrogram 表示。虽然这里不画图，但可以把它想成一棵树：底部是单个样本，越往上 cluster 越大。

关键在于怎么定义 cluster 之间的距离：

- single linkage：两个 cluster 中最近点的距离
- complete linkage：两个 cluster 中最远点的距离
- average linkage：所有点对距离的平均
- Ward linkage：合并后簇内方差增加最小

Hierarchical clustering 的优点是能看到层级关系，不一定要一开始就指定 cluster 数。缺点是计算成本高，大数据上不一定适合。

聚类（clustering）方法对比

| 方法 | 核心假设 | 优点 | 缺点 |
|---|---|---|---|
| KMeans | cluster 接近球形 | 简单、快 | 要指定 K，对 outlier 敏感 |
| DBSCAN | cluster 是密度区域 | 能识别噪声，支持非球形 | 参数敏感，高维效果可能差 |
| Hierarchical | 数据有层级结构 | 结果可解释 | 大数据成本高 |

KMeans 更像是在找中心点，DBSCAN 更像是在找密度连通区域，hierarchical clustering 更像是在整理样本之间的层级关系。

## 3. 降维（dimensionality reduction）：为什么要降维

降维的目标是把高维数据变成低维表示，同时尽量保留重要信息。

常见动机：

- 可视化：把高维 embedding 投到 2D/3D
- 去噪：去掉不重要的方向
- 压缩：减少特征数量
- 缓解维度灾难：高维空间中距离可能变得不稳定

比如一个用户有 1000 个行为特征，很多特征之间高度相关。降维可以把它们压成几十个综合特征，保留主要变化方向。

### PCA：用主成分保留主要变化

PCA 的核心思想是：找到数据方差最大的方向，然后把数据投影到这些方向上。

基本流程：

```text
1. 将数据标准化（或中心化），使得每个特征均值为0。

2. 计算协方差矩阵，反映特征间的相关性。

3. 对协方差矩阵进行特征值分解，得到特征值和特征向量。

4. 将特征值从大到小排序，选择前k个最大特征值对应的特征向量作为主成分。

5. 用这些特征向量组成投影矩阵，将原始数据乘以该矩阵，得到降维后的数据。
```

**简单的说就是找eigenvector，这个就是主成分。**

如果一个二维数据集大致沿着一条斜线分布，PCA 的第一主成分就是这条斜线方向。把数据投影到这条线，就能用一维保留大部分变化。

数学上，PCA 会找一组正交方向，第一主成分解释最多 variance，第二主成分在和第一主成分正交的条件下解释剩余最多 variance。


PCA 的几个注意点

- PCA 对 scale 敏感。一个特征范围是 $[0,100000]$，另一个是 $[0,1]$，如果不 standardize，大范围特征会主导 principal components。
- PCA 是线性降维。如果数据本身是复杂非线性结构，PCA 可能表达不好。
- PCA 的主成分不一定好解释。原始特征可能有明确含义，但主成分是多个特征的线性组合，解释起来会更抽象。


其他：t-SNE 和 UMAP

t-SNE 和 UMAP 常用于可视化高维 embedding，比如把文本 embedding 或图像 embedding 投到二维。

它们更关注局部邻域结构，适合看样本是否聚在一起。但这类方法主要用于 visualization，不太适合直接当成严肃的全局距离解释。

一个常见误区是看见二维图上两个 cluster 离得远，就断言原始高维空间里它们也有同样的全局距离关系。这个不一定成立。

## 参考资料

- hkust bdt5002课程
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
