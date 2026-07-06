---
title: "机器学习笔记 01：机器学习的基本框架"
date: 2024-06-08 20:00:00
updated: 2024-06-08 20:00:00
categories:
  - 机器学习笔记
tags:
  - 机器学习
  - 笔记
math: true
category_bar: true
---

这篇笔记整理机器学习最基础的一层框架：机器学习到底在学什么，一个模型从数据到预测结果之间经历了什么，以及为什么训练集表现好并不等于模型真的好。

可以先把 machine learning 理解成：

> 从已有数据中学习一个可以泛化到未来数据的 prediction function。

两个关键词：**学习** 和 **泛化**。学习指的是模型通过数据调整自己的参数；泛化指的是模型不只是记住训练样本，而是能在没见过的数据上仍然给出合理预测。

## 1. 一个机器学习问题由什么组成

最常见的机器学习问题可以写成：

$$
\hat{y} = f_\theta(x)
$$

其中，$x$ 是输入特征，$\hat{y}$ 是模型预测，$f_\theta$ 是模型，$\theta$ 是模型参数。

![机器学习架构](/img/ml1_overview.png)

个人认为machine learning就是从数据中学到pattern/model 这个model可以对未来的数据进行推测，得到一个结果。（简单的说可以当成一个prediction model）

所以机器学习不是直接“学一个答案”，而是学习一个从输入到输出的映射关系。

Machine Learning的流程通常分为以下几步

**ML的基本流程：**

1. 数据预处理
2. 模型训练
3. 模型评估

![机器学习流水线](/img/ml1_pipeline.png)

可以看到根据是否有样本label可以分为supervised learning和unsupervised learning，并且可以继续细分为regression，classification（在某种情况下可以互相切换，比如regression拟合出的线就可以用来分类），clustering，再加上强化学习就组成我们广义下定义的机器学习。

机器学习训练可以粗略理解成最小化 empirical risk：

$$
\theta^* = \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n} L(f_\theta(x_i), y_i)
$$

在所有可能的参数 $\theta$ 中，找到一组参数，让模型在训练数据上的平均 loss 尽可能小。



## 2. 机器学习方向的分类

机器学习通常先按有没有 label 来分。

**Supervised Learning** 有明确的输入和目标输出。模型看到的是 $(x, y)$，目标是学会从 $x$ 预测 $y$。常见任务包括 **regression** 和 **classification**。

Regression 预测连续值，比如房价、温度、点击率。Classification 预测类别，比如垃圾邮件识别、疾病诊断、图片分类。

**Unsupervised Learning** 没有明确的 label。模型只看到 $x$，目标是发现数据内部结构。常见任务包括聚类 clustering、dimensionality reduction 降维。


简单说，supervised learning 更像是在学一个带答案的映射函数，尝试让数据分布向真实的数据分布靠近。而unsupervised learning 更像是在没有标准答案的情况下整理数据的内在结构，找到数据的规律，将数据变成有信息的内容为我们所用。

**Reinforcement learning**
相比较其他的领域，强化学习定义模型agent是在一个environment中进行交互，根据当前的状态state决定action，然后根据action获得environment反馈的reward。

## 3. 数据集的划分

机器学习里经常会把数据分成：

| 数据集 | 作用 |
|---|---|
| Training set | 用来训练模型参数 |
| Validation set | 用来调超参数、选模型 |
| Test set | 用来估计最终泛化能力 |

不能只看 training set 表现，因为模型可以通过记住训练样本获得很低的 training loss，但这不代表它真的学到了规律。


![过拟合与欠拟合](/img/ml1_overfitting.png)

核心问题：模型在训练集上表现很好，但在新数据上表现很差。
所以我们希望找到模型在过拟合欠拟合之间的完美balance状态。

比如一个垃圾邮件分类器，如果它只是记住了训练集中出现过的垃圾邮件地址，那么训练集 accuracy 可能很高。但新垃圾邮件换了发件人和措辞后，它就可能失效。真正有用的模型应该学习“垃圾邮件通常有哪些模式”（我们称之为泛化性），而不是背下训练样本。

**Bias-Variance Tradeoff**

| 问题 | 现象 | 直觉 |
|---|---|---|
| High bias | train/test 都差 | 模型太简单 |
| High variance | train 好，test 差 | 模型太依赖训练集 |


## 4. 损失函数（loss function）

训练模型前，第一步是定义什么叫“错”。这个定义就是 loss function。

对于回归任务，常用 Mean Squared Error：

$$
L(\theta) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

其中，$y_i$ 是真实值，$\hat{y}_i$ 是预测值。预测和真实值差得越远，loss 越大。

比如真实房价是 500，模型预测 450，误差是 50；如果预测 300，误差是 200。MSE 会让第二种错误受到更大的惩罚，因为它离真实值更远。

对于分类任务，一般会用binary entropy loss（二分类任务上），或是cross entropy。

代码实现为：
```python

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```



不同视角下的机器学习分类：
### 1. 监督学习 vs 无监督学习
- **监督学习**：训练数据包含输入和对应的标签（目标值）。算法学习从输入到输出的映射，用于对新数据进行预测。  
  *例子*：KNN 分类/回归、线性回归、决策树、神经网络。  
  *KNN 的位置*：因为有标签，KNN 属于监督学习。  
- **无监督学习**：数据只有输入，没有标签。目标是从中挖掘结构，如聚类、降维、密度估计。  
  *例子*：K-means、PCA、自编码器。  
  *注意*：KNN 本身需标签，但也可作为无监督算法的组件（如用近邻构建相似图），不过通常它被看作监督方法。

### 2. 参数模型 vs 非参数模型
- **参数模型**：假设数据分布具有固定的参数形式，训练时用数据估计有限个参数；预测时仅依赖这些参数，不再需要原始训练数据。模型复杂度通常事先固定。  
  *优点*：计算快，能对小数据做合理推断；*缺点*：若假设与真实不符，效果受限。  
  *例子*：线性回归（参数：权重和偏置）、逻辑回归、朴素贝叶斯。  
- **非参数模型**：不对函数形式做强烈假设，模型复杂度可随数据量增长。常见形式是“记住”训练样本，预测时再用某种相似度决定输出。  
  *优点*：灵活，可拟合复杂边界；*缺点*：需要大量数据，预测慢，易受维度灾难和噪声影响。  
  *KNN 的位置*：典型的非参数模型，预测时需要访问全体训练样本。  
- 与“基于实例的学习”“懒惰学习”的关系：非参数模型常通过存储实例来实现，所以 KNN 也是**基于实例的**和**懒惰的**（训练只存数据，计算推迟到预测时）。相反，像决策树虽然非参数，但会提前构建树结构，属于急切学习。

### 3. 判别模型 vs 生成模型
- **判别模型**：直接学习决策边界或条件概率 $P(y|x)$，关注“给定输入，哪个类别最可能”。  
  *例子*：KNN、逻辑回归、支持向量机、神经网络。  
- **生成模型**：学习数据的联合分布 $P(x, y)$，然后通过贝叶斯定理得到 $P(y|x)$。它能生成新样本，展示数据的底层结构。  
  *例子*：朴素贝叶斯、高斯混合模型、隐马尔可夫模型、生成对抗网络（GAN）。  
  *KNN 的位置*：KNN 直接根据邻居的标签投票，不建模数据是如何生成的，因此是判别模型。

### 4. 分类 vs 回归（按任务类型）
这只是输出类型的不同，不是算法内在的类别，但很多算法两者都能做。
- **分类**：预测离散标签（如猫/狗）。KNN 可用多数投票。
- **回归**：预测连续值（如房价）。KNN 可用近邻标签的均值或加权平均。
- *扩展*：还有聚类（无监督分组）、排序、检测等任务，但 KNN 主要应用于分类和回归。

### 5. 基于距离 vs 基于记忆（常用描述类别）
这些不是严格的学术划分，但常用来刻画算法特性：
- **基于距离的算法**：依赖样本间的距离度量（如欧氏距离）来做出决策。KNN 是最典型的一个，其他如径向基网络、距离加权回归。  
- **基于记忆的学习**：直接存储训练实例，预测时才利用记忆进行推理。几乎等同于懒惰学习。KNN 完全符合，因为它不作抽象，只“回忆”相似样本。
- 

## 5. 小结



> 机器学习是先定义一个可学习的函数空间，再用 loss function 指定目标，通过 optimization 找到一组参数，最后用 validation/test data 判断模型训练的效果，是否真的泛化。


## 参考资料

- ucsd dsc40A,40B等课程
- Andrew Ng: Maching Learning/Deep Learning on Coursera
- [Stanford CS229: Machine Learning](https://cs229.stanford.edu/)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
