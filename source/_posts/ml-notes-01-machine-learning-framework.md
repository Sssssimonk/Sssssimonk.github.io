---
title: "机器学习笔记 01：机器学习的基本框架"
date: 2024-06-08 20:00:00
updated: 2024-06-08 20:00:00
categories:
  - 机器学习笔记
tags:
  - 机器学习
  - 监督学习
  - 泛化
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

如果是房价预测，$x$ 可以是面积、地段、楼层、房龄，$y$ 是真实房价。如果是邮件分类，$x$ 可以是邮件文本的特征表示，$y$ 是是否垃圾邮件。如果是推荐系统，$x$ 可以是用户、物品和上下文特征，$y$ 可以是点击、购买或停留时长。

所以机器学习不是直接“学一个答案”，而是学习一个从输入到输出的映射关系。

一个很小的例子：

| 样本 | 面积 $x_1$ | 房龄 $x_2$ | 真实房价 $y$ |
|---|---:|---:|---:|
| A | 80 | 5 | 500 |
| B | 60 | 15 | 320 |
| C | 100 | 2 | 720 |

模型可能会先学一个很简单的函数：

$$
\hat{y} = w_1x_1 + w_2x_2 + b
$$

训练过程就是不断调整 $w_1$、$w_2$ 和 $b$，让预测房价 $\hat{y}$ 尽量接近真实房价 $y$。

这个循环就是训练的核心：模型先预测，loss 衡量预测错了多少，optimizer 根据 loss 更新参数，然后模型再预测。

## 2. 监督学习（supervised learning）和无监督学习（unsupervised learning）

机器学习通常先按有没有 label 来分。

**Supervised Learning** 有明确的输入和目标输出。模型看到的是 $(x, y)$，目标是学会从 $x$ 预测 $y$。常见任务包括 regression 和 classification。

Regression 预测连续值，比如房价、温度、点击率。Classification 预测类别，比如垃圾邮件识别、疾病诊断、图片分类。

**Unsupervised Learning** 没有明确的 label。模型只看到 $x$，目标是发现数据内部结构。常见任务包括 clustering、dimensionality reduction 和 density estimation。

比如 KMeans 会尝试把样本分成若干簇，PCA 会尝试找到数据中方差最大的方向。这类方法不直接回答“这个样本属于哪个真实标签”，而是帮助整理数据结构。

简单说，supervised learning 更像是在学一个带答案的映射函数；unsupervised learning 更像是在没有标准答案的情况下整理数据的内在结构。

## 3. 数据集划分为什么重要

机器学习里经常会把数据分成：

| 数据集 | 作用 |
|---|---|
| Training set | 用来训练模型参数 |
| Validation set | 用来调超参数、选模型 |
| Test set | 用来估计最终泛化能力 |

不能只看 training set 表现，因为模型可以通过记住训练样本获得很低的 training loss，但这不代表它真的学到了规律。

这就是 overfitting 的核心问题：模型在训练集上表现很好，但在新数据上表现很差。

比如一个垃圾邮件分类器，如果它只是记住了训练集中出现过的垃圾邮件地址，那么训练集 accuracy 可能很高。但新垃圾邮件换了发件人和措辞后，它就可能失效。真正有用的模型应该学习“垃圾邮件通常有哪些模式”，而不是背下训练样本。

所以 test set 的意义不是“再训练一次”，而是模拟模型未来遇到没见过数据时的表现。

## 4. 损失函数（loss function）在做什么

训练模型前，第一步是定义什么叫“错”。这个定义就是 loss function。

对于回归任务，常用 Mean Squared Error：

$$
J(\theta) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

其中，$y_i$ 是真实值，$\hat{y}_i$ 是预测值。预测和真实值差得越远，loss 越大。

比如真实房价是 500，模型预测 450，误差是 50；如果预测 300，误差是 200。MSE 会让第二种错误受到更大的惩罚，因为它离真实值更远。

对于分类任务，常用 Cross Entropy。它关心的是模型有没有把概率分配给正确类别。

直觉上，loss function 规定了模型应该为什么事情付出代价。MSE 惩罚数值误差，cross entropy 惩罚正确类别概率太低，hinge loss 则强调分类边界要有 margin。

因此，不同 loss 不是公式长得不同而已，而是在表达不同的训练目标。

## 5. 模型训练的本质

机器学习训练可以粗略理解成最小化 empirical risk：

$$
\theta^* = \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n} L(f_\theta(x_i), y_i)
$$

这句话的意思是：在所有可能的参数 $\theta$ 中，找到一组参数，让模型在训练数据上的平均 loss 尽可能小。

但关键问题在于：真正关心的不是训练数据上的 loss，而是未来数据上的 loss。训练数据只是总体数据分布的一个样本。

所以机器学习一直在处理一个张力：

- 训练集上 loss 要足够低，否则模型没有学到东西。
- 模型不能只适应训练集，否则泛化能力差。

这也是为什么后面会有 regularization、cross validation、early stopping、model selection 等一整套方法。它们都在服务同一个目标：让模型不仅会做训练题，也能做新题。

## 6. 一个完整的机器学习流程（ML pipeline）

从工程角度看，一个机器学习流程通常不是“选个模型训练一下”这么简单。

一个典型流程是：

```text
data collection
-> data cleaning
-> feature engineering
-> train / validation / test split
-> model training
-> model selection
-> evaluation
-> deployment or analysis
```

每一步都可能决定最终效果。比如在房价预测里，如果房屋面积单位有的用平方米、有的用平方英尺，而清洗阶段没有统一单位，后面再复杂的模型也会被错误数据带偏。

数据清洗会影响输入质量，feature engineering 会影响模型能看到什么信息，数据划分会影响评估是否可靠，评价指标会影响最后选哪个模型。

在实际项目里，模型算法本身经常不是唯一瓶颈。数据质量、特征设计、评估方式、线上线下分布差异，往往同样重要。

## 7. 小结

一句话概括机器学习的基础框架：

> 机器学习是先定义一个可学习的函数空间，再用 loss function 指定目标，通过 optimization 找到一组参数，最后用 validation/test data 判断它是否真的泛化。

这个框架比记算法名字更重要。KNN、决策树、逻辑回归、XGBoost、神经网络看起来差别很大，但都绕不开几个问题：

- 输入特征是什么？
- 目标变量是什么？
- 模型假设是什么？
- loss 如何定义？
- 参数如何学习？
- 如何判断模型不是在过拟合？

后面看具体算法时，也尽量从这些问题出发，而不是只背每个模型的定义。

## 参考资料

- [Stanford CS229: Machine Learning](https://cs229.stanford.edu/)
- [MIT 6.036 Introduction to Machine Learning](https://ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
