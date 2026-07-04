---
title: "机器学习笔记 02：损失函数、优化与正则化"
date: 2024-06-15 20:30:00
updated: 2024-06-15 20:30:00
categories:
  - 机器学习笔记
tags:
  - 机器学习
  - 损失函数
  - 优化
  - 正则化
  - 笔记
math: true
category_bar: true
---

上一篇笔记里把机器学习看成一个从数据中学习 prediction function 的过程。这篇继续往下看：模型到底是怎么被训练出来的。

训练过程可以先抓住一条主线：

> Loss 定义“错在哪里”，gradient 定义“往哪里改”，learning rate 定义“每次改多少”，regularization 限制模型不要为了降低训练误差而变得过度复杂。

这几个概念连起来，基本就是传统机器学习和深度学习训练的共同底层逻辑。

## 1. 损失函数（loss function）：先定义什么叫错

模型训练不是凭感觉调整参数，而是先把“预测错了多少”变成一个可以计算的数值。这个数值就是 loss。

不同任务会使用不同 loss，因为它们对“错误”的定义不同。

### MSE：回归任务里最常见的 loss

Mean Squared Error 常用于回归任务：

$$
J(\theta) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

它惩罚预测值和真实值之间的平方误差。因为误差被平方，大误差会受到更大惩罚。

例子：

| 真实值 $y$ | 预测值 $\hat{y}$ | 误差 | 平方误差 |
|---:|---:|---:|---:|
| 100 | 90 | 10 | 100 |
| 100 | 50 | 50 | 2500 |

第二个预测只比第一个多错 40，但平方误差大很多。这就是 MSE 对大偏差更敏感的原因。

所以 MSE 的直觉是：不仅希望预测接近真实值，而且特别不希望出现很离谱的预测。

### 二元交叉熵（binary cross entropy）：二分类任务

二分类中，模型通常输出样本属于正类的概率 $\hat{y}$：

$$
L(y, \hat{y}) = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

如果真实标签 $y=1$，这个式子会变成 $-\log(\hat{y})$。模型给正确类别的概率越低，loss 越大。

比如一封邮件确实是垃圾邮件。如果模型预测“是垃圾邮件”的概率是 0.9，loss 很小；如果只给 0.1，loss 会很大。Cross entropy 惩罚的不是类别名本身，而是模型有没有把足够高的概率给到正确类别。

这和分类任务的目标一致：模型不只是要给出类别，还要把概率质量放到正确类别上。

### 多类交叉熵（categorical cross entropy）：多分类任务

多分类任务常用 softmax 把 logits 转成概率分布：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

再用 cross entropy 计算损失：

$$
L(y, \hat{y}) = -\sum_{j=1}^{K} y_j \log(\hat{y}_j)
$$

如果 $y$ 是 one-hot label，这个式子本质上就是取正确类别概率的负对数。

实现时一般会对 logits 做数值稳定处理，例如先减去最大值：

```python
import numpy as np

def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy(probs, labels):
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0 - eps)
    correct_probs = probs[np.arange(len(labels)), labels]
    return -np.mean(np.log(correct_probs))
```

两个容易写错的点：第一，cross entropy 前面有负号；第二，如果输入是 logits，应该先 softmax，而不是直接拿 logits 当概率。

## 2. 梯度下降（gradient descent）：模型如何改参数

有了 loss 后，训练的目标就是找到让 loss 更小的参数。

Gradient descent 的更新规则是：

$$
\theta_{\text{new}} = \theta_{\text{old}} - \alpha \nabla J(\theta)
$$

其中，$\theta$ 是模型参数，$\alpha$ 是 learning rate，$\nabla J(\theta)$ 是 loss 对参数的梯度。

直觉上，梯度指向 loss 增长最快的方向，所以参数更新要往反方向走。

训练时可以把流程想成：

```text
current parameters
-> forward prediction
-> compute loss
-> compute gradient
-> update parameters
-> repeat
```

Learning rate 很关键。如果太小，训练会很慢；如果太大，参数可能来回震荡，甚至不收敛。

比如当前参数只需要往左移动一点点就能降低 loss，但 learning rate 太大，一步跨过了最低点，下一步又从另一边跨回来，loss 就可能来回震荡。

在深度学习里，常见做法是前期 warmup，后期 decay。比如 linear warmup + cosine decay：

- warmup：训练初期逐步增大学习率，避免一开始更新过猛。
- cosine decay：后期逐渐降低学习率，让参数更新更稳定。

在传统机器学习笔记里不需要展开太多 scheduler 细节，但理解它的作用很有用：learning rate 不是一个纯粹的超参数数字，而是在控制训练动态。

## 3. 过拟合（overfitting）与欠拟合（underfitting）

训练模型时常见两类问题。

**Underfitting** 指模型太简单，训练集和测试集都表现不好。模型没有学到足够的规律。

**Overfitting** 指模型在训练集上表现很好，但测试集表现差。模型不只是学到了规律，也记住了训练集里的噪声。

一个常见例子是小数据集上的高阶多项式回归。训练点只有十几个，如果用很高阶的多项式，曲线可以穿过几乎所有训练点，但在训练点之间会剧烈摆动。这种模型看起来 training error 很低，但对新样本不稳定。

可以用 bias-variance 来理解：

| 问题 | 现象 | 直觉 |
|---|---|---|
| High bias | train/test 都差 | 模型太简单 |
| High variance | train 好，test 差 | 模型太依赖训练集 |

很多训练技巧其实都是在这两者之间找平衡。模型太弱要提升表达能力，模型太容易记训练集则要增加约束。

## 4. 正则化（regularization）：限制模型复杂度

Regularization 的作用不是让模型在训练集上 loss 更低，而是限制模型为了降低训练误差而使用过于复杂的参数。

常见方式是在原本 loss 后面加一个惩罚项：

$$
J_{\text{regularized}}(\theta) = J(\theta) + \lambda R(\theta)
$$

其中，$R(\theta)$ 是 regularization term，$\lambda$ 控制正则化强度。

### L1 正则化（L1 regularization）

L1 正则使用参数绝对值之和：

$$
R(\theta) = \sum_i |\theta_i|
$$

它的特点是容易让部分参数变成 0，因此可以产生稀疏性，也常被用于特征选择。

直觉上，如果一个特征贡献不大，L1 会倾向于直接把它的权重压到 0。

### L2 正则化（L2 regularization）

L2 正则使用参数平方和：

$$
R(\theta) = \sum_i \theta_i^2
$$

它会让参数变小，但通常不会直接变成 0。L2 更像是在鼓励模型使用更平滑、更小的权重。

很多时候提到的 weight decay 和 L2 regularization 相关，都是通过限制权重大小来降低模型复杂度。

| 方法 | 作用 | 结果 |
|---|---|---|
| L1 | 惩罚绝对值 | 参数更稀疏，部分权重为 0 |
| L2 | 惩罚平方和 | 参数更小更平滑，通常不为 0 |

简单记法：L1 更像在做“删特征”，L2 更像在做“压权重”。

## 5. 早停（early stopping）：用验证集控制训练

Early stopping 是一种很实用的防过拟合方法。

训练过程中，如果 validation loss 连续若干轮不再下降，就停止训练。关键是看 validation set，而不是只看 training loss。

```text
if validation_loss does not improve for patience steps:
    stop training
```

它的直觉是：当模型继续训练只是在降低 training loss，但 validation loss 开始变差时，模型可能已经开始记训练集噪声了。

Early stopping 的优点是简单有效，缺点是它会把“什么时候停止训练”也变成一个需要调的策略。

## 6. 这几个概念如何连在一起

把 loss、optimization 和 regularization 放在一起看，会比单独记定义更清楚。

Loss 负责定义目标，gradient descent 负责实现参数更新，regularization 负责限制模型复杂度，validation 负责判断模型是不是真的泛化。

可以按这个顺序理解：

```text
loss function: define what is wrong
gradient: tell parameters which direction to move
learning rate: control how far to move
regularization: prevent overly complex parameters
validation: check whether the learned pattern generalizes
```

机器学习训练最核心的一条线：

> 训练模型不是单纯让 loss 越低越好，而是在可控复杂度下，让模型学到能迁移到新数据的规律。

## 7. 几个点

Cross entropy 要有负号。因为概率的 log 通常小于等于 0，前面加负号后 loss 才是非负的。

Sigmoid 的导数是：

$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$

不是 $f(x)f(1-x)$。

ROC 曲线里：

$$
\text{TPR} = \frac{TP}{TP + FN}
$$

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

这几个指标后面写 evaluation 的时候再细分，当前先把定义区分开。

另外，learning rate 大不等于一定能跳出 local minimum。过大的 learning rate 也可能导致训练不稳定或不收敛。更准确的说法是，优化非凸函数时需要结合初始化、学习率策略、优化器和模型结构一起看。

## 8. 小结

Loss、optimizer、regularization 不应该当成分散知识点，而应该放在一条训练链路里理解：

- loss 决定模型追求什么；
- gradient 决定参数怎么被修改；
- learning rate 决定修改幅度；
- regularization 决定模型不能为了训练集误差任意复杂；
- validation 决定这个训练结果是否值得相信。

这条链路也能迁移到深度学习和大模型训练里。神经网络复杂很多，但底层仍然是在定义目标、计算梯度、更新参数、控制泛化。

## 参考资料

- [Stanford CS229: Machine Learning](https://cs229.stanford.edu/)
- [scikit-learn User Guide: Model selection and evaluation](https://scikit-learn.org/stable/model_selection.html)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
