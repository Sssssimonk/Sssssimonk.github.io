---
title: "ML Notes 06: Evaluation、Validation 与 Model Selection"
date: 2026-07-03 22:30:00
updated: 2026-07-03 22:30:00
categories:
  - Machine Learning Notes
tags:
  - Machine Learning
  - Evaluation
  - Cross Validation
  - Model Selection
  - Notes
math: true
category_bar: true
---

这一篇整理模型评估、验证和模型选择。训练 loss 下降只能说明模型更适合训练数据，不等于模型真的有用。评估指标和验证方式决定了最后会选择哪个模型，也决定了这个选择是否可靠。

## 1. 评估指标不是装饰

模型评估的核心问题是：

```text
这个模型在真正关心的场景里，到底有没有变好？
```

不同任务里，“好”的定义不一样。房价预测关心预测误差，垃圾邮件识别关心别把正常邮件误删，疾病筛查关心别漏掉阳性病例。

所以指标不是越多越好，而是要对应任务代价。

## 2. Regression Metrics

回归任务预测连续值，常见指标有 MSE、RMSE、MAE、$R^2$。

### MSE 和 RMSE

MSE：

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

RMSE：

$$
RMSE=\sqrt{MSE}
$$

MSE 对大误差更敏感，因为误差被平方。RMSE 和原始目标变量单位一致，更容易解释。

比如预测房价时，RMSE = 20 万，比 MSE = 400 亿更直观。

### MAE

MAE：

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

MAE 对 outlier 没有 MSE 那么敏感。

如果数据里有少数极端样本，比如一两套超级豪宅，MSE 会被这些样本强烈影响，MAE 会更稳一些。

### $R^2$

$R^2$ 衡量模型解释了多少目标变量的 variance：

$$
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

如果 $R^2=0.8$，可以粗略理解为模型解释了 80% 的变化。它适合看整体拟合程度，但不一定能说明具体业务误差是否可接受。

## 3. Classification Metrics

分类任务常见指标都可以从 confusion matrix 出发。

|  | Predicted Positive | Predicted Negative |
|---|---:|---:|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

TP 是真正例，FP 是假正例，FN 是假负例，TN 是真负例。

### Accuracy

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

Accuracy 直观，但在 imbalanced dataset 上容易误导。

比如一个疾病数据集中，99% 都是阴性。模型全部预测阴性，也有 99% accuracy，但它完全找不到阳性病人。

### Precision

$$
Precision=\frac{TP}{TP+FP}
$$

Precision 关心预测为正的样本里，有多少是真的正。

垃圾邮件识别里 precision 很重要，因为把正常邮件误判成垃圾邮件会影响用户。

### Recall

$$
Recall=\frac{TP}{TP+FN}
$$

Recall 关心真实正样本里，有多少被找出来。

疾病筛查里 recall 很重要，因为漏诊的代价可能很高。

### F1 Score

$$
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
$$

F1 是 precision 和 recall 的调和平均。它适合在两者都重要时作为综合指标。

但 F1 也不是万能的。如果业务上 FP 和 FN 代价明显不同，还是要单独看 precision 和 recall。

## 4. ROC 和 AUC

ROC 曲线关注不同 threshold 下 TPR 和 FPR 的变化。

$$
TPR=\frac{TP}{TP+FN}
$$

$$
FPR=\frac{FP}{FP+TN}
$$

AUC 是 ROC 曲线下面积。它也可以理解为：随机抽一个正样本和一个负样本，模型把正样本排在负样本前面的概率。

这就是为什么 AUC 常用于排序、推荐、广告等场景。很多时候模型输出的不是最终类别，而是一个 score，业务再根据 threshold 或排序位置使用这个 score。

一个小例子：

| 用户 | 是否点击 | 模型 score |
|---|---:|---:|
| A | 1 | 0.90 |
| B | 0 | 0.80 |
| C | 1 | 0.70 |
| D | 0 | 0.20 |

AUC 关心的是正样本是否整体排在负样本前面，而不是某个固定 threshold 下分对了多少。

## 5. PR Curve

PR Curve 的横轴是 recall，纵轴是 precision。

在正负样本极度不平衡时，PR curve 往往比 ROC curve 更敏感。

比如欺诈检测里，正样本非常少。ROC-AUC 可能看起来还不错，但 precision 可能很低：模型找出一堆“疑似欺诈”，里面真正欺诈的比例很小。这个时候 PR curve 更能反映实际使用体验。

## 6. Imbalanced Dataset

类别不平衡时，accuracy 通常不够。

常见处理方式：

- 换指标：precision、recall、F1、AUC、PR-AUC
- 重采样：oversampling / undersampling
- 调整 loss：给少数类更高权重
- 调 threshold：根据业务代价调整预测阈值
- anomaly detection：把少数类当异常点处理

一个例子：阳性率只有 1% 的疾病筛查。模型如果全部预测阴性，accuracy 是 99%，但 recall 是 0。这种模型没有实际价值。

## 7. Validation：怎么可靠地估计泛化能力

训练集表现不能代表泛化能力，所以需要 validation。

### Holdout Validation

最常见方式是 train / validation / test split。

```text
training set: train model parameters
validation set: tune hyperparameters and choose models
test set: final evaluation
```

test set 不应该频繁用来调模型。如果每次调参都看 test score，test set 也会被间接过拟合。

### K-fold Cross Validation

K-fold cross validation 把数据分成 $K$ 份，每次用其中一份做 validation，其余做 training，最后平均结果。

它比单次 holdout 更稳定，尤其适合数据量不大的情况。

缺点是训练成本更高。比如 5-fold 就要训练 5 次。

### Bootstrap

Bootstrap 是有放回采样。每次从原始数据中采出一个同样大小的数据集，有些样本会重复，有些样本不会被采到。没被采到的样本可以用于评估。

它在统计估计里很常见，但日常模型训练中更常用 holdout 和 K-fold。

## 8. Model Selection

Model selection 不只是选模型类型，也包括选超参数。

比如：

- Linear Regression vs Random Forest vs XGBoost
- Decision tree 的 `max_depth`
- KNN 的 `K`
- Logistic regression 的 regularization strength
- XGBoost 的 learning rate、tree depth、number of estimators

### Grid Search

Grid search 枚举所有组合。

```text
max_depth = [3, 5, 7]
learning_rate = [0.01, 0.1]
```

总共会试 $3 \times 2 = 6$ 种组合。

优点是简单直接，缺点是组合多时成本很高。

### Random Search

Random search 从参数空间中随机采样。它经常比 grid search 更高效，因为并不是所有超参数都同等重要。

比如 10 个超参数里真正敏感的可能只有 2 个，grid search 会浪费很多组合，random search 反而可能更快碰到好区域。

### Bayesian Optimization

Bayesian optimization 会根据已有实验结果，估计哪些参数区域更可能好，然后优先尝试这些区域。

它比 random search 更聪明，但实现和调试更复杂。小任务里不一定需要，上规模后才更有价值。

## 9. Data Leakage

Data leakage 是评估里最危险的问题之一。它指训练过程中用了不该提前知道的信息。

例子：预测用户未来是否流失，但 feature 里包含“用户是否已经取消订阅”。这个特征在预测时根本不可用，但训练时会让模型表现非常好。

另一个例子：先对全量数据做 normalization，再划分 train/test。这样 test set 的统计信息已经泄漏到了 training process。正确做法是只在 training set 上 fit scaler，再 apply 到 validation/test set。

Data leakage 会让离线评估虚高，线上效果崩掉。

## 10. 几个点

Accuracy 不适合严重类别不平衡的问题。

Precision 和 recall 要结合 FP/FN 的业务代价看，不要机械比较谁更大。

AUC 更像排序指标，适合看 score 的整体区分能力。

PR curve 在正样本稀少时更敏感。

Validation set 用来选模型，test set 留到最后做最终评估。

数据泄漏比模型小 bug 更可怕，因为它会让结果看起来特别好。

## References

- [scikit-learn User Guide: Model Selection and Evaluation](https://scikit-learn.org/stable/model_selection.html)
- [scikit-learn User Guide: Metrics and Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
