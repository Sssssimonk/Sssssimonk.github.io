---
title: 关于这个博客
date: 2026-07-01 22:00:00
updated: 2026-07-01 22:00:00
categories:
  - Blog
tags:
  - AI
  - Research
  - Notes
math: true
mermaid: true
category_bar: true
---

这个博客会主要记录我对 AI 论文、算法思想和工程实践的理解。

我希望每篇文章都尽量回答三个问题：

1. 这篇论文或技术真正解决了什么问题？
2. 它的核心假设、方法设计和实验结论是否站得住？
3. 如果把它放到真实系统或后续研究里，它给我的启发是什么？

后续内容会以论文阅读、方法拆解、实验复现、工程经验和研究想法为主。相比只摘录结论，我会更重视自己的判断：哪里简单有效，哪里可能被过度包装，哪里值得继续追。

一个简单的公式示例：

$$
\mathcal{L}(\theta) = - \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x)
$$

一个代码块示例：

```python
def summarize_claim(paper):
    return {
        "problem": paper.problem,
        "method": paper.method,
        "evidence": paper.experiments,
        "my_take": paper.limitations,
    }
```

