---
title: "Agent 系统笔记 01：从单次打分到 Macro Eval"
date: 2026-08-22 21:00:00
updated: 2026-08-23 23:30:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - Agent
  - Evaluation
  - Macro Evals
  - Trace
  - LLM-as-a-Judge
math: true
category_bar: true
---

普通问答的 eval 很容易想象：给模型一道题，对照标准答案打分。到了 Agent，这个思路会迅速失效。

一个 Agent 可能最后给出了正确回复，却调用了错误的工具；也可能过程基本合理，只在一次 handoff 上走错；还有些故障单独看只是偶发，放到几千条 trace 里才会发现它一直在同一类场景下重现。

Anthropic 的《Demystifying evals for AI agents》讲的是怎样把一条 Agent run 评清楚；OpenAI Cookbook 的《Macro Evals for Agentic Systems》进一步问：当我们已经积累了许多 run，怎样从局部评分走到系统诊断？这两篇放在一起，刚好组成了一条完整链路。

<!-- more -->

## Agent eval 评的不是最后一句话

先区分几个对象：

- **Task**：一个测试任务，包含输入和成功标准；
- **Trial**：Agent 对同一个 task 的一次实际尝试；
- **Transcript / Trace**：这次尝试的完整过程，包括模型输出、工具调用、handoff 和环境反馈；
- **Outcome**：运行结束时，外部环境真正变成了什么；
- **Grader**：检查 outcome 或 trace 某个方面的评分逻辑；
- **Harness**：负责运行任务、保存轨迹、调用 grader 和汇总结果的基础设施。

这里最容易混淆的是 transcript 和 outcome。

一个订票 Agent 可以在消息里说“航班已经预订成功”，这是 transcript 的最后一句；数据库里是否真的出现了 reservation，才是 outcome。只评自然语言回复，会把“说自己做完了”误当成“真的做完了”。

因此 Agent eval 至少要有两类视角：

$$
\text{success}=f(\text{final state},\text{execution trace})
$$

最终状态回答“事情办成了吗”，执行轨迹回答“它是怎么办的，有没有违反约束”。

## 一个 task 通常需要多个 grader

Agent 任务的成功很少是单一维度。以退款 Agent 为例，我们可能同时要求：

- 退款记录确实写入数据库；
- 退款金额不超过规则上限；
- 必须先调用身份验证工具；
- 全程不超过 10 轮；
- 回复语气清楚，没有编造政策。

这些要求适合不同的 grader。

### Deterministic grader

能用代码检查的，尽量先用代码检查：单元测试、数据库状态、文件 diff、JSON schema、工具调用参数、静态分析等。

它们便宜、稳定、容易复现。缺点是只能检查被明确写出来的条件，不能很好地判断开放式质量。

### Model-based grader

对解释是否清楚、研究报告是否完整、客服语气是否恰当等问题，可以用 rubric-based LLM judge、pairwise comparison 或 reference-based evaluation。

它更灵活，但本身也是一个有随机性和偏差的模型。需要拿人工评分校准，而不能因为输出了一个小数就把它当成客观仪器。

### Human grader

专家审阅仍然是高质量标准，尤其适合定义 rubric、处理争议样本和抽查 LLM judge。问题也很直接：慢、贵，而且不同专家之间也未必一致。

比较实际的组合是：代码 grader 负责硬约束，model grader 负责难以形式化的质量，人工负责校准和审计。不要让一个 judge prompt 承包所有事情。

## Capability eval 和 regression eval 不是同一套题

Anthropic 区分了两种常被混用的 eval。

Capability eval 关心“系统能力的上限在哪里”。题目应该有一定难度，初始通过率偏低，留出爬坡空间。如果所有题都能过，它就无法区分新方法是否真的更强。

Regression eval 关心“已经会的事情有没有被改坏”。这里反而希望通过率长期很高，一次已知故障被修好后，最好就把它加入回归集。

两者的题目分布和阈值不同。用一套满分回归集证明 Agent 能力领先，或者用一套刻意刁钻的 capability set 卡发布，都会让 eval 失去本来的用途。

## 非确定性：一次成功还不够

同一个 Agent、同一个 task，多跑几次可能得到不同结果。所以只有 task accuracy 还不够，还要说明每题跑了多少 trial。

两个常用指标是 $pass@k$ 和 $pass^k$。

$pass@k$ 表示 $k$ 次尝试里至少成功一次的概率：

$$
pass@k = 1-(1-p)^k
$$

$pass^k$ 表示 $k$ 次尝试全部成功的概率：

$$
pass^k=p^k
$$

如果单次成功率 $p=0.75$，三次里至少成功一次的概率约为 98.4%，但三次都成功只有约 42.2%。

对于代码候选生成，能多试几次并选出一个可用解，$pass@k$ 很合理；对于每次都直接面对用户的客服 Agent，$pass^k$ 更接近真实要求。一个 Agent 可以同时拥有很高的 $pass@k$ 和很差的稳定性。

## 从 20 到 50 个真实失败开始

一套好的 eval 不一定从几百道精心合成的题开始。Anthropic 给出的建议很朴素：早期先收集 20 到 50 个任务，来源就用开发中手动检查的行为、bug tracker、客服工单和用户真实失败。

然后把任务写得足够明确：初始状态是什么，允许使用什么工具，完成后环境里应该出现什么，哪些路径是禁止的。

这里有一个实践上很有用的判断：如果人类看完 task specification 都无法稳定判断是否成功，grader 也很难凭空稳定。先修任务定义，再调 judge prompt。

另外要防两类数据问题：

- **过于干净**：只有标准 happy path，完全没有真实世界的歧义、工具失败和边界条件；
- **被反复优化穿**：团队盯着固定 eval set 调 prompt，最后提升的是对这几十道题的适配，而不是产品能力。

保留 holdout、不断把新生产故障加入集合，比执着于一份永远不变的 benchmark 更实际。

## 单条 eval 的尽头，是一堆局部标签

做到这里，我们可以知道：某次 run 的数据库状态错了，某个工具调用漏了参数，某次 handoff 不合规。

但系统维护者真正想回答的往往是：

- 最近哪个故障模式增长最快？
- 它集中在哪个 Agent 版本或场景？
- 是某个 specialist 的问题，还是 orchestrator 路由的问题？
- 先修哪一类问题收益最大？

这就是 macro eval 的入口。Lower-level eval 产生局部信号，macro eval 在大量 trace 上寻找重复结构。

OpenAI Cookbook 用四类标签把层次分开：

- `case_type`：输入是什么业务情形；
- `run_outcome`：这次运行最后怎样结束；
- `eval_finding`：局部 grader 发现了什么症状；
- `behavior_pattern`：许多 trace 中反复出现的行为模式。

可以把它们理解成：开局、结局、局部症状、群体模式。`eval_finding` 和 `behavior_pattern` 尤其不能混为一谈。一个是已有规则发现的局部问题，一个是跨样本后才显现的结构。

## Macro Eval 第一步：把 trace 变成可比较的数据

原始 trace 往往混着模型长文本、工具 payload、状态更新和各种日志，直接聚类通常只会学到长度、模板和高频噪声。

Cookbook 先把数据规范化成两张表：

- `traces_df`：一行对应一次 run，保存场景、结局、版本、findings 等汇总信息；
- `events_df`：一行对应一个 event，保存时序、Agent、工具、handoff 与状态变化。

然后为每条 trace 构造一个 compact document。它不是把日志粗暴截断，而是有选择地保留：场景、关键环境信号、Agent 激活顺序、重要工具结果、review 标记、最终状态和 lower-level finding。

这一步可能比后面的聚类算法更重要。表示错了，UMAP 和 HDBSCAN 调得再精细，也只能稳定地聚类错误特征。

## 从许多 trace 里发现 behavior pattern

Cookbook 采用一条 BERTopic 风格的流程：

1. 将每份 compact trace document 编码成 embedding；
2. 用 UMAP 将向量降维，保留局部邻域；
3. 用 HDBSCAN 找密集区域，并允许一部分 trace 成为 noise；
4. 为每个 cluster 提取有区分度的词，形成可读标签。

这不是在发明一套永恒正确的故障 taxonomy，而是把上千条运行压缩成几个值得人工查看的候选模式。

发现模式后还要排序。文章用 prevalence 和 severity 构造 impact score：

$$
\text{impact}=\text{prevalence}\times\text{severity-weighted prevalence}
$$

高 impact 只代表它频繁、严重或两者兼有，不代表它一定是软件 defect，更不代表聚类已经证明了根因。它是 triage board，不是判决书。

## 从 pattern 回到可能的上游原因

聚类告诉我们“有一组相似失败”，仍然没有告诉我们从哪里开始查。

Cookbook 接着把 trace 表示为执行图 $G=(V,E)$。节点是规范化 event，边表示时间顺序、handoff、工具调用和邻近上下文。选定一个 failure、review finding 或 late-stage decision 作为 anchor，再沿图向前回溯上游事件。

示例中的 suspect score 是一个有意保持可解释的启发式：

$$
\text{suspect}=0.4\,\text{proximity}
+0.3\,\text{frequency}
+0.2\,\text{bridge}
+0.1\,\text{role}
$$

离故障近、在同类 trace 中反复出现、连接多个阶段、角色上又与问题相关的事件，会排在前面。

仍然要强调：这是 inspection priority，不是 causal proof。真正的根因要靠回放、对照实验、版本切片或直接修改后再跑 eval 来确认。

## 两篇文章合起来，其实是一套闭环

整个流程可以写成：

$$
\text{task}
\rightarrow \text{trials}
\rightarrow \text{outcome/trace graders}
\rightarrow \text{local findings}
\rightarrow \text{population patterns}
\rightarrow \text{suspect components}
\rightarrow \text{new regression tests}
$$

Anthropic 的文章解决前半段：怎样定义一次 Agent 成功，以及怎样得到可信的局部评分。OpenAI Cookbook 解决后半段：怎样不被大量局部评分淹没，把它们转成系统级维护线索。

我觉得这里最重要的变化是，eval 不再只是发布前的一张总分表。它开始承担 observability 和 debugging 的作用。

准确率告诉你系统有没有变好；trace-level grader 告诉你哪次运行哪里不对；macro eval 则告诉你，整个系统最值得先查的重复问题是什么。对于会调用工具、改变环境、运行很多步的 Agent，最后一种能力可能比再多一个 benchmark 分数更有用。

## 参考资料

- Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- OpenAI Cookbook, [Macro Evals for Agentic Systems](https://developers.openai.com/cookbook/examples/partners/macro_evals_for_agentic_systems/macro_evals_for_agentic_systems)

