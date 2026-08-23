---
title: "Agent 系统笔记 02：Agent 不等于无限循环，从 Workflow 到 Context Engineering"
date: 2026-08-23 21:00:00
updated: 2026-08-23 23:30:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - Agent
  - Workflow
  - Context Engineering
  - Tool Use
  - Anthropic
math: false
category_bar: true
---

Agent 经常被画成一个循环：模型思考，调用工具，读取结果，再思考，直到完成任务。这个图没错，但几乎没有设计信息。

真正做系统时，要先回答两个问题：任务的控制流究竟应该由代码决定，还是由模型临场决定？每次让模型做决定时，它又应该看到哪些信息？

Anthropic 的《Building Effective Agents》主要回答第一个问题，《Effective context engineering for AI agents》回答第二个。前者讨论系统怎样走，后者讨论模型每一步拿什么来走。

<!-- more -->

## Workflow 和 Agent 的分界是控制权

Anthropic 给出的区分很实用：

- **Workflow**：LLM 和工具沿预先定义的代码路径运行；
- **Agent**：LLM 动态决定接下来做什么、调用什么工具，以及任务需要多少步。

关键不在于调用了几次模型，也不在于有没有循环，而在于控制流属于谁。

一个“先抽取信息，再查数据库，最后生成回复”的三段式系统，即使用了三个 LLM，也仍然是 workflow。反过来，一个模型读完任务后自己决定先搜索还是先写代码、要改几个文件、何时回头检查，它更接近 Agent。

这条分界可以避免把所有 LLM application 都包装成 Agent。很多任务的步骤本来就稳定，硬把控制权交给模型，只会增加延迟、成本和新的失败路径。

## 起点不是多 Agent，而是 augmented LLM

最基础的 building block 是 augmented LLM：一个模型，加上 retrieval、tools 和 memory。

这一步做好以后，许多看似需要复杂框架的问题已经能解决。模型可以查资料、读文件、执行代码，也能保存少量状态。

Anthropic 对 framework 的态度也比较克制。框架能省掉工具 schema、调用循环和消息管理的样板代码，但额外抽象层也会遮住真正的 prompt、tool response 和控制流。出错时如果看不到底层发生了什么，就很难判断是模型、工具还是框架默认行为的问题。

所以系统复杂度最好由任务失败倒逼出来，而不是从一张完整的 Agent 架构图开始。

## 五种常见 Workflow，各自解决什么

### Prompt chaining：把一个难调用拆成几个简单调用

Prompt chaining 让前一步的输出成为后一步输入，中间可以插入程序化 gate。

例如先生成文章提纲，检查提纲是否覆盖指定主题，通过后再写正文。它适合能稳定拆成固定子任务的流程，用额外延迟换取每一步更聚焦。

它不适合子任务结构本身未知的工作。若任务每次需要的步骤都不同，提前写死 chain 会越来越多分支。

### Routing：先识别类型，再交给专门路径

Routing 先对输入分类，再选择不同 prompt、工具集或模型。

客服中的退款、技术支持和普通问答可以分开；简单问题交给便宜模型，困难问题交给更强模型，也是一种 routing。

它的前提是类别相对清楚，并且 router 本身足够可靠。路由错一次，后面专门化得越好，可能反而错得越坚定。

### Parallelization：并行拆分，或并行投票

并行化有两种不同目的。

**Sectioning** 是把互相独立的子任务同时执行，例如多个模型分别检查正确性、安全性和风格。它主要节省时间，也避免一个 prompt 同时承担太多注意事项。

**Voting** 是让多个调用处理同一个问题，再聚合不同结果，例如多次代码安全审查。它主要用额外计算提高覆盖率或置信度。

如果子任务有强依赖，sectioning 会制造重复工作；如果错误高度相关，多投几票也不会得到真正独立的判断。

### Orchestrator-workers：子任务无法提前写死

中心 orchestrator 先阅读具体任务，动态拆解，再把子任务分给 workers，最后汇总结果。

它看起来和 parallelization 类似，区别是子任务并非开发者预定义，而是 orchestrator 根据输入现拆。跨多个文件的代码修改、开放式资料搜索都比较适合：开始之前往往不知道要碰哪些文件、查哪些方向。

风险也来自同一个地方。拆解错了，workers 做得再认真也只是高质量地完成了错误子任务。因此 orchestrator 的计划、worker 结果和最终综合都需要可观察。

### Evaluator-optimizer：有明确标准时再迭代

一个模型生成，另一个模型按标准评价并给反馈，然后循环改进。

它适合两件事同时成立的场景：人类指出问题后，答案确实能明显改善；模型也能较稳定地给出这类反馈。例如文学翻译、需要多轮补充证据的研究任务。

如果评价标准含糊，evaluator 只会制造一种“再改一遍总会更好”的错觉。循环次数增加了，质量未必增加。

## 什么时候才需要真正的 Agent

开放式任务如果无法预先知道步骤数，也无法硬编码路径，Agent 才开始体现价值。典型例子是复杂编码任务、长时间调查、在真实环境中逐步排障。

一个可用的 Agent loop 至少需要：

1. 明确当前目标；
2. 根据状态选择下一步动作；
3. 从工具或环境获得 ground truth；
4. 根据反馈修正计划；
5. 在完成、阻塞或达到停止条件时退出。

其中第三步最容易被低估。Agent 不是只和自己的文字继续对话，而是要不断从环境拿到可验证反馈：测试有没有通过、文件是否真的修改、订单是否真的创建。没有 ground truth 的长循环，很容易变成模型对自己上一步说法的反复加工。

自主性也会带来错误累积和成本增长。所以 sandbox、最大迭代数、权限边界、人工 checkpoint 都不是外围补丁，而是 Agent 设计的一部分。

## 从 Prompt Engineering 到 Context Engineering

当控制流确定以后，另一个问题出现了：模型在每一步应该看到什么？

Prompt engineering 往往关注一句指令怎样表达；context engineering 关注的是一次采样时，整个 token 集合怎样配置：system prompt、对话历史、检索文档、工具说明、工具结果、记忆和当前工作状态。

模型某一步的行为可以粗略理解为：

```text
behavior = model(system prompt + history + retrieved context
                 + tool definitions + tool results + current state)
```

其中任何一项都可能让行为改变。只盯着 system prompt，很容易在错误的位置调参。

## Context 是有限的 attention budget

上下文窗口变长，不代表所有 token 都能得到同等有效的利用。信息越来越多时，相关线索可能被旧日志、重复说明和大段工具输出淹没。这就是 context pollution；随着上下文增长而出现的性能下降，有时也被称为 context rot。

所以 context engineering 的目标不是把一切都塞进去，而是找到足以支持当前决策的最小高信号 token 集合。

这里的“最小”不等于越短越好。任务约束、关键例子和必要背景仍然要完整。真正该删的是重复、陈旧、可随时重新获取、与当前步骤无关的内容。

## System prompt 要在合适的抽象高度

过于具体的 prompt 会变成脆弱的 if-else 清单，遇到没写过的情况就失效；过于抽象的 prompt 只有“请高质量完成任务”，又不给模型任何可操作边界。

比较合适的 system prompt 应该说明角色、目标、关键约束和决策原则，同时把具体事实留给环境与工具。可以先用强模型测试一个最小版本，根据真实失败逐项补充，而不是预先写一部假想中的百科全书。

Few-shot example 也一样。少量覆盖典型边界的例子通常比几十条相似样本更有用。例子的目的不是占满 context，而是给出难以用抽象规则表达的行为坐标。

## Tool 既是动作接口，也是 context 接口

工具设计经常只考虑“这个 API 能不能执行动作”，却忘了 tool response 会回到上下文，继续影响模型。

一个返回整份数据库 dump 的工具即使功能正确，也可能把后续推理淹没。更好的工具应该：

- 名称和参数含义明确；
- 功能之间尽量少重叠；
- 默认返回紧凑、结构化、与决策有关的信息；
- 支持过滤、分页和按需展开；
- 错误信息能指导下一步，而不是只给一个异常码。

如果人类工程师都说不清某个场景应该用哪个工具，就不该期待模型在一组重叠工具中稳定选择。

因此 tool design 本身就是 context engineering。工具决定 Agent 能拿到什么，也决定它要为这些信息付出多少 token。

## Just-in-time retrieval：保留索引，需要时再读取

一种常见做法是任务开始前把所有可能相关的资料都检索进 context。对稳定、规模小的知识库还可以；对代码库、邮件或不断变化的数据，这会同时带来过期和噪声。

Just-in-time retrieval 更像人类使用文件系统：先保留路径、链接、查询名和少量 metadata，需要时再通过工具读取具体内容。

文件名、目录层级和更新时间本身也是信息。`tests/test_utils.py` 和 `src/core_logic/test_utils.py` 即使同名，位置也暗示了不同用途。

这并不意味着传统 retrieval 没用。更实际的是混合方式：少量稳定的核心规则预先放入 context，动态且体量大的信息按需取回。

## 长任务中的三个办法

### Compaction

当历史接近窗口上限时，先将其压缩成高保真摘要，再用摘要和最近工作继续。

难点不是“会不会总结”，而是哪些细节现在看似无关、后来却会成为关键。比较安全的调法是先追求 recall，确保架构决定、未解决问题、已试失败方案和当前状态都保留，再逐步删掉冗余。

较低风险的第一步通常是清理很久以前的原始 tool result。只要关键结论已经进入当前状态，几百行旧命令输出通常没必要永久占着窗口。

### Structured note-taking

让 Agent 把计划、进度、关键决定和未解决问题写到 context 外的持久笔记中，需要时再读回来。

一个简单的 `NOTES.md` 或 task state 往往比“给 Agent 加一个神秘的长期记忆模块”更可靠。结构化笔记的价值在于，它把状态从不断增长的聊天历史中剥离出来。

### Multi-agent / subagent

把彼此独立、context 很重的子任务放到独立上下文执行，主 Agent 只接收压缩后的结果。这样可以隔离大量搜索和工具输出，也能并行工作。

但 subagent 不是免费的上下文垃圾桶。任务边界不清时，它会重复调查、丢失隐含约束，最后主 Agent 还要花更多 token 对齐结果。只有子任务确实能独立描述时，这种隔离才划算。

## 两篇文章合起来的设计顺序

我更愿意把这两篇文章整理成下面的顺序：

1. 先用单次 LLM 调用加必要工具完成任务；
2. 固定步骤明显时，选择 chaining、routing 或 parallelization；
3. 子任务结构依输入变化时，再用 orchestrator-workers；
4. 评价标准清楚且迭代确实有收益时，加入 evaluator-optimizer；
5. 只有路径和步数无法预先确定时，才把控制权交给 Agent；
6. 无论哪一种结构，都持续管理每一步的 context，而不是无限累积历史。

所以“Agent 不等于无限循环”有两层意思。

第一，不是所有任务都需要自主循环，很多问题用可控 workflow 更稳。第二，即使确实需要循环，也不能让上下文无边界地一起循环。控制流要有停止条件，context 也要有取舍、压缩和外部状态。

Agent 工程最后并不神秘：把该由代码决定的事情留给代码，把需要语义判断的部分交给模型；每当模型需要判断时，再给它当时真正需要的信息。

## 参考资料

- Anthropic, [Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents)
- Anthropic, [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

