---
title: "现代大模型笔记 03：LLM 推理与效率优化"
date: 2025-09-20 21:00:00
updated: 2025-09-20 21:00:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - 大模型
  - 推理优化
  - KV cache
  - vLLM
  - DeepSpeed
  - FlashAttention
  - 笔记
math: true
category_bar: true
---

这一篇整理 LLM inference。

训练阶段关心的是怎么把参数学出来；推理阶段关心的是怎么把模型稳定、低延迟、高吞吐地跑起来。

LLM 推理不像普通分类模型那样一次 forward 就结束。它是自回归生成：

```text
prefill prompt -> decode token 1 -> decode token 2 -> ...
```

所以它的系统瓶颈很特别：一边要算 attention，一边要维护 KV cache，一边还要处理请求长度不一、到达时间不一、输出长度不一的问题。

## 先分清 prefill 和 decode

LLM 推理可以拆成两个阶段。

Prefill 阶段：一次性处理 prompt，把输入 token 的 hidden states 和 KV cache 算出来。

Decode 阶段：每次生成一个新 token，并复用之前的 KV cache。

这两个阶段的瓶颈不一样。

Prefill 更像大矩阵计算，prompt 越长越贵。长文档问答、代码仓库输入、长对话历史，都会让 prefill 压力变大。

Decode 更像小 batch、逐 token 的循环。每一步只生成一个 token，但要反复读历史 KV cache。输出越长，decode 越贵。

所以优化 LLM inference 时，不能只说“加速推理”。要先问：

- 是首 token 慢，还是后续 token 慢？
- 是 prompt 太长，还是输出太长？
- 是单请求 latency 重要，还是整体 throughput 重要？

## KV cache 是推理的核心状态

自回归生成时，第 $t$ 步要用前面所有 token 的 key 和 value。

如果每一步都重新计算前面所有 token，成本会爆炸。

KV cache 的做法是把历史 token 的 key/value 存下来。

这样第 $t$ 步只需要算新 token 的 query，然后和缓存里的 key/value 做 attention。

粗略看，KV cache 大小和这些因素成正比：

$$
\text{layers} \times \text{sequence length} \times \text{KV heads} \times \text{head dimension}
$$

这也是为什么 GQA、MQA、MLA 这些结构重要。它们不只是模型结构变化，也是推理显存优化。

从系统角度看，KV cache 有两个麻烦点。

第一，它很大。长上下文、多 batch、多层模型都会放大这个问题。

第二，它是动态的。每个请求长度不同，生成长度也不同，cache 会不断增长和释放。

所以 LLM serving 的很多创新，本质上都是在管理 KV cache。

## Continuous batching：请求不是凑齐一批再跑

传统 batching 很简单：凑一批请求，一起 forward。

但 LLM 生成长度不一样。如果一个请求只生成 20 token，另一个请求要生成 2000 token，把它们绑死在同一个 batch 里会很浪费。

Continuous batching 的思路是动态调度。

每个 decode step 都可以把仍在生成的请求组成一个新 batch，同时把新来的请求插进来，把结束的请求移出去。

它解决的是 serving 场景里的利用率问题：

```text
请求持续到达，GPU 不应该等某一批全部生成完再接下一批。
```

这也是为什么推理引擎比裸 PyTorch generate 复杂很多。真正的瓶颈不只是单次 forward 快不快，而是调度器能不能持续把 GPU 填满。

## vLLM 和 PagedAttention：把 KV cache 当内存系统管理

vLLM 最核心的想法是 PagedAttention。

问题背景是：KV cache 很大，而且每个请求长度动态变化。如果给每个请求预留连续大块显存，会产生大量浪费和碎片。

PagedAttention 借鉴操作系统里的分页思想，把 KV cache 拆成固定大小的 blocks。逻辑上一个请求的 cache 是连续增长的，但物理显存里可以分散存放。

直觉上：

```text
不要给每个请求提前分一整块大显存；
按需分小块，需要多少给多少。
```

这样能减少碎片，提高 batch size，也方便 prefix sharing、parallel sampling、beam search 这类共享前缀的场景。

vLLM 不只是一个 attention kernel。它更像一个 LLM serving engine，围绕 KV cache、batching、OpenAI-compatible API、量化、分布式推理这些问题做工程整合。

所以我会把 vLLM 理解成：

> 用系统工程方法解决 LLM serving 里的动态内存和动态调度问题。

## FlashAttention 解决的是 attention kernel 的 IO 问题

FlashAttention 经常和 vLLM 一起被提到，但它解决的问题不同。

标准 attention 可能会把 $QK^\top$ 这样的中间矩阵写入 HBM，再读回来做 softmax 和乘 $V$。

FlashAttention 用 tiling 和 online softmax，把计算分块放进更快的 SRAM，减少 HBM 读写。

所以它不是把 attention 从 $O(n^2)$ 变成 $O(n)$，而是让 exact attention 在 GPU 上跑得更高效。

简单说：

- FlashAttention：优化一次 attention 怎么算
- vLLM/PagedAttention：优化很多请求的 KV cache 怎么存、怎么调度

这两个层次不一样，但在实际 inference system 里会一起出现。

## DeepSpeed-Inference：模型并行、kernel injection 和量化

DeepSpeed 最早更常被当成训练优化框架，比如 ZeRO。

但 DeepSpeed-Inference 关注的是 transformer 推理。

它的核心能力可以粗略拆成三类。

第一，model parallelism。

如果模型太大，单张 GPU 放不下，就需要把模型切到多张 GPU 上。DeepSpeed 可以做 model parallel / tensor parallel，让大模型能跑起来。

第二，kernel injection。

DeepSpeed 会把兼容的 Transformer 层替换成 inference-optimized kernels。也就是不改模型语义，但换更适合推理的底层实现。

第三，量化。

降低权重或 activation 的精度，可以减少显存和带宽压力。量化不是白送的，要看精度损失、硬件支持、kernel 支持。

所以 DeepSpeed-Inference 更像是：

> 给已有 PyTorch / HuggingFace / Megatron 模型加推理并行和高性能 kernel。

而 vLLM 更像是：

> 面向在线 serving 的请求调度和 KV cache 管理系统。

实际选型时要看场景是离线批量推理、在线 API 服务，还是多机大模型部署。

## 分布式推理的几种并行

大模型推理里常见几种并行方式。

Tensor parallelism：把单层里的大矩阵切到多张 GPU 上。比如一个 linear layer 的权重按列或行切分。好处是单层计算变小；代价是每层都要通信。

Pipeline parallelism：把不同层切到不同 GPU 上。前几层在 GPU 0，后几层在 GPU 1。好处是显存压力下降；代价是 pipeline bubble 和调度复杂。

Data parallel / replica serving：多份模型副本处理不同请求。好处是简单、吞吐线性扩展；代价是每份副本都要完整放一份模型。

Expert parallelism：MoE 模型里不同 experts 放到不同设备上。难点是 token routing 会带来不均衡和通信。

Context parallelism：超长上下文下，把 sequence 维度切开并行处理。它更多出现在长上下文和超大模型服务里。

没有一种并行是万能的。

如果模型单卡放得下，replica serving 最简单。

如果模型单卡放不下，先考虑 tensor parallel。

如果模型层数很多、显存压力大，pipeline parallel 也会进来。

如果是 MoE，expert parallel 基本绕不开。

## 量化：省显存，也省带宽

推理时很多瓶颈不是算力，而是显存带宽。

权重越大，从显存读权重越贵。KV cache 越大，每步 decode 读 cache 越贵。

量化的目标是用更低 bit 表示权重或 cache。

常见方向包括：

- weight-only quantization：只量化权重，比如 INT8 / INT4
- activation quantization：activation 也量化，要求 kernel 和校准更严格
- KV cache quantization：降低长上下文 decode 的 cache 显存压力
- FP8：在新硬件上用更低精度获得更高吞吐

量化最容易误解的点是：模型文件变小不等于端到端一定变快。

如果 kernel 不好、反量化开销大、batch 太小、瓶颈不在带宽，速度可能不明显。

所以量化要看三个东西：精度掉多少、显存省多少、实际吞吐提升多少。

## Speculative decoding：用小模型帮大模型起草

Speculative decoding 的直觉是：让一个便宜的 draft model 先生成几个候选 token，再让大模型一次性验证。

如果 draft token 被大模型接受，就相当于大模型一次 forward 推进了多个 token。

它适合 decode 阶段，因为 decode 是逐 token 的，容易被循环开销和内存读写限制。

但 speculative decoding 不是永远有效。

如果 draft model 太弱，接受率低，白算很多。

如果 draft model 太强，成本又高。

如果任务需要复杂推理，draft 和 target 分布差异大，也可能收益下降。

所以它的关键不是“有没有 draft model”，而是接受率和额外成本之间是否划算。

## Long context 只是推理效率问题的一部分

长上下文当然重要，但它不是唯一主线。

长 prompt 会放大 prefill 成本。

长生成会放大 decode 成本。

长对话会放大 KV cache 显存。

所以长上下文应该放到 inference system 里理解，而不是单独只看位置编码。

RoPE scaling、YaRN 这些方法解决的是模型结构和位置外推问题。

FlashAttention、GQA、MLA、PagedAttention、KV cache quantization 解决的是算得动、存得下、调得起来的问题。

Needle-in-a-haystack 这类评测解决的是模型能不能用好长上下文的问题。

这三件事相关，但不是同一件事。

## 读推理优化论文或文档时看什么

我会看这些问题：

- 优化的是 prefill、decode，还是 serving 调度？
- 指标是 latency、throughput、TTFT，还是 TPOT？
- batch size 和 request length 分布是什么？
- KV cache 怎么分配、释放、共享？
- 是否支持 continuous batching？
- 是否改变 attention 语义，还是 exact attention？
- 是否依赖特定硬件或特定 kernel？
- 多卡并行时通信开销怎么算？
- 量化后精度下降多少？
- benchmark 是离线 batch，还是在线 serving workload？

很多推理优化看起来都在“加速 LLM”，但实际目标差很多。一个方法提升 long prompt prefill，不一定提升短 prompt 高并发服务。一个方法提升吞吐，不一定降低单请求延迟。

## 几个点

- LLM inference 要先分 prefill 和 decode，两者瓶颈不同。
- KV cache 是推理系统的核心状态，长上下文和高并发都会把它放大。
- Continuous batching 解决的是在线服务里请求动态进出的调度问题。
- vLLM / PagedAttention 重点在 KV cache 内存管理和 serving 调度。
- FlashAttention 重点在 attention kernel 的 IO 效率。
- DeepSpeed-Inference 更偏模型并行、kernel injection 和量化。
- 分布式推理要看模型是否放得下、通信是否可控、吞吐和延迟哪个更重要。

## 参考资料

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- vLLM Docs, [Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/)
- vLLM Docs, [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html)
- DeepSpeed, [Getting Started with DeepSpeed for Inferencing Transformer based Models](https://www.deepspeed.ai/tutorials/inference-tutorial/)
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
