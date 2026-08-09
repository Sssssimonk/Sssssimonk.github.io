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
  - SGLang
  - TensorRT-LLM
  - FlashAttention
  - 笔记
math: true
category_bar: true
---

> 有一个挺有意思的说法：算法就是数据，数据就是 Infra，Infra就是算法。

模型能力的提升离不开数据质量，而大规模数据处理、训练和部署都离不开 Infrastructure。

## 先划清推理加速的边界

推理加速主要优化三个目标：

- 降低延迟（latency）
- 提升吞吐量（throughput）
- 降低显存或内存占用（memory footprint）

这三个目标不一定同时改善。

比如增大 batch size 通常能提高吞吐量，但可能增加排队时间；把模型切到更多 GPU 可以解决显存不足，但跨卡通信可能让单请求延迟变高；量化能降低显存和带宽压力，但如果没有对应的低精度 kernel，模型文件虽然变小，实际速度未必提升。


常见指标包括：

- TTFT（time to first token）：从请求到达，到第一个输出 token 返回
- TPOT（time per output token）：生成阶段每个 token 的平均耗时
- E2E latency：整个请求从进入到生成完成的耗时
- Throughput：每秒处理的请求数或 token 数
- Concurrency：系统同时承载的请求数

**模型推理中的KV Cache**

LLM 生成可以拆成两个阶段。

Prefill 阶段一次性处理 prompt，并计算每层的 KV cache。

Decode 阶段每次生成一个新 token，然后把这个 token 的 key/value 继续追加到 cache 中。

Prefill 通常是大矩阵计算，算术强度较高，更容易受计算能力限制。prompt 越长，prefill 的计算压力越大，直接影响 TTFT。

Decode 每一步只处理少量新 token，却要读取大量模型权重和历史 KV cache，通常更偏访存或显存带宽瓶颈。输出越长，TPOT 和总生成时间越重要。

这一区分贯穿整篇：

- FlashAttention、chunked prefill 更直接影响 prefill
- KV cache、continuous batching、speculative decoding 更直接影响 decode 或在线 serving

## 推理优化的五条主线

可以把主要方法分成五层：

| 层次 | 主要问题 | 代表方法 |
| --- | --- | --- |
| 模型层 | 模型本身太大、计算太多 | 量化、剪枝、蒸馏、MoE、低秩近似 |
| 生成算法层 | 自回归逐 token 生成效率低 | KV cache、投机解码、上下文压缩 |
| 算子与编译层 | 单次 forward 的 kernel 和访存效率低 | FlashAttention、算子融合、编译器 |
| 服务调度层 | 请求长短不一，GPU 利用率低 | continuous batching、chunked prefill |
| 分布式层 | 单卡放不下或吞吐不够 | TP、PP、DP、EP、context parallel |



## 模型结构压缩

模型层的目标是减少需要存储、搬运或计算的参数。

### 量化（quantization）

量化使用更低精度表示权重、activation 或 KV cache。

![FP32、TF32、FP16 与 BF16 的位宽结构对比](/img/llm-inference/floating-point-formats-nvidia.png)

*图源：[NVIDIA Technical Blog](https://developer.nvidia.com/blog/accelerating-ai-inference-workloads-with-nvidia-a30-gpu/)*

常见记法是 `WxAy`：

- W8A16：8-bit 权重，16-bit activation
- W4A16：4-bit 权重，16-bit activation
- W8A8：8-bit 权重和 8-bit activation
- FP8：权重和 activation 使用 FP8，具体格式与硬件、框架有关

GPTQ、AWQ等主要减少权重显存和权重读取带宽，属于权重量化这一方向。权重与 activation 同时量化可以进一步降低计算与访存成本，但 activation 分布会随输入变化，通常比只量化权重更难。

动态量化在运行时计算量化参数，对输入变化更灵活，但会增加额外开销。

静态量化提前使用 calibration data 确定 scale，运行时更简单，但校准数据需要覆盖真实输入分布。

混合精度量化则让敏感层保留 FP16/BF16，只压低更稳健的层。Embedding、输出头、少数 outlier channels 经常需要特殊处理。

量化要同时看三件事：

- 精度下降多少
- 显存减少多少
- 在目标硬件和 batch size 下，端到端吞吐到底提升多少


## 推理算法优化

针对 LLM 的自回归生成过程，核心是减少重复计算、减少每步读取的数据，或者让一次 target-model forward 推进多个 token。

### KV cache：生成阶段的核心状态

自回归生成到第 $t$ 步时，attention 需要历史 token 的 key 和 value。

如果每一步都重新计算完整前缀，重复工作会迅速增长。KV cache 把历史 key/value 保存下来，每一步只计算新 token 对应的部分。

KV cache 的显存可以粗略写成：

$$
M_{\mathrm{KV}}
\approx
2BLSh_{\mathrm{kv}}d_hb
$$

其中：

- $2$ 表示 key 和 value
- $B$ 是Batch Size
- $L$ 是层数
- $S$ 是已缓存序列长度
- $h_{\mathrm{kv}}$ 是 KV heads 数量
- $d_h$ 是 head dimension
- $b$ 是每个元素占用的字节数

所以并发、上下文长度和模型层数都会线性放大 KV cache。

KV cache 优化可以继续分成几类。

- PagedAttention 把每个请求的 KV cache 拆成固定大小的 blocks。逻辑序列可以连续增长，物理显存不需要连续分配，从而减少预留浪费和内存碎片。

- Prefix caching 缓存已经计算好的公共前缀。多个请求共享 system prompt、长文档或多轮对话历史时，可以直接复用对应 KV blocks，跳过重复 prefill。

- KV cache quantization 使用 INT8、FP8 或更低精度存储 cache，减少显存和 decode 阶段的读取量。代价是额外的量化/反量化操作以及可能的精度损失。

- KV cache eviction 在显存不足时淘汰不重要或较旧的 cache。Sliding-window KV 只保留最近窗口，适合局部注意力模型，但会丢失超出窗口的精确信息。

- KV cache offloading 把部分 cache 放到 CPU 内存、远端内存，极端情况下也可能进入持久化存储。它能扩大容量，但 PCIe、网络或磁盘带宽会成为新瓶颈。

### 投机解码与MTP

投机解码让便宜的 draft model 先生成一组候选 token，再由 target model 并行验证。

如果多个候选被接受，target model 一次 forward 就能推进多个 token，同时保持目标分布正确。

收益取决于：

- draft model 的成本
- 候选 token 的接受率
- 一次提出多少 token
- target model 并行验证的效率

Draft 太弱，接受率低；draft 太强，起草成本又高。对于分布稳定、输出可预测的任务，投机解码通常更容易获益。

MTP 让模型在训练或推理时预测多个未来 token。它可以为 speculative decoding 提供 draft tokens，也可以减少严格逐 token 串行带来的限制。

不过“模型有 MTP head”不等于可以无条件一次输出多个 token。候选仍然需要验证或满足特定解码假设，否则会改变生成分布。

当前一些方法包括DFlash， DSPark。
并行解码还包括 tree-based verification、Medusa/EAGLE 类 draft heads、n-gram speculation 等方案。它们的共同目标都是提高每次 target forward 能确认的 token 数量。

### 上下文与注意力优化

长上下文会同时增加 prefill 计算和 KV cache 占用。

常见方向包括：

- sliding-window attention：只看最近窗口
- sparse/block attention：只计算部分 token pairs
- context compression：先摘要、检索或压缩 token
- KV compression：合并、量化或选择性保留 cache
- recurrent memory：把长历史压到持续更新的 memory state

这些方法通过放弃部分全局交互换效率。关键不是窗口能开多大，而是被丢弃的信息是否影响当前任务。

## 算子与编译优化（kernel and compiler）

算子的角度不改变模型输出目标，重点是让同样的计算在硬件上跑得更有效率。

### FlashAttention 与 FlashInfer

标准 attention 容易产生巨大的中间矩阵，并在 HBM 与片上 SRAM 之间反复搬运数据。

FlashAttention 使用 tiling 和 online softmax，把 attention 分块计算，减少 HBM IO。

它仍然是 exact attention，并没有把理论上的两两交互改成线性复杂度。它优化的是实际访存和中间结果存储。

FlashAttention-2 进一步改善 work partitioning 和 GPU 利用率。

FlashInfer 更偏 LLM serving 场景，提供 attention、sampling、MoE 等推理 kernel，尤其关注不同 batch、不同 KV layout 和 decode workload。

所以可以这样区分：

- FlashAttention：一种 IO-aware attention 算法与 kernel 实现
- FlashInfer：面向推理服务的 kernel library

### Fused kernel

GPU kernel launch 和 HBM 读写都有成本。

如果每个小操作都单独执行：

```text
read -> RMSNorm -> write
read -> projection -> write
read -> RoPE -> write
```

中间结果会反复写回显存。

Fused kernel 尽量把多个操作合并，例如：

- RMSNorm + quantization
- bias + activation
- QKV projection + RoPE + KV cache write
- dequantization + matmul

算子融合减少 kernel launch 和中间访存，但融合过度也可能降低复用性，让动态 shape 更难处理。

## 批处理与请求调度（服务侧的优化）

模型在实际服务中，还要处理请求到达时间、prompt 长度、输出长度和优先级都不同的问题。

服务侧优化的目标是：减少GPU的空闲时间，提高有效利用率。

### Continuous batching

Static batching 会等一批请求全部结束，再处理下一批。

![Static batching 与 continuous batching 的调度方式对比](/img/llm-inference/continuous-batching-redhat.png)

*图源：[Red Hat Developer](https://developers.redhat.com/articles/2026/06/15/llamacpp-vs-vllm-choosing-right-local-llm-inference-engine)*

但 LLM 请求的输出长度差异很大。一个请求生成 20 token，另一个生成 2000 token，把它们固定在同一 batch 会产生大量等待。

Continuous batching会在每个 scheduling step：

- 加入新请求
- 保留仍在 decode 的请求
- 移除已经结束的请求
- 重新组成下一轮 batch

这样短请求结束后，空出的资源可以立刻交给新请求。

调度器可以抽象成下面这个循环。它的关键不是提前固定 batch，而是在每个 step 后重新决定下一批要处理哪些 token：

```text
waiting_queue = incoming requests
active_requests = {}

while waiting_queue is not empty or active_requests is not empty:
    release KV blocks owned by finished requests
    remove finished requests from active_requests

    while scheduler still has token/KV budget:
        request = waiting_queue.pop_next()
        allocate KV blocks for request
        active_requests.add(request)

    batch = scheduler.select_next_step(active_requests)
    # batch 中可以同时包含 chunked prefill 和 one-token decode

    model_output = model.forward(batch.tokens, batch.kv_block_tables)

    for request in batch:
        append new K/V to request.kv_blocks

        if request is decoding:
            token = sample(model_output[request])
            stream token to client
            request.next_input = token

        if request reaches EOS, length limit, or cancellation:
            request.mark_finished()
```

真实 serving engine 还会考虑优先级、抢占、prefix locality 和 prefill/decode 配额，但基本状态变化就是：加入请求、推进一步、释放完成请求、重新组 batch。


## 分布式推理（distributed inference）

分布式推理有两个不同目标：

- 单个模型实例太大，需要切到多张卡
- 单实例能运行，但吞吐不够，需要复制更多实例

这两类问题对应不同并行方式。



### 张量并行（tensor parallelism）

TP 把同一层的大矩阵切到多张 GPU 上。

它能解决单卡放不下模型的问题，并让多个 GPU 同时计算同一层。

代价是几乎每个 Transformer block 都需要 collective communication。卡间带宽不足时，增加 GPU 反而可能降低效率。

TP 更适合节点内通过 NVLink/NVSwitch 连接的 GPU。

### 流水线并行（pipeline parallelism）

PP 按层切分模型：

```text
GPU 0: layers 0-19
GPU 1: layers 20-39
GPU 2: layers 40-59
```

PP 减少每张卡需要存储的层数，但会产生 pipeline bubble。自回归 decode 的 microbatch 较小时，bubble 和 stage imbalance 更明显。

### 数据并行与 replica serving

如果一份模型能放进一组 GPU，可以启动多个 replicas，让不同副本处理不同请求。

Replica serving 不减少单请求延迟，但通常是横向提高吞吐最直接的方式。

路由器需要根据 queue length、KV cache 占用、prefix locality 和模型版本把请求分配到合适实例。

### 序列并行与上下文并行

这两个术语在不同框架里定义不完全一致。

Sequence parallelism 通常沿 sequence 维度拆分 activation 或部分算子，用来减少单卡 activation 占用。

Context parallelism 更明确地把长上下文 token 分到多张卡上，让各卡只保存部分序列状态，并通过通信完成全局 attention。

它们适合超长 context，但通信模式复杂，不能把“序列长度除以卡数”直接当成线性加速。

### 专家并行（expert parallelism）

EP 把 MoE experts 分散到不同 GPU。

Token 经过 router 后，需要发送到对应 expert 所在设备，通常涉及 all-to-all communication。

EP 的主要问题包括：

- expert load imbalance
- hot experts
- token dispatch 通信
- 跨节点带宽

所以 MoE 推理优化经常同时涉及 TP、EP、路由负载均衡和通信 kernel。

### 分布式 KV cache

跨卡 KV cache 可以出现在几种场景：

- prefill 实例把 KV 传给 decode 实例
- 多个 replicas 共享外部 KV cache
- 相同前缀请求复用其他节点已经计算的 blocks
- GPU 显存不足时把 KV 放到 CPU 或远端内存

它能减少重复 prefill 或扩大 cache 容量，但会把问题转化为网络传输、cache consistency、block layout 和 locality 管理。

小借：

- Dense model：TP + PP，外加 replicas 扩吞吐
- MoE model：TP + EP，必要时再加 PP
- Long-context model：TP + context parallel



## 业务层策略
类似传统后端的一些处理方法了

### 请求与结果缓存

完全相同的确定性请求可以直接返回历史结果。

共享长前缀的请求可以复用 prefix KV cache。

但 temperature、sampling seed、权限信息、用户上下文和模型版本都会影响 cache key，不能只拿 query 文本做缓存。

### 路由和负载均衡

请求可以根据任务难度、上下文长度、模型版本路由到不同模型：

- 简单请求走小模型
- 复杂请求走大模型
- 长上下文走专用实例
- 共享前缀的请求尽量去已有 cache 的实例

好的路由策略有时比单个 kernel 优化带来的收益更大。

## 参考资料

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- vLLM Docs, [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- vLLM Docs, [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
- SGLang Docs, [SGLang Documentation](https://docs.sglang.io/)
- NVIDIA, [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)
- NVIDIA, [Paged Attention, In-Flight Batching and Request Scheduling](https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html)
- DeepSpeed, [DeepSpeed-Inference Tutorial](https://www.deepspeed.ai/tutorials/inference-tutorial/)
