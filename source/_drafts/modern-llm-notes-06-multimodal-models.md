---
title: "现代大模型笔记 06：多模态模型，图像、视频和音频怎么接进 LLM"
date: 2025-10-11 20:30:00
updated: 2025-10-11 20:30:00
categories:
  - 现代大模型与前沿论文笔记
tags:
  - 大模型
  - 多模态
  - VLM
  - Qwen-VL
  - Omni
  - 笔记
math: true
category_bar: true
---

这一篇整理多模态模型。

多模态模型经常被描述成“模型能看图、看视频、听音频”。但从结构上看，问题其实是：

> 怎么把不同模态的信息变成 LLM 能处理的 representation。

LLM 本来处理的是 token sequence。图像、视频、音频不是离散文本 token，所以要先经过编码、投影、对齐，再接入语言模型。

## 1. VLM 的基本结构

一个典型 vision-language model 可以拆成三块：

- Vision encoder
- Projector / adapter
- LLM

Vision encoder 把图片编码成视觉特征。

Projector 把视觉特征映射到 LLM 的 hidden dimension。

LLM 把视觉 token 和文本 token 放在一起处理，并生成回答。

粗略写成：

$$
z_v = \mathrm{VisionEncoder}(I)
$$

$$
h_v = \mathrm{Projector}(z_v)
$$

然后把 $h_v$ 和文本 embedding 拼接，送入 LLM。

这个结构很直接，但每一步都有细节。

## 2. 图像 token 是怎么来的

图像通常会被切成 patches。

ViT 的做法是把图片分成固定大小 patch，然后每个 patch 映射成一个向量。

如果一张图片被切成 $N$ 个 patches，就会得到 $N$ 个视觉 token。

问题是：图片分辨率越高，patch 数越多，视觉 token 越多。

视觉 token 多了以后，LLM 的上下文压力变大。

所以多模态模型经常要做 token compression 或 dynamic resolution。

这里的核心 trade-off 是：

- token 多，视觉细节保留更好，但计算更贵
- token 少，推理更便宜，但细节可能丢失

比如 OCR、图表理解、数学几何题，都需要细粒度视觉信息。普通场景描述可能不需要那么多 token。

## 3. Projector 不只是维度对齐

很多入门解释会说 projector 只是把 vision encoder 输出维度变成 LLM hidden size。

这不算错，但太简单。

Projector 还承担模态对齐作用。

Vision encoder 的表示空间来自图像预训练，LLM 的表示空间来自文本预训练。两者不是天然兼容。

Projector 要把视觉特征变成 LLM 能理解的“类 token 表示”。

如果 projector 太弱，视觉信息进了 LLM 也不好用。

如果 projector 太强或训练不稳定，又可能破坏语言模型原有能力。

所以多模态训练经常分阶段：

1. 先训练 projector，让视觉特征对齐文本空间
2. 再做 instruction tuning，让模型学会多模态问答
3. 后面再加入更复杂的 reasoning、OCR、video、agent 数据

## 4. 拼接视觉 token vs cross-attention

视觉信息接入 LLM 有不同方式。

一种是直接把视觉 token 拼到文本 token 前面或中间，让 LLM self-attention 统一处理。

这种方式简单，符合 decoder-only LLM 的接口。

另一种是 cross-attention。语言 token 在某些层通过 cross-attention 去读取视觉特征。

直接拼接的好处是结构统一。

Cross-attention 的好处是视觉特征可以作为单独 memory，控制更灵活。

但两者都绕不开一个问题：视觉 token 太多时，attention 成本会很高。

所以现代 VLM 的关键不是“能不能接图像”，而是“怎么高效接入足够多的视觉信息”。

## 5. 多图输入更难在哪里

单图问答已经不简单，多图更难。

多图任务需要模型区分不同图片，并在图片之间建立关系。

比如：

> 图 1 是模型结构图，图 2 是实验结果表。请判断结构改动是否带来了稳定提升。

这时模型不只是描述图像，而是要跨图对齐信息。

多图输入涉及：

- 每张图的边界
- 图片顺序
- 跨图引用
- 视觉信息和文本问题的对应关系

如果视觉 token 全部拼在一起，模型必须学会哪些 token 属于哪张图。

这就是为什么多模态位置编码和 image boundary token 很重要。

## 6. 视频：多了时间维度

视频不是图片列表这么简单。

视频有时间顺序、运动、事件持续、因果关系。

如果只抽几帧，可能丢掉关键动作。

如果抽很多帧，token 数会爆炸。

所以视频模型通常要处理三个问题：

- frame sampling：取哪些帧
- temporal encoding：怎么表示时间顺序
- long video understanding：怎么保留长时间信息

举个例子。

如果视频里一个人先拿起杯子，再把杯子放进柜子，最后关上柜门。只看最后一帧，模型可能知道柜门关了，但不知道杯子被放进去了。

视频理解需要事件链，而不是静态描述。

## 7. 音频：连续信号怎么变 token

音频比图像更不一样。

文本是离散 token，音频是连续波形。

音频模型通常会先把波形变成特征，比如 spectrogram，或者用 audio encoder 得到表示。

如果要生成语音，还需要把文本或 hidden states 转成 speech tokens 或 codec tokens，再解码成波形。

Qwen3-Omni 这类模型会涉及 Thinker-Talker 结构。可以粗略理解为：

- Thinker 负责多模态理解和推理
- Talker 负责生成自然语音

这样做的原因是理解和语音生成不是同一个问题。理解需要语义、推理、对齐；语音生成还需要音色、节奏、流式延迟。

## 8. Omni model 难在哪里

Omni model 希望一个模型同时处理文本、图像、视频、音频，并且可能还要输出文本和语音。

这比 VLM 更难，因为不同模态的时间尺度和信息密度不同。

一张图片可以被压成几百个视觉 token。

一段音频可能每秒产生大量 acoustic tokens。

视频又同时包含空间和时间信息。

把这些东西统一到一个模型里，需要解决：

- 模态表示统一
- 长上下文压力
- 训练数据混合
- 不同任务 loss 的平衡
- 实时交互延迟
- 输出模态控制

所以 omni model 不是简单把 VLM、ASR、TTS 拼起来。真正难的是统一训练和统一推理。

## 9. 多模态 reasoning

多模态模型最有价值的地方不是“描述图片”，而是跨模态 reasoning。

比如：

- 看图表并推断趋势
- 看论文公式截图并解释推导
- 看 UI 截图并判断用户下一步该点哪里
- 看视频并找出事件因果
- 听语音并结合语气理解意图

这些任务要求模型把视觉或音频证据和语言推理结合。

所以多模态 benchmark 也在从 captioning 走向 MMMU、MathVista、video QA、document understanding 等更复杂任务。

## 10. 幻觉问题更复杂

文本模型会 hallucinate，多模态模型也会，而且更难检查。

比如模型可能说图片里有某个物体，但实际没有。

或者 OCR 把数字看错，后面推理全错。

或者视频里漏掉关键动作，却生成一个合理故事。

多模态 hallucination 的根源可能来自：

- 视觉 encoder 没捕捉到细节
- projector 对齐不好
- LLM 依赖语言先验脑补
- 图像分辨率不够
- 训练数据里 caption 偏泛化

所以多模态模型的回答要特别注意 evidence grounding。最好能让模型指出依据来自图像哪个区域、哪一帧、哪段音频。

## 11. 训练数据决定上限

多模态模型很依赖数据。

图片-caption 数据可以教模型描述图片。

OCR 数据可以教模型读文字。

图表问答数据可以教模型理解可视化。

视频问答数据可以教模型时间推理。

音频对话数据可以教模型语音交互。

如果数据只覆盖普通图片描述，模型很难自动变成强视觉推理模型。

这和语言模型一样：结构提供可能性，数据决定能力落点。

## 12. 读多模态技术报告时看什么

我会看这些问题：

- Vision encoder 是什么
- 视觉 token 数怎么控制
- Projector 是线性层、MLP，还是更复杂结构
- 是否冻结 LLM 或 vision encoder
- 训练分几个阶段
- 支持单图、多图、视频、音频中的哪些输入
- 是否支持语音输出
- benchmark 覆盖 caption、OCR、chart、math、video reasoning 吗
- 长视频或高分辨率图片怎么处理
- 有没有 grounding 或 hallucination 评估

尤其要看“输入支持”和“能力支持”的区别。能输入视频，不等于能理解长视频事件。能输入音频，不等于能做自然低延迟语音对话。

## 13. 小结

多模态模型的核心问题是模态对齐。

图像、视频、音频都要先变成 LLM 能处理的表示，然后再和文本 token 一起完成理解和生成。

VLM 的关键是视觉 token、projector 和视觉语言对齐。

视频模型多了时间维度。

音频模型多了连续信号、codec、实时生成的问题。

Omni model 则进一步要求一个系统统一处理多个模态的输入和输出。

所以我会把多模态看成现代大模型从“语言接口”走向“世界接口”的一步。但这一步不是靠一个接口参数实现的，而是结构、数据、训练阶段和系统延迟一起决定的。

## 参考资料

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Liu et al., [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- Bai et al., [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- Xu et al., [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
- OpenAI, [Video generation models as world simulators](https://openai.com/research/video-generation-models-as-world-simulators)
- GLM-4.5 Team, [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471)
