# AI 算法岗校招知识思维导图

> 与[面试题速查](./README.md)配套使用。每张图按“大类 → 核心知识点 → 子知识点”展开，用于开始复习前建立全局框架，以及面试前快速查漏补缺。

## 目录

1. [数学、概率与统计](#1-数学概率与统计)
2. [机器学习](#2-机器学习)
3. [深度学习](#3-深度学习)
4. [CV、NLP 与推荐系统](#4-cvnlp-与推荐系统)
5. [强化学习](#5-强化学习)
6. [Transformer 与 LLM](#6-transformer-与-llm)
7. [大模型微调与对齐](#7-大模型微调与对齐)
8. [RAG](#8-rag)
9. [Agent](#9-agent)
10. [训练、推理与 MLOps](#10-训练推理与-mlops)
11. [Python、SQL 与计算机基础](#11-pythonsql-与计算机基础)
12. [手撕代码与数学推导](#12-手撕代码与数学推导)
13. [项目追问、系统设计与行为面](#13-项目追问系统设计与行为面)

## 1. 数学、概率与统计

```mermaid
mindmap
  root((数学、概率与统计))
    概率基础
      条件概率与贝叶斯公式
      独立与不相关
      常见概率分布
      期望、方差与协方差
    统计推断
      最大似然估计 MLE
      最大后验估计 MAP
      假设检验与 p 值
      置信区间
      大数定律与中心极限定理
    线性代数
      向量与矩阵运算
      特征值与特征向量
      正定与半正定
      SVD 与低秩近似
    优化基础
      梯度与方向导数
      凸函数与 Jensen 不等式
      L1 与 L2 范数
      拉格朗日乘子与对偶
    信息论
      熵与交叉熵
      KL 散度
      互信息
```

## 2. 机器学习

```mermaid
mindmap
  root((机器学习))
    问题定义
      监督、无监督与自监督
      分类、回归与排序
      生成式与判别式
      业务目标与机器学习目标
    数据与特征
      数据切分与交叉验证
      缺失值与异常值
      类别不平衡
      特征缩放与编码
      数据泄漏
    经典模型
      线性回归与逻辑回归
      决策树
      随机森林
      GBDT 与 XGBoost
      SVM 与核方法
      K-means 与 PCA
    泛化能力
      偏差与方差
      欠拟合与过拟合
      L1 与 L2 正则
      学习曲线
    评估与选型
      Precision、Recall 与 F1
      ROC-AUC 与 PR-AUC
      校准与阈值
      超参数搜索
      离线指标与线上指标
    生产问题
      数据漂移
      概念漂移
      可解释性
      公平性
```

## 3. 深度学习

```mermaid
mindmap
  root((深度学习))
    神经网络基础
      线性层与非线性激活
      前向传播
      计算图
      反向传播与链式法则
    优化训练
      SGD 与 Momentum
      Adam 与 AdamW
      学习率和调度器
      权重初始化
      梯度裁剪
    稳定性与泛化
      梯度消失与爆炸
      BatchNorm 与 LayerNorm
      Dropout
      数据增强
      Early Stopping
    网络结构
      CNN 与感受野
      RNN、LSTM 与 GRU
      残差连接
      Attention 与 Transformer
    损失函数
      均方误差
      交叉熵
      对比学习损失
      Focal Loss
    压缩与部署
      知识蒸馏
      剪枝
      量化
```

## 4. CV、NLP 与推荐系统

```mermaid
mindmap
  root((CV、NLP 与推荐系统))
    计算机视觉 CV
      图像分类
        CNN 与数据增强
      目标检测
        单阶段与两阶段
        Anchor、IoU 与 NMS
      图像分割
        语义分割
        实例分割
      视觉表征
        ViT
        多模态对齐
    自然语言处理 NLP
      文本表示
        One-hot 与 TF-IDF
        Word2Vec
        上下文表示
      序列建模
        RNN、LSTM 与 CRF
        Transformer
      预训练模型
        BERT
        GPT
      文本评估
        BLEU
        ROUGE
        语义与事实性评估
    推荐系统
      召回
        协同过滤
        双塔与 ANN
        多路召回
      排序
        粗排与精排
        CTR 与转化率预估
        重排与多样性
      评估
        Recall、NDCG 与 AUC
        在线 A/B 测试
      工程问题
        冷启动
        反馈回路
        实时特征
```

## 5. 强化学习

```mermaid
mindmap
  root((强化学习))
    MDP 基础
      状态、动作与奖励
      转移概率
      马尔可夫性质
      折扣回报
    价值函数
      状态价值 V
      动作价值 Q
      优势函数 A
      Bellman 方程
    价值学习
      Monte Carlo
      Temporal Difference
      Q-learning
      SARSA
      DQN
    策略学习
      Policy Gradient
      REINFORCE
      Actor-Critic
      PPO
    学习范式
      On-policy 与 Off-policy
      Model-free 与 Model-based
      离线强化学习
    关键挑战
      探索与利用
      稀疏奖励
      样本效率
      奖励塑形
      Reward Hacking
```

## 6. Transformer 与 LLM

```mermaid
mindmap
  root((Transformer 与 LLM))
    输入表示
      Tokenization
        BPE
        WordPiece
        Unigram
      Token Embedding
      位置编码
        绝对位置
        相对位置
        RoPE
    Transformer Block
      Self-Attention
        Q、K、V
        Scaled Dot-Product
        Multi-Head Attention
        Causal Mask
      FFN
        GELU
        SwiGLU
      残差连接
      LayerNorm 与 RMSNorm
      Pre-Norm 与 Post-Norm
    模型结构
      Encoder-only
      Decoder-only
      Encoder-Decoder
      Dense 与 MoE
    训练
      下一 Token 预测
      Teacher Forcing
      Scaling Law
      数据质量与去重
    生成
      Greedy 与 Beam Search
      Temperature
      Top-k 与 Top-p
      重复与幻觉
    长上下文与推理
      Prefill 与 Decode
      KV Cache
      MQA 与 GQA
      FlashAttention
      Speculative Decoding
```

## 7. 大模型微调与对齐

```mermaid
mindmap
  root((大模型微调与对齐))
    训练阶段
      预训练
      持续预训练
      指令微调 SFT
      偏好对齐
    微调方式
      Full Fine-tuning
      PEFT
        LoRA
        Adapter
        Prefix Tuning
      QLoRA
        4-bit NF4
        双重量化
        分页优化器
    RLHF
      人类偏好数据
      奖励模型
      PPO
      KL 约束
    直接偏好优化
      DPO
      Chosen 与 Rejected
      参考模型
    推理强化学习
      GRPO
      规则奖励
      组内相对优势
    数据问题
      指令覆盖与多样性
      长度和位置偏差
      灾难性遗忘
      Reward Hacking
    评估
      指令遵循
      人类偏好
      安全与事实性
      通用能力回归
```

## 8. RAG

```mermaid
mindmap
  root((RAG))
    数据接入
      文档解析
      清洗与去重
      Chunk Size
      Chunk Overlap
      元数据与版本
    索引
      Embedding
      向量数据库
      ANN
      稀疏倒排索引
    检索
      Dense Retrieval
      BM25
      Hybrid Search
      Query Rewriting
      HyDE
    排序与上下文
      Bi-encoder
      Cross-encoder Rerank
      去重与多样性
      Context Packing
    生成
      Grounded Prompt
      引用与证据
      不可答检测
      冲突证据处理
    评估
      Recall@K
      MRR 与 NDCG
      Faithfulness
      答案正确率
      端到端任务成功率
    生产治理
      ACL 与租户隔离
      增量更新与删除
      缓存、成本与延迟
      Prompt Injection
```

## 9. Agent

```mermaid
mindmap
  root((Agent))
    基本循环
      目标与指令
      观察
      决策
      行动
      终止条件
    架构模式
      Workflow
      ReAct
      Planner-Executor
      Evaluator-Optimizer
      单 Agent 与多 Agent
    工具系统
      Function Calling
      输入 Schema
      结构化结果
      错误与重试
      幂等性
    MCP
      Host、Client 与 Server
      Tools
      Resources
      Prompts
      Transport
    状态与记忆
      工作记忆
      情节记忆
      语义记忆
      显式任务状态
      压缩与检索
    可靠性
      超时与预算
      死循环检测
      Checkpoint 与恢复
      可观测性与审计
    安全
      Prompt Injection
      最小权限
      沙箱
      Human-in-the-loop
      敏感信息保护
    评估
      任务成功率
      工具选择与参数
      步骤效率
      成本与延迟
      安全攻击集
```

## 10. 训练、推理与 MLOps

```mermaid
mindmap
  root((训练、推理与 MLOps))
    分布式训练
      数据并行 DDP
      张量并行 TP
      流水线并行 PP
      参数与状态分片
      通信与 All-Reduce
    显存与数值
      FP32、FP16 与 BF16
      Automatic Mixed Precision
      Gradient Accumulation
      Gradient Checkpointing
      OOM 定位
    模型压缩
      PTQ 与 QAT
      INT8、INT4 与 FP8
      剪枝
      蒸馏
    LLM 推理
      Prefill
      Decode
      KV Cache
      PagedAttention
      Continuous Batching
      Speculative Decoding
    服务指标
      TTFT
      TPOT
      Latency
      Throughput
      GPU 利用率与成本
    发布与运维
      模型注册和版本
      Shadow、Canary 与 Blue-Green
      监控与告警
      回滚与降级
      A/B 测试
    数据与反馈
      Feature Store
      数据质量
      漂移检测
      反馈闭环与再训练
```

## 11. Python、SQL 与计算机基础

```mermaid
mindmap
  root((Python、SQL 与计算机基础))
    Python 语言
      list 与 tuple
      可变与不可变对象
      浅拷贝与深拷贝
      可变默认参数
      装饰器
      迭代器与生成器
      GIL
    操作系统
      进程与线程
      协程与事件循环
      死锁
      虚拟内存
      I/O 多路复用
    计算机网络
      TCP 与 UDP
      TCP 三次握手
      HTTP 与 HTTPS
      TLS
      超时、重试与幂等
    数据库与 SQL
      B+ 树与索引
      事务与隔离级别
      JOIN
      GROUP BY 与 HAVING
      窗口函数
    数据结构
      数组与链表
      哈希表
      栈与队列
      堆
      树与图
    算法复杂度
      时间复杂度
      空间复杂度
      均摊分析
```

## 12. 手撕代码与数学推导

```mermaid
mindmap
  root((手撕代码与数学推导))
    数据结构与算法
      二分查找
      链表反转
      Top-K 与堆
      合并区间
      LRU Cache
      编辑距离
      Reservoir Sampling
    机器学习代码
      线性回归
      逻辑回归
      K-means
      指标计算
    深度学习代码
      稳定 Softmax
      交叉熵
      二维卷积
      Scaled Dot-Product Attention
    数学推导
      线性回归正规方程
      Sigmoid 与 BCE 梯度
      Softmax 与 CE 梯度
      Backpropagation
      SVM 对偶
    实现检查
      输入输出与形状
      数值稳定性
      边界条件
      时间与空间复杂度
      测试用例
```

## 13. 项目追问、系统设计与行为面

```mermaid
mindmap
  root((项目追问、系统设计与行为面))
    项目表达
      问题与业务价值
      个人职责
      技术方案与取舍
      量化结果
      失败与复盘
    项目证据
      数据来源与切分
      Baseline
      评估指标
      消融实验
      统计显著性
      线上验证
    系统设计框架
      需求、规模与约束
      离线、在线与护栏指标
      数据和标签
      模型训练与评估
      在线服务
      发布、监控与反馈
    高频设计题
      分类与风控
      推荐系统
      企业 RAG
      LLM 推理服务
      工具型 Agent
    行为面
      岗位动机
      冲突处理
      失败经历
      快速学习
      团队协作
      反问面试官
    回答方法
      先结论后证据
      STAR
      明确个人贡献
      说明边界与取舍
      给出验证方案
```
