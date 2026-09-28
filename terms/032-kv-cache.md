# 032 · KV Cache

- ① 英文术语：KV Cache（Key-Value Cache，全称「键值缓存」）
- ② 中文翻译：KV 缓存（键值缓存）
- ③ 一句话定义：大模型在「一个字一个字往外吐」的生成过程中，把已经算过的每个 token 的 Key 和 Value 向量存起来复用，避免每次都从头重算，用「多吃一点显存」换「大幅提速」的一种推理优化技术。

---

## ④ 核心知识

### 先理解背景：为什么生成会慢

大模型生成文字，是**自回归**（Autoregressive）的：一个字一个字地吐。你先给它「今天天气」，它算出下一个字「真」，然后把「今天天气真」再喂回去，算出「好」，再把「今天天气真好」喂回去……如此循环，直到吐出一个结束符。

这里有个关键矛盾：模型要预测第 N 个字时，必须「看到」前面 N-1 个字。如果每预测一个字，都要把**整个句子从头到尾重新算一遍**，那生成一个 1000 字的回答，就等于算了 1 + 2 + 3 + … + 1000 ≈ 50 万次「字级别」的运算，其中 99% 都是在重复计算前面已经算过的东西。这显然是巨大的浪费。

KV Cache 就是专门消灭这种重复计算的。它之所以叫「KV 缓存」，跟 Transformer 里注意力机制的内部结构直接相关，我们得先把注意力拆开看。

### 原理：注意力机制里的 Q、K、V 三兄弟

Transformer 的核心是**自注意力（Self-Attention）**。简单说，它让每个 token 去「看」句子里其他所有 token，判断「谁跟我关系最密切」，然后按关系远近加权汇总信息。

具体到数学上，每个 token 的向量会被三组矩阵分别投影成三个东西：

- **Q（Query，查询）**：代表「我在找什么」。当前这个 token 想知道「谁对我重要」。
- **K（Key，键）**：代表「我能提供什么」。每个 token 用来被「匹配」的标签。
- **V（Value，值）**：代表「我实际有什么信息」。真正被汇总的内容。

注意力计算的流程是：用当前 token 的 Q，去跟所有 token 的 K 做点积，算出「注意力分数」（谁和我相关），再经过 softmax 变成权重，最后用这些权重去加权求和所有 token 的 V，得到当前 token 的输出。公式长这样：

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
```

关键点在因果生成（Causal Decoding）里：第 N 个 token 只能看到第 1 到 N 个 token，不能偷看未来。于是你会注意到一个「不对称」现象——

**当我预测第 N 个 token 时，前面 N-1 个 token 的 K 和 V 是永远不变的**，只有「当前这个 token 的 Q」在变。

想想看：token 1 的 K 和 V，在预测 token 2、token 3、token 4……时，每次算出来的结果都一样。那为什么每次都要重算一遍呢？**把它们算一次、存起来，以后直接查表拿**——这就是 KV Cache 的全部思想。

### 工作机制：Prefill 和 Decode 两个阶段

有了 KV Cache，一次生成被分成两个性质完全不同的阶段：

**1. Prefill（预填充阶段，也叫「编码阶段」）**
你输入的 prompt 一次性全部进来，模型并行地把 prompt 里**所有 token** 的 K 和 V 都算出来，存进缓存。这个阶段是「并行」的（因为 prompt 是已知的，可以一次性算），同时它**计算密集**——要一口气算很多 token 的注意力。你感觉到的「它怎么想了一会儿才开始输出」，主要就是这个阶段。

**2. Decode（解码阶段，也叫「逐字生成阶段」）**
之后每生成一个新 token，模型只做两件事：① 算出**这个新 token** 的 Q、K、V，把它自己的 K、V 追加进缓存；② 用新 token 的 Q，去跟缓存里**所有历史 K** 算注意力、再对所有历史 V 加权求和，得出下一个 token。

这个阶段的特点是：每次只算一个 token，**计算量很小，但要从显存里读大量缓存数据**，所以它是**内存带宽密集（Memory-bound）**，而不是计算密集。换句话说，解码阶段的速度瓶颈不在「算得动」，而在「显存读得快不快」。

总结一句话：**KV Cache 用显存换时间**。它省掉了重复计算，代价是要在显存里常驻一大块 K、V 数据。

### 显存到底吃多少？算一笔账

这是最需要「心里有数」的部分。KV Cache 的显存占用量，可以用一个公式估算：

```
KV 缓存字节数 = 2（K 和 V 两份）× 层数 × 注意力头数 × 每个头的维度 × 序列长度 × 每个数值的字节数 × 批大小
```

举个具体例子感受一下（数值是近似）：一个 70B 参数的模型（比如 Llama 3 70B），假设 80 层、8 个 KV 头、每个头维度 128、用 16-bit（2 字节）存：

- 每个 token 大约占：2 × 80 × 8 × 128 × 2 字节 ≈ 328 KB/token
- 如果上下文拉到 128K（131072 个 token）：328 KB × 131072 ≈ **40+ GB 的显存**，光 KV 缓存这一项。

也就是说，**长上下文真正吃显存的大头，往往不是模型权重，而是 KV Cache**。权重是固定的（70B 模型 16-bit 权重约 140GB，但会量化压缩），而 KV Cache 是随对话变长**线性增长**的、没有上限的。这就是为什么「上下文窗口」虽然标着 128K、1M，实际跑起来，你的显存先撑不住了。

### 历史脉络：一条「省显存」的主线

KV Cache 严格说不是某个时刻「发明」的，而是 Transformer 走向实用化的过程中，逐渐被共识出来的一套做法。沿着时间线看，它大致经历了「出现 → 成形 → 疯狂优化」三个阶段：

- **2017 年**：Vaswani 等人发表《Attention Is All You Need》，提出 Transformer。但这篇论文主要关注**训练**（训练是并行的，不需要缓存），并没有把「解码时复用 K、V」这件事展开讲。KV Cache 的雏形，是大家在做自回归推理时才意识到、并自然形成的工程实践。
- **2019 年**：Dai 等人的 **Transformer-XL** 提出「段级循环（Segment-level Recurrence）」——把上一段的隐状态缓存下来给下一段用，从而把有效上下文拉长。这虽然不是今天意义的 KV Cache，但「缓存历史状态、跨段复用」的思想已经非常明确，是长上下文缓存的重要源头。
- **2019 年**：Shazeer 的论文《Fast Transformer Decoding: One Write-Head is All You Need》提出 **Multi-Query Attention（MQA，多查询注意力）**：让所有注意力头**共享同一套 K 和 V**，直接把 KV Cache 砍到原来的 1/头数。这是第一次明确把「省 KV Cache」当作设计目标。
- **2022 年**：Dao 等人的 **FlashAttention** 虽然主要解决的是「注意力计算时的显存读写」，但它让「大序列上的高效注意力」成为可能，和 KV Cache 优化是同一场战役里的两个战壕。
- **2023 年**：Ainslie 等人的 **GQA（Grouped-Query Attention，分组查询注意力）** 论文《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》，在「多头的表达能力」和「MQA 的省显存」之间取了个折中：把 Q 头分成几组，每组共享一套 K、V。现在 Llama 2/3、Mistral 等主流模型几乎都用 GQA，成了事实标准。
- **2023 年**：vLLM 团队的 Kwon 等人发表 **PagedAttention**（论文《Efficient Memory Management for Large Language Model Serving》），把操作系统的「虚拟内存 + 分页」思想搬进 KV Cache 管理，解决了 KV Cache 碎片化和预分配浪费的问题，吞吐量成倍提升。
- **2024 年起**：DeepSeek-V2 提出 **MLA（Multi-head Latent Attention，多头潜在注意力）**，把 K、V 先压缩到一个低维「潜在空间」再存，进一步大幅压缩 KV Cache。这是当前「省 KV Cache」方向的最新标杆之一。

万sir记住这个主线：**KV Cache 从「省重复计算」出发，演化到「怎么少存、怎么存得巧、怎么管理得高效」——本质是一场和显存做斗争的持久战。**

### 三个「省缓存」的关键招式（现在的主流做法）

**1. GQA（Grouped-Query Attention，分组查询注意力）**
前面说了，MHA 是「Q 头和 K/V 头一一对应」（缓存大），MQA 是「所有 Q 头共享一套 K/V」（缓存最小，但表达能力受损），GQA 是中间态：比如 8 个 Q 头分成 4 组，每 2 个 Q 头共享一套 K/V。KV 头数从 8 降到 4，缓存直接减半，能力损失很小。这也是为什么你看到很多模型参数表里会单独标注「KV Heads」这个数字。

**2. MLA（Multi-head Latent Attention，多头潜在注意力）**
DeepSeek 的思路更进一步：不直接存 K 和 V，而是把它们**先投影到一个维度更低的「潜在向量」**里缓存，用到时再从这个低维向量「解压」回 K、V。这样缓存的就不是完整的 K、V，而是一个小得多的中间表示。DeepSeek-V2/V3 能在超长上下文下还保持低显存、低成本，MLA 是核心功臣之一。

**3. KV Cache 量化（KV Cache Quantization）**
K、V 本来用 16-bit 存，能不能用 8-bit、甚至 4-bit 存？可以，但要小心精度损失（因为注意力分数对这些值比较敏感）。主流方案如 KIVI、KVQuant 等，用「按通道/按组缩放」等技巧，在几乎不损失效果的前提下，把 KV Cache 显存砍掉一半甚至更多。

### 更激进的方向：能不能干脆「扔掉」一部分缓存？

除了「存得省」，还有一派在问：**我非得把每个 token 的 K、V 都存着吗？能不能只保留「重要的」那些？** 这就是 **KV Cache 压缩/淘汰（Eviction）** 方向：

- **StreamingLLM（2023）**：发现模型对「开头几个 token」和「最近几个 token」最依赖，于是保留开头（Attention Sink，注意力汇点）+ 最近的，丢掉中间的大部分，实现「无限长流式生成」。
- **H2O（Heavy Hitter Oracle，重击手预言，2023）**：跟踪哪些 token 的注意力分数长期最高（「重击手」），只保留它们和最近的 token。
- **SnapKV、PyramidKV、Quest** 等（2024 前后）：用不同策略在「压缩率」和「效果」之间做权衡，目标是让几十万 token 的长上下文，也能在单卡上跑。

这些方法背后的共同洞察是：**注意力是极度稀疏的——绝大多数 token 对后续生成几乎没贡献，真正被「反复查询」的 token 很少。** 所以缓存可以「挑着存」。

### 管理层面：PagedAttention 和 Prefix Caching

省下的缓存，还得「管得好」：

- **PagedAttention（vLLM）**：传统做法是给每个请求预留一块「最大长度」的连续显存，但实际生成的往往远短于预留，造成大量浪费，而且多请求并发时碎片严重。PagedAttention 把 KV Cache 切成固定大小的「页」（page），按需分配、非连续存储，像操作系统的虚拟内存一样。这让显存利用率从约 30% 提升到 90% 以上，吞吐量大幅上涨。
- **Prefix Caching（前缀缓存）**：如果多个请求共享同一段开头（比如同一个 system prompt、同一段长文档），这段的 KV Cache 只需算一次，大家复用。SGLang 的 RadixAttention 用「基数树（Radix Tree）」来管理和复用这些公共前缀，在「多轮对话 + 共享文档」场景下能省下大量重复的 Prefill 计算。

### 一个重要「边界」：为什么 Q 不缓存？

这个问题很值得想清楚。缓存只存 K 和 V，**不存 Q**，原因是因果掩码的不对称性：

- 当前 token 要算注意力，只需要「自己的 Q」加上「所有历史 token 的 K、V」。
- 历史 token 的 Q，在预测「未来」token 时**根本用不到**（因为因果注意力里，token A 的 Q 只用来聚合它之前的 token，对「预测 A 之后的 token」没有贡献）。

所以 Q 是「一次性消耗品」，用完就丢；而 K、V 会被未来的所有 token 反复查询，才值得缓存。想通这一点，就真正理解了 KV Cache 为什么偏偏叫「KV」而不是「QKV」。

### 重要论文与人物（顺着往下挖）

- **2017**：Vaswani 等《Attention Is All You Need》——Transformer 的源头，注意力机制 Q/K/V 的出处。
- **2019**：Dai 等《Transformer-XL》——段级循环 + 缓存，长上下文缓存的先行者。
- **2019**：Shazeer《Fast Transformer Decoding: One Write-Head is All You Need》——提出 MQA，首开「省 KV Cache」设计。
- **2022**：Dao 等《FlashAttention》——IO 感知的注意力，高效长序列的基石。
- **2023**：Ainslie 等《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》——GQA 成为主流。
- **2023**：Kwon 等《Efficient Memory Management for Large Language Model Serving》——PagedAttention / vLLM。
- **2023**：Xiao 等《Efficient Streaming Language Models with Attention Sinks》——StreamingLLM，注意力汇点。
- **2023**：Zhang 等《H2O: Heavy-Hitter Oracle for Efficient Generative Inference》——重击手淘汰。
- **2024**：DeepSeek-V2 技术报告中的 **MLA**——把 KV Cache 压缩到潜在空间。

### 前沿进展（现在大家在忙什么）

- **KV Cache 压缩的「无损 vs 有损」之争**：怎么在几乎不掉点的前提下把缓存压到极致，是当前最热的课题之一。
- **KV Cache 量化 + 低比特**：4-bit 甚至 2-bit 的 KV 缓存，配合「重要性感知」的混合精度。
- **与长上下文推理引擎深度耦合**：MLA、GQA、PagedAttention、前缀缓存、淘汰策略，正在被打包进 vLLM、SGLang、TensorRT-LLM 等推理引擎，变成「开箱即用」的默认能力。
- **「免缓存」或「极低缓存」的注意力替代方案**：有人在做线性注意力（Linear Attention）、状态空间模型（SSM，如 Mamba）等，试图从根上摆脱「序列越长、缓存越大」的宿命，让推理做到「O(1) 显存随长度」。这是 KV Cache 这条主线之外、但密切相关的一条支线。

---

## ⑤ 实际场景（你在哪儿会碰到它）

- **本地跑开源模型时**：你用 llama.cpp、Ollama、vLLM 起模型，日志里会看到 `KV cache size`、`ctx_len`、`--n_ctx` 之类的参数，那就是在配置 KV 缓存的显存占用。显存不够「OOM」时，很多时候就是 KV Cache 撑爆了，而不是模型本身装不下。
- **选模型时看参数表**：看到模型标注 `num_kv_heads`（KV 头数）或「GQA / MLA」字样，你就知道它的 KV 缓存是省还是费。同样参数规模，KV 头越少，长上下文越省显存。
- **调 API / 推理引擎**：理解 Prefill（首字慢、算得多）和 Decode（逐字快、读内存多）的区别，你就能看懂为什么「短 prompt 长输出」和「长 prompt 短输出」的计费和延迟体验不一样。
- **处理长文档、多轮对话**：一个 system prompt + 一堆历史消息，其实就是在不断「拉长序列」，KV Cache 随之线性增长。做 RAG、做 agent 长会话时，你会真实撞到「上下文越长越吃显存、越慢」这堵墙。

---

## ⑥ 关联术语（先混个脸熟）

- **Attention（注意力机制）**：KV Cache 之所以成立，全靠 Q/K/V 这套机制，后面第 36 讲会单独深入讲注意力本身。
- **Quantization（量化）**：KV Cache 量化是量化的一个重要应用场景，第 35 讲会专门展开。
- **Context Window（上下文窗口）**：第 4 讲已经学过，KV Cache 就是「上下文窗口」落地时真正的显存代价，两者是一体两面。
