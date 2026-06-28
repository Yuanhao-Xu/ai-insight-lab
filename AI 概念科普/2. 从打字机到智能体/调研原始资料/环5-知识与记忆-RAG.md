> **调研环节**：⑤ 知识与记忆（RAG + Memory）
> **来源**：并行深度调研 · 研究员 E（general-purpose / sonnet）
> **时间**：2026 年 6 月
> **说明**：研究员原始返回，基本未压缩。综合版见主报告第一部分「环⑤」。

---

# 第 5 环「知识与记忆」——给打字机一本随时可查的书，和一个笔记本

## 1. 一句话定位 + 它在全链路中的位置

**RAG（Retrieval-Augmented Generation，检索增强生成）** 和 **Memory（记忆）** 是解决 LLM「参数化知识冻结」问题的两条并行工程路径：RAG 给模型接上一个实时可查的外部「图书馆」，Memory 给模型配备一个跨会话的「笔记本」。
- 上游（环 4）：工具调用——RAG 本质上就是工具调用的特化形态，检索器即工具。
- 下游（环 6）：Agent 循环需要 RAG 提供知识 grounding，需要 Memory 提供跨步骤状态延续。没有本环，Agent 就是「失忆的行动者」。

## 2. 通俗讲法（破壁课用）

**比喻 A：参数记忆 vs 检索记忆 = 学生的「长期记忆」vs「开卷考试」**
语言模型的参数权重相当于学生合上书本后大脑里固化的知识——训练截止就冻结了。RAG 相当于允许学生带参考书进考场：不用背下整本书，翻到相关页直接参考。**Lewis et al.（2020）** 原始论文将两类知识命名为 **参数记忆（parametric memory）** 与 **非参数记忆（non-parametric memory）**。

**比喻 B：向量检索 = 「语义气味」而非「关键词匹配」**
关键词搜索（BM25）像按书名找书：搜「苹果公司」只能找含「苹果」的记录。**Embedding（向量化）** 像把每段文字提炼成一瓶「语义香水」，含义相近气味接近——即使用词完全不同也能检索到。这是 **DPR（Dense Passage Retrieval）**（Karpukhin et al., EMNLP 2020）相较 BM25 的核心跨越。

**比喻 C：短期记忆 vs 长期记忆 = RAM vs 硬盘**
对话上下文是 RAM：快、直接，但断电即清、容量有限（再大也遭遇 lost-in-the-middle）。长期记忆（写入向量库/结构化存储）是硬盘：需「I/O」（检索）才能读取，但持久可扩展。ChatGPT Memory 将此产品化。

## 3. 硬核机制（报告用）

### A. RAG 核心管线

**离线索引阶段：** 原始文档 → 分块 Chunking（512–1024 token，20–25% 重叠）→ 向量化 Embedding（双塔 bi-encoder）→ 建索引（HNSW 近似最近邻）。
- 分块的坑：块太大检索噪声大；块太小语义不完整。
- Embedding 模型：OpenAI `text-embedding-3-large`（MTEB 64.6%）、BAAI/BGE-large-en-v1.5（63–65%，MIT License）、Voyage AI。相似度度量须与训练损失一致。
- 向量数据库：FAISS（~33k star，纯内存）、Milvus（~40k+，生产首选）、Chroma（~6k，原型）、Weaviate（~12k+）、Qdrant（~22k+，Rust）。
- 索引算法：**HNSW（分层可导航小世界）**，Malkov & Yashunin 2016，是绝大多数向量库默认。

**在线检索阶段：** Query → Query Embedding → ANN 检索（top-K，K=20 常见起点）→（可选）重排 Rerank（交叉编码器 cross-encoder 精排）→（可选）混合检索 Hybrid（稠密向量 + BM25，RRF 融合）→ 注入上下文 → 生成。
- 重排：bi-encoder 静态向量无法 query-time 精细交互；cross-encoder 拼接后 full attention，只适合精排少量候选（50–100 条）。代表：Cohere Rerank。
- 混合检索：Hybrid Search 召回率可达 91%@10，相比纯向量 +17%。

**Anthropic Contextual Retrieval（2024）：** 在 embedding 前用 Claude 为每个 chunk 生成 50–100 token 上下文前缀，再 embedding + BM25。实测（官方）：
- Contextual Embeddings 单用：检索失败率 5.7% → 3.7%（降 35%）
- + BM25 混合：降 49%（→ 2.9%）
- + Reranking：降 **67%**（→ 1.9%）
- 成本：借助 prompt caching，约 $1.02 / 百万 document tokens。

**RAG 三范式（Gao et al., arXiv:2312.10997）：** Naive RAG / Advanced RAG（chunking 优化、reranking、HyDE）/ Modular RAG（可插拔，接近 Agentic）。

### B. Memory 记忆机制

**短期记忆（In-Context）：** 对话历史在 context window 内直接堆叠，零延迟零检索，但受窗口容量上限 + Lost-in-the-Middle 约束。

**长期记忆（External / Persistent）：** 跨会话信息写入外部存储，需要时检索回来。技术选型：向量库（语义检索）+ 结构化 KV（精确匹配）。
- 写入策略：事件触发写入；Dreaming（OpenAI 2025-04 上线，后台异步整理）；ADD-only 追加（Mem0，支持时序推理）。
- 读取策略：语义召回 + 实体链接 + 多信号融合（RRF）。
- 产品化：ChatGPT Memory（2024-04 首发，2025-04 Dreaming 升级，factual recall 67.9%→82.8%）；Mem0（mem0ai/mem0，~50k star，比 full-context 方案 p95 延迟低 91%、token 成本降 90%，LOCOMO 91.6）。

## 4. 权威来源清单

1. **[论文-原始RAG]** Lewis et al. NeurIPS 2020 — https://arxiv.org/abs/2005.11401
2. **[论文-DPR]** Karpukhin et al. EMNLP 2020 — https://github.com/facebookresearch/DPR
3. **[论文-Survey]** Gao et al.《RAG for LLMs: A Survey》— https://arxiv.org/abs/2312.10997
4. **[论文-Lost in the Middle]** Liu et al. — https://arxiv.org/abs/2307.03172
5. **[论文-Mem0]** — https://arxiv.org/abs/2504.19413
6. **[官方博客]** Anthropic《Contextual Retrieval》(2024-09) — https://www.anthropic.com/news/contextual-retrieval
7. **[官方博客]** OpenAI《Memory and new controls for ChatGPT》(2024-04) — https://openai.com/index/memory-and-new-controls-for-chatgpt/
8. **[官方文档]** Cohere Rerank（AWS Bedrock）— https://aws.amazon.com/blogs/machine-learning/cohere-rerank-3-5-is-now-available-in-amazon-bedrock-through-rerank-api/
9. **[官方文档]** OpenAI embedding 模型 — https://openai.com/index/new-embedding-models-and-api-updates/
10. **[GitHub]** run-llama/llama_index（~48k star）— https://github.com/run-llama/llama_index
11. **[GitHub]** langchain-ai/langchain（~130k star）— https://github.com/langchain-ai/langchain
12. **[GitHub]** milvus-io/milvus（~40k+ star）— https://github.com/milvus-io/milvus
13. **[GitHub]** mem0ai/mem0（~50k star）— https://github.com/mem0ai/mem0
14. **[知乎]** 「2025 年末 RAG 技术全景总结」— https://zhuanlan.zhihu.com/p/1987561705794986878
15. **[知乎]** 「万字详解 RAG 向量索引算法和向量数据库」— https://zhuanlan.zhihu.com/p/2029585116058370657
16. **[知乎]** 「RAG 应用开发之文本嵌入与重排序模型」— https://zhuanlan.zhihu.com/p/12508335512

## 5. 金句 / 高赞观点 / 关键数据

- 「We combine **parametric memory** with **non-parametric memory**.」—— Lewis et al. NeurIPS 2020
- 「Reranking 是 RAG 系统里 ROI 最高的单项改进。」
- Contextual Retrieval：检索失败率最高降 **67%**（5.7%→1.9%）。
- Hybrid Search vs Dense-Only：91% recall@10，+17%。
- Mem0：vs full-context，p95 延迟降 91%、token 成本降 90%，LOCOMO 91.6。
- ChatGPT Memory（Dreaming）：factual recall 67.9%→82.8%。

## 6. 常见误解与「经不起推敲」的坑

1. ❌「RAG = 搜索 + 粘贴」。✅ 忽略 chunking/rerank/hybrid 的坑；naive RAG 生产中常很差。
2. ❌「向量检索能彻底消灭幻觉」。✅ 只能减少：检索不到 / 检索到却不用 / 跨文档推理出错。
3. ❌「Embedding = 关键词搜索」。✅ 是语义相似度；对精确 ID/错误码反弱于 BM25，故需混合检索。
4. ❌「长期记忆 = 把所有历史塞进上下文」。✅ 贵且受 Lost-in-the-Middle 拖累。
5. ❌「RAG 和 Fine-tuning 是同层次选择」。✅ 微调改「说话方式」，RAG 给「最新可溯源知识」，常组合使用。

## 7. 与上一环（工具调用）/ 下一环（Agent 循环）的衔接逻辑

- **RAG 是工具调用的「语义特化形态」**：检索器即一种工具（LlamaIndex `QueryEngine`、LangChain `RetrieverTool`）。区别：普通工具返回结构化数据，RAG 返回非结构化文本段落，需后续 generation 合成。
- **Memory 是 Agent 循环的「状态载体」**：短期记忆 = episode 内 scratchpad；长期记忆 = 跨 episode 知识积累。没有 Memory，Agent 每次行动后失忆。**Agentic RAG** 是融合两环的主流模式：Agent 在 ReAct 循环中动态决策「何时检索、检索什么、是否足够」。
