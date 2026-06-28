> **调研环节**：⑥ Agent 循环、编排与护栏/评测（智能体诞生 + 生产现实）
> **来源**：并行深度调研 · 研究员 F（general-purpose / sonnet）
> **时间**：2026 年 6 月
> **说明**：研究员原始返回，基本未压缩。综合版见主报告第一部分「环⑥」。
> ⚠️ **勘误**：原始返回里把「workflow=流水线 vs agent=手艺人」当作 Anthropic 原话；经核实，Anthropic《Building Effective Agents》原文只区分 workflow（沿预定义代码路径）与 agent（动态自主），**并无「手艺人/craftsman」此词**——「流水线 vs 手艺人」是便于理解的二次转译，引用勿当原文。

---

## 1. 一句话定位 + 它在全链路中的位置

**到这一环，打字机终于「成了智能体」。** 此前所有工程层——tokenizer、推理引擎、KV Cache、工具调用接口——都是在让模型「能说话、能算快、能伸手」；本环解决的是：把那只「伸出来的手」装进一个**自驱动的循环**里，赋予模型感知-思考-行动-再感知的闭环能力，于是静态语言预测器真正变为可完成多步骤任务的**自主智能体**。

## 2. 通俗讲法（破壁课用）

**比喻 A：特工接任务**——不在出发前把每步都计划好，而是观察（见保安）→思考→行动（绕道）→再观察。这就是 ReAct 的「边走边看边想」。（参考 Lilian Weng《LLM Powered Autonomous Agents》对 ReAct 的描述。）

**比喻 B：实验室里的研究生**——查文献（工具）→发现漏洞→再查→写草稿→请导师批注（反思/Reflexion）→修改→再提交。Reflexion（Shinn et al., NeurIPS 2023）正是把「做完一轮→自我批评→带批注重试」编码进语言智能体。

**比喻 C：流水线 vs 手艺人**（⚠️ 通俗转译，**非 Anthropic 原话**）：Anthropic《Building Effective Agents》原文区分 **workflow**（LLM 与工具沿预先写好的代码路径编排）与 **agent**（LLM 动态自主决定流程与工具）。「流水线 vs 手艺人」是便于理解的二次表述。原文核心建议：「寻找最简方案，只在必要时才增加智能体的复杂度。」

## 3. 硬核机制（报告用）

### 3.1 ReAct（Reasoning + Acting）
Yao et al. 2022-10《ReAct: Synergizing Reasoning and Acting in Language Models》（arXiv:2210.03629，ICLR 2023；Princeton + Google）。核心：将**推理轨迹**与**外部行动**交错生成于同一序列：
```
Thought: [对当前状态的分析]
Action: [调用工具，如 Search("量子计算")]
Observation: [工具返回结果]
(循环，直到完成或触发终止)
```
比纯 CoT 减少幻觉（用真实工具结果更新推理链），比纯行动更可解释。
基准：HotpotQA/FEVER 显著超 baseline；ALFWorld **+34%** 绝对成功率；WebShop **+10%**。

### 3.2 Reflexion（反思 / 言语强化）
Shinn et al.（arXiv:2303.11366，NeurIPS 2023）。不更新权重，而把每轮失败反馈转化为**语言批评**，存入**情节记忆缓冲区**，下轮以 in-context 注入（相当于「错题本」）。HumanEval pass@1 = **91%**，超 GPT-4 的 80%。
与 ReAct 关系：ReAct 处理单轮内的 think-act-observe；Reflexion 在轮次之间加「回顾-反思-记忆」元循环，可叠加。

### 3.3 编排模式：Planner–Executor–Critic
| 角色 | 职责 |
|---|---|
| Planner（规划器）| 接收高层目标，分解为子任务 |
| Executor（执行器）| 接收原子任务，调用具体工具 |
| Critic/Evaluator（评估器）| 检查结果，决定是否重新规划 |

### 3.4 编排框架全景
- **LangChain**（~99K star）：最早的 LLM 应用框架，奠定「chain」抽象；线性链式不适合有循环的 Agent。
- **LangGraph**（~30K+ star）：2024 起官方主推，以**有向有状态图**建模，原生支持循环、条件分支、状态持久化；2025 底 v1.0；400+ 公司生产部署。LangChain 官方立场：「做 Agent 用 LangGraph。」
- **AutoGPT**（~18.3 万 star）：2023-03-30 发布，两周破 10 万 star（被称 GitHub 史上增长最快开源项目之一）。承诺「全自动」，但败给**复合可靠性衰减**（每步 85%，10 步后 ≈20%；每步 95%，20 步后 ≈36%）；最终转型为带人工节点的可视化工作流构建器。教训：「GitHub star 是带幻觉的书签」，star 与实际下载相关性仅 0.14。
- **CrewAI**（~48K star）：角色扮演型多智能体编排，2 billion+ agent executions，150+ 企业客户。
- **OpenAI Swarm → Agents SDK**：Swarm（2024-10，教育性实验，<1000 行）→ Agents SDK（2025-03，生产级，加护栏、追踪、TS）。

### 3.5 Anthropic《Building Effective Agents》核心框架（2024-12）
**核心区分**：Workflow（LLM 与工具沿**预定义代码路径**编排，决策逻辑在代码里）vs Agent（LLM **动态自主**决定调用哪些工具、何顺序、何时停止）。
**五种工作流模式**：① Prompt Chaining ② Routing ③ Parallelization ④ Orchestrator-Workers ⑤ Evaluator-Optimizer。
**核心建议原文**：
> "finding the simplest solution possible, and only increasing complexity when needed"
> "The autonomous nature of agents means higher costs, and the potential for compounding errors."

### 3.6 生产现实：护栏、评测与三大代价
- **Prompt Injection 防御**：OWASP 2026 仍列 LLM 安全 #1。间接注入（恶意内容伪装成工具返回结果）。Anthropic 多层防御：训练识别 + 流量监控 + 红队 + 最小权限 + 子 Agent 结果过安全分类器。
- **Human-in-the-Loop**：三层模型——in-the-loop（执行中高风险步骤须人审）/ on-the-loop（完成后复核）/ out-of-the-loop（低风险全自动）。EU AI Act 第 14 条、NIST AI RMF 要求高风险系统人类监督。
- **为何 Agent 评测更难**：要衡量多步会话的目标级结果。《AI Agents That Matter》（arXiv:2407.01502, 2024）指出当前基准过于聚焦准确率、忽视成本。可靠性衰减：整体成功率 = pⁿ（单步 95% × 20 步 = 36%；单步 99% × 20 步 = 82%）。
- **成本**：Agent 约为普通对话 4 倍 token；多智能体约 15 倍。每轮迭代典型 $0.50–$2.00。
- **延迟**：每轮工具调用+推理增加 1–5 秒。

## 4. 权威来源清单

| 类型 | 标题 | URL |
|---|---|---|
| 论文 | ReAct（Yao et al., ICLR 2023）| https://arxiv.org/abs/2210.03629 |
| 论文 | Reflexion（Shinn et al., NeurIPS 2023）| https://arxiv.org/abs/2303.11366 |
| 论文 | AI Agents That Matter（2024）| https://arxiv.org/abs/2407.01502 |
| 博客 | Lilian Weng《LLM Powered Autonomous Agents》| https://lilianweng.github.io/posts/2023-06-23-agent/ |
| 博客 | Anthropic《Building Effective Agents》| https://www.anthropic.com/research/building-effective-agents |
| 博客 | Anthropic《How We Built Our Multi-Agent Research System》| https://www.anthropic.com/engineering/multi-agent-research-system |
| GitHub | Significant-Gravitas/AutoGPT（~18.3万 star）| https://github.com/Significant-Gravitas/AutoGPT |
| GitHub | langchain-ai/langgraph | https://github.com/langchain-ai/langgraph |
| GitHub | crewAIInc/crewAI（~48K star）| https://github.com/crewAIInc/crewAI |
| GitHub | openai/swarm → Agents SDK | https://github.com/openai/swarm |
| 分析 | AutoGPT Got 100K Stars and Then What? | https://vibeagentmaking.com/blog/autogpt-got-100k-stars-and-then-what/ |
| 知乎 | AI Agent模式全景图：从ReAct到Multi-Agent | https://zhuanlan.zhihu.com/p/1968071452067620000 |

## 5. 金句 / 高赞观点 / 关键数据

- 「Agent = LLM + memory + planning skills + tool use.」—— Lilian Weng（2023-06）
- 「Agentic systems often trade latency and cost for better task performance.」—— Anthropic
- 「The autonomous nature of agents means higher costs, and the potential for **compounding errors**.」—— Anthropic
- 可靠性衰减：单步 95% × 20 步 = **36%**（0.95²⁰）。
- AutoGPT：13 天破 10 万 star；Reflexion HumanEval pass@1 = 91%；ReAct ALFWorld +34%。
- Agent token 消耗约普通对话 4 倍，多智能体约 15 倍。

## 6. 常见误解与「经不起推敲」的坑

1. ❌「Agent 就是套个 while 循环，很简单」。✅ 难在终止条件、跨步状态维护、防注入劫持。
2. ❌「AutoGPT 全自动」。✅ 可靠性数学 pⁿ 使多步任务必然衰减；2024 已转型为带人工节点工作流。
3. ❌「智能体越多越好」。✅ 复杂多智能体常在成本评测里输给简单单体。
4. ❌「Demo 能运行 = 生产可用」。✅ 真实用户触发 edge case、延迟累加、成本量级跃升。
5. ❌「Prompt Injection 是小概率，上线后再处理」。✅ OWASP #1；子 Agent 被污染可链式传播（Prompt Infection）。

## 7. 从「打字机」到「智能体」整条链路的一句话回望

从 **Transformer 的 token 打字机**，经 **推理引擎** 把预测加速至可用，再通过 **工具调用** 让模型伸出手，最后由 **ReAct 循环** 把那只手装进一个持续感知-思考-行动的闭环——一个冻结在参数里的语言预测器，通过工程编排，活成了能在真实世界「边走边想、边想边做、知错能改」的自主智能体。而 AutoGPT 的兴衰提醒：**让模型「会循环」只是起点，让循环「可靠、可控、可审计」才是工程的全部难度。**
