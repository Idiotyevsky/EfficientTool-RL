<p align="center">
  <img src="./assets/logo.svg" alt="MiniAgentRL 标志" width="88" />
</p>

<h1 align="center">MiniAgentRL</h1>

<p align="center">
  <strong>从 Tool Calling 到真正的 Agent RL。</strong><br />
  从一个最小 Tool Agent 开始，逐步理解 Multi-turn Interaction、Trajectory、Reward 与 GRPO，最后亲手完成一次真实的参数更新。
</p>

<p align="center">
  <code>Qwen3</code> · <code>Tool Calling</code> · <code>Multi-turn</code> ·
  <code>GRPO</code> · <code>verl</code> · <code>vLLM</code>
</p>

<p align="center">
  <a href="README.md">中文</a> ·
  <a href="README_EN.md">English</a>
</p>

<p align="center">
  <a href="https://idiotyevsky.github.io/EfficientTool-RL/learn/00-start">开始学习</a> ·
  <a href="https://idiotyevsky.github.io/EfficientTool-RL/">在线文档</a> ·
  <a href="https://idiotyevsky.github.io/EfficientTool-RL/playground/trajectories">Trajectory Explorer</a> ·
  <a href="https://idiotyevsky.github.io/EfficientTool-RL/research/">Research</a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10 或更高版本" /></a>
  <a href="https://huggingface.co/Qwen"><img src="https://img.shields.io/badge/Qwen3-model-5B5CE2?style=flat-square" alt="Qwen3 模型系列" /></a>
  <a href="https://huggingface.co/docs/trl/main/en/grpo_trainer"><img src="https://img.shields.io/badge/GRPO-training-6F6FE8?style=flat-square" alt="GRPO 训练" /></a>
  <a href="https://github.com/volcengine/verl"><img src="https://img.shields.io/badge/verl-agent%20RL-0D8CA8?style=flat-square" alt="verl Agent RL" /></a>
  <a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/vLLM-rollouts-16845B?style=flat-square" alt="vLLM Rollout" /></a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="./assets/hero-light.svg" />
  <img src="./assets/hero-light.svg" alt="MiniAgentRL 架构：Qwen Agent 调用 Search、接收 Observation、产生 Reward，并通过 GRPO 更新策略" width="1200" />
</picture>

<p align="center"><em>从一次工具调用，到一条完整的 Agent RL 训练链路。</em></p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/course-roadmap.svg" />
  <source media="(prefers-color-scheme: light)" srcset="./assets/course-roadmap.svg" />
  <img src="./assets/course-roadmap.svg" alt="从环境检查、Tool Calling 到 Qwen、多轮 Agent、GRPO 和高效工具使用的学习路线" width="1200" />
</picture>

<p align="center"><em>从 Start 到 Efficient Tool Use 的连续学习路线。</em></p>

## 为什么做 MiniAgentRL？

很多 Agent 教程讲到 Tool Calling 就结束了：给模型定义几个工具，写一个 ReAct Prompt，让它搜索、调用 API 或执行函数。但如果继续追问：**Agent 到底是怎么训练出来的？** 问题就会变成另一套东西。

MiniAgentRL 把这条链路完整拆开：

```text
Tool Calling → Multi-turn Agent → Trajectory / Rollout
             → Reward → GRPO → Updated Policy
```

课程从小而可观察的例子开始，再逐步接回真实的 Agent RL 系统。你可以先用 CPU 理解协议和状态转移，再用 Qwen3-1.7B 运行真实 Tool Calling，最后通过 `verl + vLLM` 完成一次 GRPO 参数更新。

## 你会亲手搭出什么？

| 模块 | 你会学会什么 |
| --- | --- |
| **Tool Calling** | 区分模型文本、结构化 action 与工具执行。 |
| **Multi-turn Agent** | 让 Observation 回到上下文，并影响下一次决策。 |
| **GRPO Training** | 从 grouped rollouts、Reward 和 Advantage 走到参数更新。 |
| **Efficient Tool Use** | 区分必要探索与 wasted calls，而不是只数调用次数。 |

## 学习路线

| Chapter | 内容 | 环境 |
| --- | --- | --- |
| **00 · Start** | 环境检查与 Agent RL 全景图 | CPU |
| **01 · Tool Calling** | 从模型文本到真实工具执行 | CPU |
| **02 · Real Qwen** | 让 Qwen3 生成真实 Tool Call | GPU |
| **03 · Multi-turn** | Observation 如何进入下一状态 | CPU |
| **04 · ReAct + HotpotQA** | 在真实多跳 QA 上运行 Agent | GPU |
| **05 · Rollout & Reward** | 从完整 trajectory 计算训练信号 | CPU |
| **06 · GRPO** | 从 Reward 到 Advantage 与 Policy Update | CPU |
| **07 · Real Update** | 真正完成一次 GRPO 参数更新 | GPU |
| **08 · Efficient Tools** | 分析 useful 与 wasted Tool Calls | CPU |

推荐从 [Chapter 00](https://idiotyevsky.github.io/EfficientTool-RL/learn/00-start) 顺序开始；熟悉 ReAct 或 Function Calling 的读者，也可以直接进入 [Multi-turn](https://idiotyevsky.github.io/EfficientTool-RL/learn/03-multiturn) 或 [GRPO](https://idiotyevsky.github.io/EfficientTool-RL/learn/06-grpo)。

## 看见 Agent 的行为

<img src="./assets/trajectory-preview.svg" alt="教学轨迹：两次搜索分别获得证据，最终回答正确，并统计 executed、useful 和 wasted calls" width="1200" />

只看 Final Answer 会丢掉 Agent 最重要的信息。项目会把 `attempted → valid → executed → useful / wasted` 分开记录：一次必要的搜索，和一次没有带来新证据的搜索，不应被当成同一种成本。

## Reward 如何变成参数更新？

<img src="./assets/grpo-group.svg" alt="GRPO 概念图：同一问题的四条 rollout 产生组内相对优势并驱动 policy update" width="1200" />

同一个 Prompt 生成多条 trajectory，模型学习的是它们之间的相对优劣。Learn Track 包含一次通过真实 `verl + vLLM` pipeline 的单步训练演示：你可以亲自看到 reward、gradient 和 optimizer step。一次参数更新证明训练链路成立，但不等于 benchmark 性能已经提升。

## Quick Start

最快的入口不需要 GPU，也不需要下载模型：

```bash
git clone https://github.com/Idiotyevsky/EfficientTool-RL.git
cd EfficientTool-RL
pip install -e ".[test]"
PYTHONPATH=src python examples/01_tool_calling.py
```

你会依次看到 `Model Output → Parsed Action → Search Observation`。接下来可以[开始完整课程](https://idiotyevsky.github.io/EfficientTool-RL/learn/)，或[让 Qwen3 生成真实 Tool Call](https://idiotyevsky.github.io/EfficientTool-RL/learn/02-real-qwen)。

## Learn Track 与 Research Track

| 路线 | 内容 |
| --- | --- |
| **Learn Track** | CPU-first examples、Qwen3-1.7B、trajectory inspection 与真实 one-update smoke。 |
| **Research Track** | Qwen3-8B、Hotpot-MT Strict、Natural Bridge-Hard、verl/vLLM 与工具成本分析。 |

Research Track 已完成 Qwen3-8B 的 vanilla GRPO baseline comparison。Natural Bridge-Hard 上任务质量与多步检索均明显提升；cost-aware Tool RL 是下一阶段。详见 [Research Track](https://idiotyevsky.github.io/EfficientTool-RL/research/)。

### Latest vanilla baseline

Natural Bridge-Hard · 200 examples · Qwen3-8B Base → Step 62

| Metric | Base | Step 62 |
| --- | ---: | ---: |
| EM | 32.5% | 51.5% |
| F1 | 42.03% | 62.53% |
| Multi-search | 31.5% | 86.0% |
| Wasted search | 0.370 | 0.515 |

Vanilla GRPO improved task quality and encouraged more multi-step retrieval; both useful and wasted searches increased. This is not a cost-aware result.

## 研究问题

强化学习能否让 Multi-turn Tool Agent 在保持任务能力的同时，减少没有信息增益的工具调用？如果工具调用减少但准确率同时下降，那并不是有意义的效率提升。

## 项目结构

```text
MiniAgentRL
│
├── README.md             # 中文项目入口
├── README_EN.md          # English project entry
├── src/efficienttool_rl/ # Agent、protocol、tools、rewards、metrics
├── examples/             # 最小可运行示例
├── tutorials/            # 00→08 source tutorials
├── website/              # VitePress 在线学习网站
├── configs/              # Agent / GRPO 配置
├── scripts/              # 数据、训练与评估入口
├── research/             # 研究设计与实验说明
├── tests/                # 单元与集成测试
└── assets/               # README 视觉素材与技术图示
```

## 技术栈

| Component | 作用 |
| --- | --- |
| **Qwen3** | Agent Policy |
| **Transformers** | 本地模型推理 |
| **BM25** | 可复现的 Search Environment |
| **HotpotQA** | Multi-hop QA 任务 |
| **verl** | GRPO Training |
| **vLLM** | Rollout Generation |
| **Ray** | 分布式执行 |
| **PyTorch / FSDP** | 模型训练 |
| **VitePress** | 在线学习网站 |

## 适合谁？

MiniAgentRL 更适合已经会使用 Python、知道 Transformer / LLM 基本概念、跑过 Hugging Face 模型推理，并听说过 ReAct、PPO 或 GRPO 的读者。

它不是“零基础十分钟学会强化学习”，也不是另一个 LangChain API Demo，而是一条从 LLM inference 走向 Agentic RL / LLM Post-training 的实践路线。

## 致谢

MiniAgentRL 构建在 [Qwen](https://huggingface.co/Qwen)、[verl](https://github.com/volcengine/verl)、[vLLM](https://github.com/vllm-project/vllm)、Hugging Face Transformers 与 HotpotQA 之上。基于项目继续开发时，请遵守相关上游项目与数据集的许可证和引用要求。

## 许可证

当前仓库尚未指定项目级许可证。在补充 `LICENSE` 之前，请不要默认获得对 MiniAgentRL 项目代码进行再分发、修改或衍生使用的授权。上游依赖与数据集仍分别受其自身许可证约束。
