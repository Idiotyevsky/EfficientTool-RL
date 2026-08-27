<p align="center">
  <img src="./assets/logo.svg" alt="MiniAgentRL logo" width="88" />
</p>

<h1 align="center">MiniAgentRL</h1>

<p align="center">
  <strong>Learn Agentic RL by building it.</strong><br />
  从 Tool Calling 到 Multi-turn Agent，再到真正的 GRPO 参数更新。
</p>

<p align="center">
  <code>Qwen3</code> · <code>Tool Calling</code> · <code>Multi-turn</code> ·
  <code>GRPO</code> · <code>verl</code> · <code>vLLM</code>
</p>

<p align="center">
  <a href="#中文">中文</a> ·
  <a href="#english">English</a>
</p>

<p align="center">
  <a href="https://idiotyevsky.github.io/EfficientTool-RL/learn/00-start">开始学习 / Start Learning</a> ·
  <a href="https://idiotyevsky.github.io/EfficientTool-RL/">在线文档 / Documentation</a> ·
  <a href="https://idiotyevsky.github.io/EfficientTool-RL/playground/trajectories">Trajectory Explorer</a> ·
  <a href="https://idiotyevsky.github.io/EfficientTool-RL/research/">Research</a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10 or newer" /></a>
  <a href="https://huggingface.co/Qwen"><img src="https://img.shields.io/badge/Qwen3-model-5B5CE2?style=flat-square" alt="Qwen3 model family" /></a>
  <a href="https://huggingface.co/docs/trl/main/en/grpo_trainer"><img src="https://img.shields.io/badge/GRPO-training-6F6FE8?style=flat-square" alt="GRPO training" /></a>
  <a href="https://github.com/volcengine/verl"><img src="https://img.shields.io/badge/verl-agent%20RL-0D8CA8?style=flat-square" alt="verl agent reinforcement learning" /></a>
  <a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/vLLM-rollouts-16845B?style=flat-square" alt="vLLM rollouts" /></a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="./assets/hero-light.svg" />
  <img src="./assets/hero-light.svg" alt="MiniAgentRL architecture: a Qwen agent calls search, receives observations, produces a reward, and updates with GRPO" width="1200" />
</picture>

<p align="center"><em>从一次工具调用，到一条完整的 Agent RL 训练链路。<br />From a single tool call to a complete Agentic RL training loop.</em></p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/course-roadmap.svg" />
  <source media="(prefers-color-scheme: light)" srcset="./assets/course-roadmap.svg" />
  <img src="./assets/course-roadmap.svg" alt="Learning roadmap from environment setup and tool calling to real Qwen, multi-turn agents, GRPO, and efficient tool use" width="1200" />
</picture>

<p align="center"><em>从 Start 到 Efficient Tool Use 的连续学习路线。<br />A continuous path from Start to Efficient Tool Use.</em></p>

## 中文

### 为什么做 MiniAgentRL？

很多 Agent 教程讲到 Tool Calling 就结束了：给模型定义几个工具，写一个 ReAct Prompt，让它搜索、调用 API 或执行函数。但如果继续追问：**Agent 到底是怎么训练出来的？** 问题就会变成另一套东西。

MiniAgentRL 把这条链路完整拆开：

```text
Tool Calling → Multi-turn Agent → Trajectory / Rollout
             → Reward → GRPO → Updated Policy
```

课程从小而可观察的例子开始，再逐步接回真实的 Agent RL 系统。你可以先用 CPU 理解协议和状态转移，再用 Qwen3-1.7B 运行真实 Tool Calling，最后通过 `verl + vLLM` 完成一次 GRPO 参数更新。

### 你会亲手搭出什么？

| 模块 | 你会学会什么 |
| --- | --- |
| **Tool Calling** | 区分模型文本、结构化 action 与工具执行。 |
| **Multi-turn Agent** | 让 Observation 回到上下文，并影响下一次决策。 |
| **GRPO Training** | 从 grouped rollouts、Reward 和 Advantage 走到参数更新。 |
| **Efficient Tool Use** | 区分必要探索与 wasted calls，而不是只数调用次数。 |

### 学习路线

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

### 看见 Agent 的行为

<img src="./assets/trajectory-preview.svg" alt="教学轨迹：两次搜索分别获得证据，最终回答正确，并统计 executed、useful 和 wasted calls" width="1200" />

只看 Final Answer 会丢掉 Agent 最重要的信息。项目会把 `attempted → valid → executed → useful / wasted` 分开记录：一次必要的搜索，和一次没有带来新证据的搜索，不应被当成同一种成本。

### Reward 如何变成参数更新？

<img src="./assets/grpo-group.svg" alt="GRPO 概念图：同一问题的四条 rollout 产生组内相对优势并驱动 policy update" width="1200" />

同一个 Prompt 生成多条 trajectory，模型学习的是它们之间的相对优劣。Learn Track 包含一次通过真实 `verl + vLLM` pipeline 的单步训练演示：你可以亲自看到 reward、gradient 和 optimizer step。一次参数更新证明训练链路成立，但不等于 benchmark 性能已经提升。

### Quick Start

最快的入口不需要 GPU，也不需要下载模型：

```bash
git clone https://github.com/Idiotyevsky/EfficientTool-RL.git
cd EfficientTool-RL
pip install -e ".[test]"
PYTHONPATH=src python examples/01_tool_calling.py
```

你会依次看到 `Model Output → Parsed Action → Search Observation`。接下来可以[开始完整课程](https://idiotyevsky.github.io/EfficientTool-RL/learn/)，或[让 Qwen3 生成真实 Tool Call](https://idiotyevsky.github.io/EfficientTool-RL/learn/02-real-qwen)。

### Learn Track 与 Research Track

| 路线 | 内容 |
| --- | --- |
| **Learn Track** | CPU-first examples、Qwen3-1.7B、trajectory inspection 与真实 one-update smoke。 |
| **Research Track** | Qwen3-8B、Hotpot-MT Strict、Natural Bridge-Hard、verl/vLLM 与工具成本分析。 |

Research Track 正在评估 Qwen3-8B 上的 vanilla GRPO；cost-aware 训练将在基线评估完成后开展，结果会随实验进展更新。详见 [Research Track](https://idiotyevsky.github.io/EfficientTool-RL/research/)。

### 研究问题

强化学习能否让 Multi-turn Tool Agent 在保持任务能力的同时，减少没有信息增益的工具调用？如果工具调用减少但准确率同时下降，那并不是有意义的效率提升。

## English

### Why MiniAgentRL?

Many Agent tutorials stop at Tool Calling: define a few tools, write a ReAct-style prompt, and let the model search, call APIs, or execute functions. The more interesting question is: **how is the agent actually trained?**

MiniAgentRL breaks that full pipeline into understandable pieces:

```text
Tool Calling → Multi-turn Agent → Trajectory / Rollout
             → Reward → GRPO → Updated Policy
```

The course starts with small, inspectable examples and gradually reconnects them into a real Agentic RL system. Begin with CPU lessons, let Qwen3-1.7B generate real Tool Calls, and finish with an actual GRPO parameter update through `verl + vLLM`.

### What will you build?

| Module | What you learn |
| --- | --- |
| **Tool Calling** | Understand the boundary between model text, structured actions, and tool execution. |
| **Multi-turn Agent** | Feed observations back into context and change the next decision. |
| **GRPO Training** | Connect grouped rollouts, rewards, and advantages to a parameter update. |
| **Efficient Tool Use** | Separate necessary exploration from wasted calls instead of counting calls alone. |

### Learning path

| Chapter | What you learn | Runtime |
| --- | --- | --- |
| **00 · Start** | Environment check and the Agentic RL map | CPU |
| **01 · Tool Calling** | From generated text to actual tool execution | CPU |
| **02 · Real Qwen** | Let Qwen3 generate a real Tool Call | GPU |
| **03 · Multi-turn** | How observations become the next state | CPU |
| **04 · ReAct + HotpotQA** | Run the agent on real multi-hop QA | GPU |
| **05 · Rollout & Reward** | Turn complete trajectories into reward | CPU |
| **06 · GRPO** | From reward to advantage and policy update | CPU |
| **07 · Real Update** | Run an actual GRPO parameter update | GPU |
| **08 · Efficient Tools** | Analyze useful and wasted tool calls | CPU |

Start with [Chapter 00](https://idiotyevsky.github.io/EfficientTool-RL/learn/00-start), or jump to [Multi-turn](https://idiotyevsky.github.io/EfficientTool-RL/learn/03-multiturn) or [GRPO](https://idiotyevsky.github.io/EfficientTool-RL/learn/06-grpo) if you already know ReAct or Function Calling.

### See the agent's behavior

A final answer hides most of what matters in an agent trajectory. MiniAgentRL separates `attempted → valid → executed → useful / wasted` so a necessary search is not treated as the same cost as a call that adds no new evidence.

### How does reward become a parameter update?

The Learn Track includes a real one-update demonstration through the `verl + vLLM` pipeline. You can inspect reward, gradient, and optimizer-step evidence. A successful update proves that the training path works; it does not, by itself, prove benchmark improvement.

### Quick Start

The fastest entry point requires no GPU and no model download:

```bash
git clone https://github.com/Idiotyevsky/EfficientTool-RL.git
cd EfficientTool-RL
pip install -e ".[test]"
PYTHONPATH=src python examples/01_tool_calling.py
```

You should see `Model Output → Parsed Action → Search Observation`. Next, [start the full course](https://idiotyevsky.github.io/EfficientTool-RL/learn/) or [let Qwen3 generate a real Tool Call](https://idiotyevsky.github.io/EfficientTool-RL/learn/02-real-qwen).

### Learn Track and Research Track

| Track | What it contains |
| --- | --- |
| **Learn Track** | CPU-first examples, Qwen3-1.7B, trajectory inspection, and a real one-update smoke. |
| **Research Track** | Qwen3-8B, Hotpot-MT Strict, Natural Bridge-Hard, verl/vLLM, and tool-cost analysis. |

The Research Track is currently evaluating vanilla GRPO on Qwen3-8B. Cost-aware training will follow after the baseline evaluation is complete; results will be updated as experiments progress. See the [Research Track](https://idiotyevsky.github.io/EfficientTool-RL/research/).

### Research question

Can reinforcement learning reduce tool calls that add no new information while preserving the capability of a multi-turn tool agent? If tool usage decreases while accuracy also drops, that is not a meaningful efficiency improvement.

## Project Structure

```text
MiniAgentRL
│
├── src/efficienttool_rl/  # core Agent, protocol, tools, rewards, metrics
├── examples/              # minimal runnable examples
├── tutorials/             # 00→08 source tutorials
├── website/               # VitePress learning website
├── configs/               # reproducible Agent / GRPO configurations
├── scripts/               # data, training, and evaluation entry points
├── research/              # research design and experiment notes
├── tests/                 # unit and integration tests
└── assets/                # README visual identity and technical diagrams
```

## Tech Stack / 技术栈

| Component | Role / 作用 |
| --- | --- |
| **Qwen3** | Agent policy / Agent Policy |
| **Transformers** | 本地模型推理 / Local model inference |
| **BM25** | 可复现的 Search Environment / Reproducible search environment |
| **HotpotQA** | Multi-hop QA 任务 / Multi-hop QA tasks |
| **verl** | GRPO training / GRPO Training |
| **vLLM** | Rollout generation / Rollout Generation |
| **Ray** | Distributed execution / 分布式执行 |
| **PyTorch / FSDP** | Model training / 模型训练 |
| **VitePress** | Learning website / 在线学习网站 |

## Who is this for? / 适合谁？

MiniAgentRL is for readers who know Python, basic Transformer / LLM concepts, and Hugging Face inference, and have heard of ReAct, PPO, or GRPO. It is a practical path from LLM inference to Agentic RL / LLM post-training.

MiniAgentRL 不是“零基础十分钟学会强化学习”，也不是另一个 LangChain API Demo。它更适合想从 LLM inference 继续走向 Agentic RL / LLM Post-training 的读者。

## Credits / 致谢

MiniAgentRL builds on [Qwen](https://huggingface.co/Qwen), [verl](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm), Hugging Face Transformers, and HotpotQA. Please follow the licensing and citation requirements of the corresponding upstream projects and datasets.

## License / 许可证

A project-level license has not yet been specified. Until a `LICENSE` is added, please do not assume permission to redistribute, modify, or create derivative works from MiniAgentRL itself.

当前仓库尚未指定项目级许可证。在补充 `LICENSE` 之前，请不要默认获得对 MiniAgentRL 项目代码进行再分发、修改或衍生使用的授权。
