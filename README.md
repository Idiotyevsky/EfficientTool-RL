<p align="center"><img src="website/public/logo.svg" width="72" alt="MiniAgentRL logo"></p>

<h1 align="center">MiniAgentRL</h1>

<p align="center"><strong>Learn Agentic Reinforcement Learning by building it.</strong></p>
<p align="center">从 Tool Calling 到 Multi-turn Agent，再到真正的 GRPO 参数更新。</p>

<p align="center">
  <a href="https://idiotyevsky.github.io/EfficientTool-RL/"><strong>Learning Website</strong></a>
  · <a href="https://idiotyevsky.github.io/EfficientTool-RL/playground/trajectories">Trajectory Explorer</a>
  · <a href="research/README.md">Research Track</a>
</p>

MiniAgentRL 面向已有 Python 与基础 LLM inference 经验的读者。入口使用 CPU 示例与 Qwen3-1.7B bounded smoke；研究内核保留 Qwen3-8B、verl、vLLM、FSDP 与 native multi-turn GRPO。

## Why MiniAgentRL?

许多 Agent 教程停在 `LLM + Tool + ReAct prompt`。这里继续追问：trajectory 如何形成 Reward？同一 Prompt 的 grouped rollouts 如何产生 Advantage？这些信号如何真正更新 policy？

```text
Tool Calling → Multi-turn Interaction → Rollout → Reward
             → GRPO Update → Efficient Tool-use Analysis
```

- **Tool Calling**：区分 model output、parsed action 与 tool execution。
- **Multi-turn**：让 Observation 回到 state，并检查真实 trajectory。
- **GRPO**：从 group-relative Advantage 走到 verl/vLLM 的真实 optimizer update。
- **Efficient Tool Use**：统计 attempted、valid、executed、useful 与 wasted calls。

## Quick Start

CPU-only examples 不下载模型：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

PYTHONPATH=src python examples/00_environment_check.py
PYTHONPATH=src python examples/01_tool_calling.py
PYTHONPATH=src python examples/02_multiturn_agent.py
PYTHONPATH=src python examples/04_grpo_concepts.py
```

完整 00→08 课程、expected output、交互式 GRPO group 与 trajectory UI 在 [Learning Website](https://idiotyevsky.github.io/EfficientTool-RL/) 中。网站本地运行：

```bash
cd website
npm ci
npm run dev
```

## Research Track

```text
Strict Hotpot-MT → Qwen3-8B → Vanilla multi-turn GRPO
                 → Tool-use audit → Cost-aware RL
                 → Natural Bridge-Hard evaluation
```

Strict Hotpot-MT 是 controlled multi-turn stress test，不是未修改的 HotpotQA benchmark。Formal strict vanilla run 仍在进行，最终 M4 与所有 cost-aware/M5 结论保持 **TBD**。证据、失败记录与 artifact fingerprints 见 [research/README.md](research/README.md) 和 [PROGRESS.md](PROGRESS.md)。

## Repository Map

```text
website/                 VitePress learning product and interactive components
src/efficienttool_rl/    production agent, protocol, tools, rewards, evaluation
examples/                bounded entry points that reuse production code
tutorials/               legacy Markdown source and compatibility notes
configs/ + scripts/      real verl/vLLM training, evaluation, and analysis
research/ + docs/        experiment index, evidence, environment, debug history
tests/                   deterministic unit and integration tests
```

Public brand 使用 MiniAgentRL；Python package、配置变量与已有 artifact 继续使用 `efficienttool_rl` / `ETRL_*`，避免破坏研究复现。

## Credits and release status

项目基于 Qwen、HotpotQA、Hugging Face、verl 与 vLLM；再分发时请保留各上游项目和数据集的 license/citation 要求。

**Public release blocker:** 本仓库当前没有顶层项目 LICENSE。在 owner 明确选择 license 之前，请勿将仓库宣传为已授权的 open-source project。

Contributor 与 research-safety 规则见 [AGENTS.md](AGENTS.md)。
