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
  <a href="website/learn/00-start.md">Start Learning</a> ·
  <a href="website/index.md">Documentation source</a> ·
  <a href="research/README.md">Research</a> ·
  <a href="#quick-start">Quick Start</a>
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

## Why MiniAgentRL?

很多 Agent 教程停在 `LLM + Tool + Prompt`。MiniAgentRL 再向前走一层：把 Tool Call 放进真实的多轮环境，记录完整 trajectory，计算 reward，并用真实的 verl/vLLM GRPO pipeline 更新策略。

入口足够小，底层实现保持真实：先用 CPU 示例理解协议，再进入 Qwen3 的 Tool Calling 与 one-update smoke；想研究 Agent 行为时，可以继续查看 Strict Hotpot-MT、Qwen3-8B 和工具成本分析。

## What you will build

<table>
  <tr>
    <td width="25%" valign="top"><h3>01 · Tool Calling</h3><p>把模型生成的文本 action 解析成真实工具调用。</p></td>
    <td width="25%" valign="top"><h3>02 · Multi-turn</h3><p>把 observation 放回 state，让 Agent 继续决策。</p></td>
    <td width="25%" valign="top"><h3>03 · GRPO</h3><p>从 grouped trajectories 与 relative rewards 学习。</p></td>
    <td width="25%" valign="top"><h3>04 · Efficient Tools</h3><p>区分 useful 与 wasted calls，而不是只数调用次数。</p></td>
  </tr>
</table>

## Learning path

<img src="./assets/course-roadmap.svg" alt="Learning roadmap from environment setup and tool calling to real Qwen, multi-turn agents, GRPO, and efficient tool use" width="1200" />

从 [Chapter 00](website/learn/00-start.md) 开始，或直接打开 [Learn Track source](website/learn/index.md)。完整的交互式学习体验位于 `website/`，README 只保留入口和项目地图。

## See an Agent use tools

<img src="./assets/trajectory-preview.svg" alt="Illustrative scripted trajectory with two search calls, two observations, a final answer, and tool-use accounting" width="1200" />

上图是一个 **illustrative scripted teaching fixture**，不是 benchmark 结果。项目的 analyzer 会把 `attempted → valid → executed → useful / wasted` 分开记录：一次必要的搜索，和一次没有带来新证据的搜索，不应被当成同一种成本。

## Train the Agent with GRPO

<img src="./assets/grpo-group.svg" alt="Conceptual GRPO view: four rollouts for one prompt become group-relative advantages and a policy update" width="1200" />

同一个 prompt 生成多条 trajectory，模型学习的是它们之间的相对优劣。Learn Track 已经包含一次通过真实 verl/vLLM pipeline 的 one-update GRPO smoke：它证明 rollout、reward、gradient 和 optimizer update 串起来了，但不等于 benchmark 已经提升。

## Quick Start

先跑不需要模型下载或 GPU 的 CPU 示例：

```bash
git clone https://github.com/Idiotyevsky/EfficientTool-RL.git
cd EfficientTool-RL
pip install -e ".[test]"
PYTHONPATH=src python examples/01_tool_calling.py
```

接下来：

- [Run the Learn Track](website/learn/00-start.md)
- [Run the real Qwen example](examples/README.md)
- [Run GRPO smoke](website/learn/07-grpo-smoke.md)

## Learn Track · Research Track

| Track | 适合谁 | 内容 |
| --- | --- | --- |
| **Learn Track** | 想建立 Agent RL 心智模型的学习者 | CPU-first examples、Qwen3-1.7B、trajectory inspection、真实 one-update smoke |
| **Research Track** | 想复现实验与分析行为的研究者 | Qwen3-8B、Strict Hotpot-MT、verl/vLLM、vanilla GRPO、Natural Bridge-Hard |

Research Track 的正式 strict vanilla run 仍在进行；cost-aware RL 与最终 M4/M5 结论保持 **TBD**。请从 [research/README.md](research/README.md) 和 [PROGRESS.md](PROGRESS.md) 查看证据与状态，不把 pilot/sanity evidence 当成最终 benchmark claim。

## Current status

| Area | Status |
| --- | --- |
| Learn Track | CPU examples、真实 Qwen Tool Calling、one-update GRPO smoke 已有验证记录 |
| Research pipeline | Strict Hotpot-MT + Qwen3-8B vanilla GRPO formal run in progress |
| Cost-aware objective | Planned after the vanilla gate; results TBD |

## Project structure

```text
src/efficienttool_rl/  production Agent, protocol, tools, rewards, metrics
examples/               CPU-first learning entry points
tutorials/              00→08 source tutorials
website/                VitePress learning product
configs/                reproducible training configurations
scripts/                evaluation and GRPO entry points
research/               research-facing index and evidence links
tests/                  deterministic unit/integration tests
assets/                 README visual identity and technical diagrams
```

## Credits & release status

MiniAgentRL builds on [verl](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm), Hugging Face tooling, Qwen3 and HotpotQA. Please preserve upstream licenses and dataset terms when extending or redistributing the project.

This repository currently has no top-level `LICENSE` file. Before calling the project open-source or adding a license badge, choose and add an explicit project license.
