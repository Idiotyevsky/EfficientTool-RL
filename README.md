# MiniAgentRL

**Learn Agentic Reinforcement Learning by building a real multi-turn tool agent with GRPO.**

MiniAgentRL is a minimal but complete learning project for moving from ordinary LLM inference to Agentic RL. It follows the full loop:

**Tool Calling → Multi-turn Interaction → Rollout → Reward → GRPO → Efficient Tool Use**

The public-facing brand is MiniAgentRL. The repository and Python package still use EfficientTool-RL and efficienttool_rl for compatibility; no package-wide rename is required.

## Why this project?

Many agent tutorials stop at “LLM + tool + ReAct prompt”. This project shows how to train the agent: a Qwen policy interacts with a deterministic search environment, produces trajectories, receives task reward, and is optimized with grouped GRPO rollouts.

## Architecture

~~~mermaid
flowchart LR
  Q[Question] --> A[Qwen Agent]
  A --> C[Tool Call]
  C --> S[Deterministic BM25 Search]
  S --> O[Observation]
  O --> A
  A --> F[Final Answer]
  F --> R[Task Reward]
  R --> G[GRPO]
  G --> P[Updated Policy]
~~~

## What you will learn

- Tool schemas, tagged actions, parsing, and structured observations.
- Multi-turn agent state, actions, trajectories, termination, and ReAct.
- Rollouts, sparse task reward, grouped rewards, and GRPO’s relative signal.
- verl/vLLM integration and the cost of agentic rollouts.
- How to measure necessary versus wasted tool use.

## Learning path

Follow the hands-on [00→08 course](tutorials/):

1. [00 — Environment and Agent RL map](tutorials/00_environment_and_map.md)
2. [01 — Tool Calling basics](tutorials/01_tool_calling_basics.md)
3. [02 — Qwen’s first real Tool Call](tutorials/02_real_qwen_tool_calling.md)
4. [03 — Multi-turn Agent](tutorials/03_multiturn_agent.md)
5. [04 — ReAct + HotpotQA](tutorials/04_react_hotpot.md)
6. [05 — Rollouts, Environment, and Reward](tutorials/05_rollouts_rewards_environment.md)
7. [06 — GRPO from formula to config](tutorials/06_grpo_for_agents.md)
8. [07 — Real GRPO smoke](tutorials/07_grpo_smoke.md)
9. [08 — Efficient Tool Use](tutorials/08_efficient_tool_use.md)

The matching runnable entry points are indexed in [examples/README.md](examples/README.md). The [research track](research/) documents the full Qwen3-8B experiment.

## Quick Start — Learn Track

The CPU-only examples below use deterministic inputs and do not download a model.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

PYTHONPATH=src python examples/00_environment_check.py
PYTHONPATH=src python examples/01_tool_calling.py
PYTHONPATH=src python examples/02_multiturn_agent.py
PYTHONPATH=src python examples/04_grpo_concepts.py
python -m pytest -q
~~~

To see a real Qwen-generated Tool Call, use the smallest model-backed lesson:

~~~bash
PYTHONPATH=src python examples/02_real_qwen_tool_calling.py \
  --model /path/to/Qwen3-1.7B
~~~

For a real local ReAct episode, install the model extras and provide a local normalized HotpotQA JSONL file and a local Qwen checkpoint:

~~~bash
pip install -e ".[test,data,hf]"
PYTHONPATH=src python examples/03_react_hotpot.py \
  --data /path/to/hotpotqa_distractor_validation.jsonl \
  --model /path/to/Qwen3-1.7B \
  --limit 1
~~~

This command is intentionally bounded. The example reports the question, answer metrics, turns, and attempted/valid/executed searches. Use scripts/evaluate_react.py for a stored baseline with trajectories and metrics.

## Research Track

The research path preserves the real implementation:

**Strict Hotpot-MT → Qwen3-8B → vanilla multi-turn GRPO → tool-use analysis → cost-aware RL → Natural Bridge-Hard evaluation**

It uses verl, vLLM, FSDP, native multi-turn rollout, deterministic BM25 retrieval, and explicit executed-search accounting. Start with the [research index](research/README.md), [experiment history](PROGRESS.md), and [environment report](docs/environment_report.md). Full training needs a compatible CUDA/verl installation and should only be launched after checking resources and selecting a unique output directory.

## Evidence so far (not final results)

These rows come from different evaluation purposes and must not be read as one leaderboard:

| Evaluation | EM | F1 | Executed-search evidence |
| --- | ---: | ---: | --- |
| Held-out ReAct baseline, 60 examples | 0.400 | 0.506 | 1.000 average |
| Strict Qwen3-8B pilot, 200 examples | 0.215 | 0.3344 | 1.345 average; P(search ≥ 2) = 31.5% |
| Formal strict vanilla GRPO | TBD | TBD | Active run; final gate open |

The strict pilot establishes a behavior diagnostic, not a causal GRPO result. Exact provenance and evaluation distinctions are maintained in the [research index](research/README.md).

## A lesson from the project: multi-turn is not automatic

The controlled Hotpot-MT environment uses bridge questions, top-k 1 retrieval, bounded observations, and a three-search budget. Earlier top-k 3 retrieval often exposed enough evidence in one call. In the fixed Qwen3-8B strict pilot, P(search >= 2) = 31.5%; exactly-two-search episodes had 52.63% EM versus 9.49% after one search. This is an observed pilot association, not causal proof or a finished GRPO result.

The **Strict Hotpot-MT** set is a controlled multi-turn stress test, not an unmodified standard HotpotQA benchmark. **Natural Bridge-Hard** is the less-filtered secondary evaluation; both are described in the research index.

## Tool-use metrics

The project separates:

attempted → emitted tool-call openings; valid → parseable calls; executed → calls accepted by the environment; useful → searches that add a supporting title; wasted → executed searches that add no new supporting title.

This makes “use fewer tools” testable without treating necessary evidence gathering as waste. Cost-aware reward remains planned until the active vanilla gate is accepted.

## Research status

M0–M3 are accepted. The formal strict Qwen3-8B vanilla GRPO run is active, so final M4 and all cost-aware/M5 claims remain TBD. The original canonical 2,000-example run and bounded strict pilots are archived as evidence; see [PROGRESS.md](PROGRESS.md) for exact status and numbers.

## Hardware and environments

These are validated tiers, not universal minimums:

| Tier | Requirement | Purpose |
| --- | --- | --- |
| Concepts / parser / search | CPU and Python 3.10+ | Examples 01, 02, and 04 |
| Local ReAct | Local Qwen3-1.7B checkpoint and CUDA-capable memory as available | One bounded inference-only episode or baseline |
| Full research | Four RTX A6000-class GPUs were used for the active Qwen3-8B strict run | Native verl + vLLM GRPO |

The reference environment is recorded in [docs/environment_report.md](docs/environment_report.md). Inspect GPU ownership, memory, disk, and process ownership before every long run; large models, checkpoints, data, and logs belong outside the Git checkout.

## Repository structure

~~~text
src/efficienttool_rl/   reusable agent, protocol, tools, data, rewards
examples/                bounded learning entry points
tutorials/               concise hands-on explanations
research/                research question, evidence, and experiment index
configs/                 reproducible training/tool configurations
scripts/                 data preparation, evaluation, training, analysis
tests/                   deterministic unit and integration tests
docs/                    environment, milestone, and debugging records
analysis/                failure analysis and diagnostics
~~~

## Contributing and credits

Contributor workflow and research-safety rules are documented in [AGENTS.md](AGENTS.md). The project builds on Qwen, HotpotQA, Hugging Face Transformers/datasets, verl, and vLLM. Preserve their licenses and cite them when redistributing results. This checkout does not currently add a top-level project license; resolve that before publishing derived artifacts.
