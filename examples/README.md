# Runnable Examples

The examples are thin adapters around the real modules under src/efficienttool_rl/. They are arranged to match the Learn Track.

## CPU-only examples

~~~bash
PYTHONPATH=src python examples/00_environment_check.py
PYTHONPATH=src python examples/01_tool_calling.py
PYTHONPATH=src python examples/02_multiturn_agent.py
PYTHONPATH=src python examples/04_grpo_concepts.py
PYTHONPATH=src python examples/05_trajectory_reward.py
PYTHONPATH=src python examples/08_efficiency_metrics.py
~~~

These examples use deterministic inputs, real parsing/search/reward/analysis code, and no model download.

## Model-backed examples

~~~bash
PYTHONPATH=src python examples/02_real_qwen_tool_calling.py \
  --model /path/to/Qwen3-1.7B \
  --device cuda:0

PYTHONPATH=src python examples/03_react_hotpot.py \
  --data /path/to/hotpotqa_distractor_validation.jsonl \
  --model /path/to/Qwen3-1.7B \
  --limit 1
~~~

The first command is the smallest real-model Tool Calling lesson. The second runs a bounded ReAct episode on normalized HotpotQA.

## Training smoke

The real one-update GRPO entry point is [scripts/train_grpo_smoke.py](../scripts/train_grpo_smoke.py). Follow [Chapter 07](../tutorials/07_grpo_smoke.md) for path checks, a dry-run command, expected evidence, and resource safety.

## Important labels

The scripted policies in examples/02_multiturn_agent.py, examples/05_trajectory_reward.py, and examples/08_efficiency_metrics.py are teaching fixtures. They exercise production loop and analysis code but are not language-model predictions.
