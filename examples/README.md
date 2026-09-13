# SearchAgent-RL Runnable Examples

These examples are thin entry points around modules under
`src/efficienttool_rl/`. They exercise the same parser, search tool, episode
loop, reward, and analysis code used by training and evaluation.

## CPU Examples

```bash
PYTHONPATH=src python examples/00_environment_check.py
PYTHONPATH=src python examples/01_tool_calling.py
PYTHONPATH=src python examples/02_multiturn_agent.py
PYTHONPATH=src python examples/04_grpo_concepts.py
PYTHONPATH=src python examples/05_trajectory_reward.py
PYTHONPATH=src python examples/08_efficiency_metrics.py
```

These use deterministic inputs and require no model download.

## Model-backed Examples

```bash
PYTHONPATH=src python examples/02_real_qwen_tool_calling.py \
  --model /path/to/Qwen3-1.7B \
  --device cuda:0

PYTHONPATH=src python examples/03_react_hotpot.py \
  --data /path/to/hotpotqa_distractor_validation.jsonl \
  --model /path/to/Qwen3-1.7B \
  --limit 1
```

The first command tests model-generated actions. The second runs a bounded
HotpotQA episode.

## GRPO Smoke

Use [`scripts/train_grpo.py`](../scripts/train_grpo.py) with
`configs/grpo/qwen1.7b_smoke.yaml` for a real one-update verl/vLLM smoke.

The scripted policies in the CPU examples are explicitly teaching fixtures.
They exercise core code paths but are not model predictions.
