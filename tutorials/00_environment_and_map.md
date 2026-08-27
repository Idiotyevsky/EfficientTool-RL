# Chapter 00 — Environment and the Agent RL Map

## What you will learn

By the end of this chapter you should be able to:

1. tell which lessons need only Python and which need a model or GPU;
2. locate the reusable agent, tool, reward, training, and evaluation code;
3. run a CPU-only health check before installing the heavy stack.

## See the final effect first

From the repository root:

~~~bash
PYTHONPATH=src python examples/00_environment_check.py
~~~

A healthy installation looks like this:

~~~text
MiniAgentRL Learn Track environment check
Python:     3.12.x
Platform:   Linux-...
Core package: PASS (...)

Optional components:
  torch        available
  transformers available
  ...
Next step: run examples/01_tool_calling.py.
~~~

The exact optional-component list depends on your environment. The important checkpoint is Core package: PASS. This lesson does not load a model, contact the network, or start Ray.

## 1. The map

MiniAgentRL teaches one system in layers:

~~~text
Tool Calling
    ↓
Multi-turn Agent
    ↓
ReAct baseline
    ↓
Environment + Trajectory + Reward
    ↓
Grouped Rollouts
    ↓
GRPO policy update
    ↓
Efficient Tool Use analysis
~~~

The Learn Track uses tiny local examples where possible. The Research Track then reuses the same concepts with Qwen3, HotpotQA, verl, vLLM, and FSDP.

## 2. Install only what the next lesson needs

For Chapters 00, 01, 03, 05, 06, and 08:

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
~~~

For the real local Qwen lesson, add the model dependencies:

~~~bash
pip install -e ".[test,hf]"
~~~

The full Research Track additionally needs the compatible verl/Ray/vLLM environment described in [the environment report](../docs/environment_report.md). Do not install a random latest combination just to make a tutorial command shorter.

## 3. Where the pieces live

| Question | Start here |
| --- | --- |
| How is one action parsed? | [protocol.py](../src/efficienttool_rl/protocol.py) |
| How are turns and observations connected? | [agent.py](../src/efficienttool_rl/agent.py) |
| How does local search work? | [search.py](../src/efficienttool_rl/tools/search.py) |
| How is task reward calculated? | [task.py](../src/efficienttool_rl/rewards/task.py) |
| How are answers evaluated? | [metrics.py](../src/efficienttool_rl/evaluation/metrics.py) |
| How does native training start? | [run_ppo_m3.py](../scripts/run_ppo_m3.py) |

The examples are intentionally thin adapters around these modules. If an example and production code disagree, production code and its tests are the source of truth.

## Common problems

### The core package cannot be imported

Run commands from the repository root and keep PYTHONPATH=src in the command. An editable install also works, but the explicit path makes the import boundary visible.

### torch or verl is not installed

That is fine for the CPU lessons. The health check reports optional components instead of failing because a later track is not installed yet.

### Python is too old

The package requires Python 3.10 or newer. Use the interpreter printed by the health check when debugging an environment mismatch.

## Exercise

Run the health check with the system Python and with the virtual environment Python. Compare the executable and optional-component lines. Which lessons can you run in each environment?

## Checkpoint

Continue when you can answer:

- Which file parses a model action?
- Which object stores one complete episode?
- Which command checks the core package without loading a model?

## Next

[Chapter 01 — Tool Calling Basics](01_tool_calling_basics.md) turns one string into one real tool observation.
