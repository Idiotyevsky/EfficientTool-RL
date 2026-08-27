# Chapter 07 — Run a Real GRPO Smoke

## What you will learn

You will:

1. launch one real GRPO update through the repository entry point;
2. identify rollout, reward, gradient, and checkpoint evidence;
3. keep a teaching smoke separate from formal research runs.

This is the first lesson that can allocate a GPU and start Ray. Read the command before running it.

## See the final effect first

The Learn Track smoke uses eight prompts and four rollouts per prompt for one update. It reuses the real task-only reward, native multi-turn tool loop, and verl trainer. It is not a performance experiment.

First print the resolved command without starting Ray:

~~~bash
PYTHONPATH=src python scripts/train_grpo_smoke.py \
  --model /path/to/Qwen3-1.7B \
  --data-dir /path/to/efficienttool-rl-data \
  --run-dir /path/to/efficienttool-rl-runs/learn_smoke_01 \
  --verl-config-path /path/to/verl/verl/trainer/config \
  --run-name learn_smoke_01 \
  --dry-run
~~~

If the paths exist and the run directory is empty, remove --dry-run to launch:

~~~bash
PYTHONPATH=src python scripts/train_grpo_smoke.py \
  --model /path/to/Qwen3-1.7B \
  --data-dir /path/to/efficienttool-rl-data \
  --run-dir /path/to/efficienttool-rl-runs/learn_smoke_01 \
  --verl-config-path /path/to/verl/verl/trainer/config \
  --run-name learn_smoke_01
~~~

The wrapper refuses a non-empty run directory and never reuses the formal output path. It also validates that the Learn Track data files exist:

~~~text
verl_hotpotqa_train_500.parquet
verl_hotpotqa_val_100.parquet
~~~

## 1. What the wrapper actually does

scripts/train_grpo_smoke.py does not implement another trainer. It:

1. resolves model, data, run, and verl-config paths;
2. exports the ETRL variables expected by Hydra;
3. selects configs/learn_grpo_smoke.yaml;
4. overrides only the unique experiment and output paths;
5. calls scripts/run_ppo_m3.py as a subprocess.

The smoke config inherits the validated task-only M3 setup and changes only the teaching scale: eight training prompts, eight validation prompts, one update, one GPU by default, and group size four.

## 2. What should you watch in the logs?

Names vary by installed verl version, but look for evidence at each boundary:

~~~text
validation before training
rollout generation
mean reward / reward std
group variance or zero-variance ratio
actor loss / KL statistics
non-zero gradient or optimizer step
validation after training
checkpoint or model output
~~~

A process that starts is not automatically a successful smoke. Save stdout and the run directory, then check that rollouts, metrics, and a checkpoint or optimizer update were actually produced.

Reference output from a checked Qwen3-1.7B run was:

~~~text
mean reward = 0.15625    actor/grad_norm = 4.333    global_step = 1
validation valid-answer rate = 0.25    validation EM/F1 = 0/0
~~~
These numbers demonstrate a real learning signal, not task improvement.

## 3. Resource safety

Before a real run:

~~~bash
nvidia-smi
df -h
~~~

Use a fresh output directory, a persistent shell when needed, and a model that fits the selected GPU. Do not run this tutorial against the active formal 8B directory. The smoke is intentionally small, but rollout generation can still take minutes and model loading can use substantial memory.

## Common problems

### The wrapper says data files are missing

Prepare or locate the normalized 500-row parquet files. The wrapper refuses to guess a different dataset because changing the input changes the lesson.

### Hydra cannot find ppo_trainer

Pass the installed verl config directory through --verl-config-path. This is the directory that contains the trainer YAMLs, not the repository root.

### Ray or CUDA fails before the update

Capture the exact error, check process ownership and GPU availability, then retry only after a bounded diagnosis. Do not modify the formal configuration to fix a teaching smoke.

### vLLM reports that ninja is missing

The ninja package may be installed while the detached shell cannot find its executable. This wrapper prepends the selected Python interpreter’s bin directory to PATH. If you invoke the lower-level entry point yourself, activate the same environment or export its bin directory first:

~~~bash
export PATH=/path/to/your/python-environment/bin:$PATH
~~~

Then rerun a fresh smoke directory. Do not install a second copy blindly.

### The run finishes with no learning signal

Inspect response formatting, reward extraction, group variance, and validation. One update can verify plumbing; it cannot establish task improvement.

## Exercises

1. Run --dry-run and read every path in the printed command.
2. After a successful smoke, inspect the first rollout JSONL file and find one final answer.
3. Compare the configured group size with the number of rows per prompt.
4. Change only the run name and run directory for a second smoke; never overwrite the first.

## Checkpoint

You are done when you can show evidence for:

- four grouped rollouts per prompt;
- a task reward computation;
- an actor update or non-zero gradient;
- post-update validation;
- a preserved output directory.

## Next

[Chapter 08 — Efficient Tool Use](08_efficient_tool_use.md) measures whether executed calls were useful or wasteful.
