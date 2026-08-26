# M4 Vanilla GRPO Plan

## Question

Does task-only multi-turn GRPO improve held-out HotpotQA performance or
measurably change search behavior relative to the frozen ReAct baseline?

## Stage 1

- Model: Qwen3-1.7B.
- Training data: fingerprinted 500-example train subset.
- Validation: separate 100-example validation slice; final reporting remains
  on the held-out ReAct evaluation slice.
- Objective: `0.5 EM + 0.5 token F1`; no search or token penalty.
- Rollouts: group size 4, temperature 0.8, maximum five assistant turns,
  maximum 1024 generated response tokens.
- Optimization: one epoch, 32 prompts per update, 16 updates.

Run with:

```bash
python scripts/run_ppo_m3.py --config-name qwen1.7b_grpo
```

The launcher must be run with the project environment, offline model/cache
paths, a short `RAY_TMPDIR`, and a confirmed free GPU. Each run uses a unique
output directory and retains rollout/validation dumps plus the resolved Hydra
configuration.

## Gate

Advance to 2,000 prompts only if the 500-example stage has complete logs,
non-degenerate group variance, valid held-out evaluation, and interpretable
changes in EM/F1, search calls, turns, and response length. A negative but
reproducible result is acceptable; cost shaping remains blocked until vanilla
GRPO is understood.
