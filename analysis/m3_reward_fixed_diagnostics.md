# M3 Reward-Fixed Diagnostics

Run: `m3_sanity_a6000_6_g2_reward_fix` on 2026-08-26, Qwen3-1.7B,
task-only `0.5 EM + 0.5 token F1`, seed 42, eight training prompts,
group size four, and a 1024-token response cap.

## Training Rollouts

| Metric | Value |
|---|---:|
| Rollouts | 32 |
| Mean reward | 0.03125 |
| Positive rewards | 1 / 32 |
| Valid-answer rate | 6 / 32 |
| Groups with non-zero reward variance | 1 / 8 |
| Zero-variance group ratio | 0.875 |
| Mean group reward variance | 0.0234375 |
| Distinct trajectories per group | 3, 3, 4, 4, 4, 4, 4, 4 |

Groups were reconstructed from the native dump's `input` field; JSONL row
order is not a valid grouping assumption.

## Update and Validation

- Advantage std: `0.870591`; range: `-0.499999` to `1.499997`.
- `grad_norm`: `2.658426`; actor parameter deltas were verified against the
  original Qwen checkpoint (layer-0 q projection mean absolute delta
  `9.87e-7`).
- Validation task score: `0.0` before and after the step.
- Validation valid-answer rate: `0.0` → `0.25`; five of eight outputs changed.

## Decision

The technical M3 learning-signal gate passes. This run does not support a
task-improvement claim; M4 must measure vanilla GRPO at larger scale.
