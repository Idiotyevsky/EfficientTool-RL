# Results

Comparison of Qwen3-8B training recipes on the shared multi-turn tool-agent
pipeline (Hotpot-MT strict protocol, Natural Bridge-Hard held-out
evaluation). All numbers come from stored run artifacts; `TBD` marks runs
that have not completed. Never update this table without the corresponding
stored evidence.

## Held-out comparison (Natural Bridge-Hard, 200 examples)

| Method | Reward | EM | F1 | Completion | Invalid action | Executed searches | Multi-search |
|---|---|---:|---:|---:|---:|---:|---:|
| Base (no training) | — | 32.5% | 42.03% | 93.5% | 10.06% | 1.335 | 31.5% |
| Vanilla GRPO (task-only, step 62) | task-only | 51.5% | 62.53% | 97.5% | 0.17% | 1.960 | 86.0% |
| GRPO + composite reward | 0.8/0.15/0.05 | TBD | TBD | TBD | TBD | TBD | TBD |
| DAPO (task-only) | task-only | TBD | TBD | TBD | TBD | TBD | TBD |
| DAPO + composite reward | 0.8/0.15/0.05 | TBD | TBD | TBD | TBD | TBD | TBD |

## Composite reward offline validation (no training)

The composite reward was replayed offline over **stored fixed-policy
trajectories** (no training involved). Inputs: the Qwen3-8B Base trajectories
from the strict bridge-hard 200 run, and the vanilla GRPO step-62 checkpoint
trajectories (`fixed_step62_500x4.jsonl`, 500 prompts x 4 rollouts on the
strict train set, the M5.1 fixed-policy audit artifact); supporting titles
come from the corresponding data artifacts. Replay script:
`scripts/validate_composite_reward.py`; raw reports:
`analysis/composite_reward_pilot/*.json`.

From `analysis/composite_reward_pilot/` — the composite reward replayed over
stored fixed-policy trajectories before any training run:

| Policy stored | answer mean | evidence coverage mean | format mean | corr(evidence, EM) | aggregate answer share |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Base | 0.2747 | 0.5575 | 0.9483 | 0.532 | 0.627 |
| GRPO step 62 | 0.6597 | 0.8100 | 0.9872 | 0.585 | 0.755 |

Read: the evidence component carries real variance and correlates positively
with task success; the format component stays near ceiling (auxiliary only);
the answer component keeps the largest share of total reward mass.

## Training diagnostics to be collected per run

Recorded from rollout dumps + trainer metrics for every completed run:

- reward mean/std and per-component means (`answer/evidence/format`)
- advantage mean/std; zero-variance (all-same-reward) group ratio
- DAPO dynamic sampling: effective prompt utilization per optimizer step
  (kept groups / generated groups), generation batches per step
- response length distribution and truncation (overlong) rate
- invalid action rate, multi-search rate, duplicate-query rate
- actor grad norm, entropy, KL (when the config enables KL loss)
