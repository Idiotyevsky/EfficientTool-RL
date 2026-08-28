# M5.0 Offline Cost-Reward Design

Status: complete as an offline reward-design study. No cost-aware policy was
trained in this step, and this analysis does not claim a performance gain.

## Objective

The candidate reward is:

    R_CA = R_task - lambda * R_task * N_waste
    R_task = 0.5 * EM + 0.5 * F1

Only wasted executed searches are charged. Useful searches are not directly
penalized, and a zero-task-reward trajectory receives zero cost penalty.

## Inputs

The analysis re-scored the stored 200-example Natural Bridge-Hard evaluation
for both the base model and the vanilla GRPO Step 62 checkpoint. The evaluation
artifact is hotpotqa_mt_natural_bridge_hard_val_200.jsonl; its parquet
fingerprint is:

    1835707b46734751610d42a6f5ebba8bb3098789f841fede1c88a63b3cbf5fdc

Evaluated lambdas:

    0, 0.025, 0.05, 0.10, 0.20, 0.30

## Stored behavior

| Source | Task reward | Executed | Useful | Wasted | Multi-search |
|---|---:|---:|---:|---:|---:|
| Base | 0.3727 | 1.3350 | 0.9650 | 0.3700 | 31.5% |
| Step 62 | 0.5701 | 1.9600 | 1.4450 | 0.5150 | 86.0% |

The counterfactual mean penalty at lambda=0.10 is 0.0043 for Base and
0.0125 for Step 62. At lambda=0.30 it is 0.0128 and 0.0375 respectively.
Task quality itself is unchanged by offline re-scoring.

## Ranking diagnostics

| Lambda | Base task-order inversions | Step 62 task-order inversions | Correct/wrong inversions |
|---:|---:|---:|---:|
| 0.00 | 0 / 12,300 | 0 / 12,779 | 0 / 0 |
| 0.025 | 0 / 12,300 | 0 / 12,779 | 0 / 0 |
| 0.05 | 0 / 12,300 | 0 / 12,779 | 0 / 0 |
| 0.10 | 0 / 12,300 | 3 / 12,779 | 0 / 0 |
| 0.20 | 5 / 12,300 | 23 / 12,779 | 0 / 0 |
| 0.30 | 14 / 12,300 | 35 / 12,779 | 3 / 2 |

The general task-order count includes benign reordering among partial-quality
trajectories. The stricter correct-versus-wrong check shows that lambda values
through 0.20 did not reverse a correct trajectory below an incorrect one in
these stored rows. At 0.30, a few high-waste correct rows were overtaken by
lower-quality zero-waste rows, so 0.30 is not a preferred first training value.

Stored-pair checks also passed for:

- equal task quality with zero waste: remains tied;
- correct multi-search versus wrong zero-search: Base artifact;
- partial task signal with waste versus wrong zero-search: Base artifact;
- correct useful searches versus an otherwise equal correct trajectory with
  added waste: Step 62 artifact.

Unavailable pairs were reported as unavailable rather than synthesized.

## Decision

The formula passes the offline anti-under-search and waste-isolation checks.
Use lambda values 0.025, 0.05, and 0.10 for the first bounded 8-by-4
cost-aware smoke. Treat 0.20 as a stress value and leave 0.30 out of the first
main comparison.

This is still a reward-design result, not a learned-policy result. The next
step is to implement the objective in an isolated reward module, verify that
waste metadata reaches the reward, and run the bounded smoke while preserving
the vanilla task-only reward unchanged.

## Reproduction

Set DATA, BASE, and STEP62 to the external artifact locations, then run:

    PYTHONPATH=src:. python scripts/analyze_cost_reward.py \
      --examples "$DATA" \
      --base-trajectories "$BASE" \
      --step62-trajectories "$STEP62" \
      --output-dir outputs/m5_cost_reward_offline

The command writes JSON, JSONL, CSV, and Markdown diagnostics under the
ignored outputs/m5_cost_reward_offline/ directory.
