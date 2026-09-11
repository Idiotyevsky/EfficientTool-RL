# ToolAgentLab Experimental Results

All completed rows below come from stored trajectory artifacts. Every policy is
evaluated on the same Natural Bridge-Hard set: 200 official HotpotQA validation
rows with `type=bridge`, `level=hard`, and no strict candidate filter.

## Held-out Comparison

| Method | Training reward | EM | F1 | Completion | Invalid action | Searches | Multi-search | Useful | Wasted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B Base | — | 32.5% | 42.03% | 93.5% | 10.06% | 1.335 | 31.5% | 0.965 | 0.370 |
| Vanilla GRPO, step 62 | task-only | 51.5% | 62.53% | 97.5% | 0.17% | 1.960 | 86.0% | 1.445 | 0.515 |
| Fresh DAPO, step 62 | task-only | 33.0% | 41.83% | 100% | 0% | 1.100 | 10.0% | 0.880 | 0.220 |
| GRPO + composite reward | 0.8 / 0.15 / 0.05 | in progress | in progress | in progress | in progress | in progress | in progress | in progress | in progress |
| Corrected assistant-only DAPO | task + overlong | in progress | in progress | in progress | in progress | in progress | in progress | in progress | in progress |

Fresh DAPO was re-aggregated from 200 rows in the stored artifact family
dapo_fresh62_nbh200/shard_{0,1,2}. Its average task reward is 0.3742,
average turns 2.10, and useful/executed ratio 80.0%. Its lower absolute tool cost is not an improvement:
EM regresses 18.5 percentage points relative to vanilla GRPO.

## Interpretation

- Base tends to stop after one search and under-retrieves for many bridge
  questions.
- Vanilla GRPO substantially improves task quality and moves the policy toward
  multi-step retrieval. Both useful and wasted calls increase.
- Fresh DAPO is protocol-stable but one-search dominated. Its completed run did
  not apply effective overlong shaping because async prefilled scores bypassed
  that path.
- The corrected assistant-only DAPO experiment is needed before drawing a
  causal conclusion about overlong shaping.

## Composite Reward Offline Audit

The process-aware reward was replayed over stored fixed-policy trajectories
before training:

| Stored policy | Answer mean | Evidence mean | Format mean | corr(evidence, EM) | Answer share |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Base | 0.2747 | 0.5575 | 0.9483 | 0.532 | 0.627 |
| Vanilla GRPO step 62 | 0.6597 | 0.8100 | 0.9872 | 0.585 | 0.755 |

Evidence coverage carries non-zero signal; format is near ceiling and remains
auxiliary; answer reward retains the largest aggregate contribution. Raw
reports and methodology are in
[the composite reward pilot](../analysis/composite_reward_pilot/README.md).

## Metrics Required for New Rows

Each completed run must report:

- EM, token F1, completion, and invalid-action rate;
- attempted, valid, executed, useful, and wasted tool calls;
- turns, generated length, and search-count distribution;
- reward/component distributions and zero-variance group ratio;
- gradient norm, entropy, KL where enabled;
- config, seed, commit, dataset fingerprint, and artifact provenance.
