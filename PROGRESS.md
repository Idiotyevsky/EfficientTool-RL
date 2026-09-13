# SearchAgent-RL — Experiment Status

## Current Focus

Recipe-level comparison and diagnosis for a Qwen3-8B multi-turn search agent.

```text
Qwen3-8B Base
  → Vanilla GRPO, task reward                [completed and evaluated]
  → GRPO, process-aware composite reward    [completed and evaluated]
  → GRPO, Reward v2                         [completed and evaluated]
```

Auxiliary study: Fresh DAPO is completed and evaluated; the corrected
assistant-only DAPO experiment remains pending.

All methods use the same strict multi-turn BM25 environment for training and
the same Natural Bridge-Hard 200-example held-out protocol. Public conclusions
remain recipe-level unless a dedicated ablation isolates one component.

## Completed Evidence

### Base and Vanilla GRPO

Natural Bridge-Hard, 200 examples:

| Method | EM | F1 | Searches | Multi-search | Useful | Wasted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 32.5% | 42.03% | 1.335 | 31.5% | 0.965 | 0.370 |
| Vanilla GRPO | 51.5% | 62.53% | 1.960 | 86.0% | 1.445 | 0.515 |

Vanilla GRPO improves answer quality and strengthens multi-step retrieval.
Both useful and wasted searches increase.

### Fresh DAPO

Natural Bridge-Hard, 200 examples:

- EM: 33.0%
- F1: 41.83%
- completion: 100%
- invalid action rate: 0%
- searches: 1.100
- multi-search: 10.0%
- useful/wasted: 0.880 / 0.220
- useful/executed: 80.0%
- average task reward: 0.3742
- average turns: 2.10

Fresh DAPO is one-search dominated and regresses task quality relative to
vanilla GRPO. The completed run did not exercise effective overlong shaping:
async prefilled reward scores bypassed that term.

### Process-aware Composite GRPO

Natural Bridge-Hard, 200 examples:

- EM: 45.0%
- F1: 55.25%
- completion: 98.0%
- invalid action rate: 0.52%
- searches: 1.885
- multi-search: 77.5%
- useful/wasted: 1.250 / 0.635
- useful/executed: 66.31%
- average task reward: 0.5012
- average turns: 2.90

The composite-reward policy remains substantially better than Base on answer
quality, but it does not outperform task-only GRPO. Relative to task-only
GRPO, it retrieves less useful evidence and performs more wasted searches.
This result does not support the current composite reward as an improvement.

### Reward v2 GRPO

Reward v2 uses the trajectory-level objective
`answer + beta(step) * marginal evidence - 0.02 * answer * wasted`, with beta
linearly annealed from 0.10 to 0 over 62 optimizer updates. Natural Bridge-Hard,
200 examples:

- EM: 45.5%
- F1: 56.83%
- completion: 99.0%
- invalid action rate: 0.37%
- searches: 1.675
- multi-search: 64.0%
- useful/wasted: 1.250 / 0.425
- useful/executed: 74.63%
- average task reward: 0.5117
- average turns: 2.685
- average generated length: 45.59 tokens
- search distribution: 72 one-search / 121 two-search / 7 three-search

Reward v2 is stronger than Composite v1 while using fewer searches: useful
retrieval is preserved and wasted retrieval decreases. It still underperforms
task-only GRPO and retrieves less useful evidence, so it is not accepted as a
Pareto improvement over the task-only baseline.

### Reward and Infrastructure Validation

- Process-aware reward implemented as
  `0.8 answer + 0.15 evidence coverage + 0.05 format`.
- Offline replay confirms non-zero evidence variance, positive correlation
  with exact match, and answer-dominant reward mass.
- Project-local assistant-only DAPO reward manager uses the response mask for
  generated-token length and logs assistant length, total trajectory length,
  penalty, and search count.
- 108 unit/integration tests passed in the latest full validation.

## Pending Experiment

- Corrected DAPO with assistant-only overlong accounting.

Do not attribute the Fresh DAPO collapse to overlong shaping until the
corrected ablation is finished and evaluated.

## Next Actions

1. Compare Reward v2 and task-only GRPO failures to identify where reduced
   retrieval changes answer quality.
2. Complete and evaluate corrected assistant-only DAPO when resources permit.
3. Compare task quality, protocol reliability, search distribution,
   useful/wasted retrieval, and generated length.
4. Decide whether a revised process reward or component-level DAPO ablation is
   justified.

## Stable References

- Main narrative: [README.md](README.md)
- Results: [experiments/results.md](experiments/results.md)
- Baselines: [experiments/baselines.md](experiments/baselines.md)
- Composite reward audit:
  [analysis/composite_reward_pilot/README.md](analysis/composite_reward_pilot/README.md)
- DAPO diagnosis:
  [analysis/dapo_diagnostics/README.md](analysis/dapo_diagnostics/README.md)
- Engineering audit: [docs/debug_log.md](docs/debug_log.md)
- Historical milestone records: [docs/archive/](docs/archive/)
