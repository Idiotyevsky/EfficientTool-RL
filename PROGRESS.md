# ToolAgentLab — Experiment Status

## Current Focus

Recipe-level comparison and diagnosis for a Qwen3-8B multi-turn search agent.

```text
Qwen3-8B Base
  → Vanilla GRPO, task reward                [completed and evaluated]
  → GRPO, process-aware composite reward    [experiment in progress]
  → Fresh DAPO, task reward                 [completed and evaluated]
  → Corrected assistant-only DAPO           [experiment in progress]
```

All methods use the same strict multi-turn BM25 environment for training and
the same Natural Bridge-Hard 200-example held-out protocol. Public conclusions
remain recipe-level unless a dedicated ablation isolates one component.

## Completed Evidence

### Base and Vanilla GRPO

Natural Bridge-Hard, 200 examples:

| Method | EM | F1 | Searches | Multi-search | Useful | Wasted |
|---|---:|---:|---:|---:|---:|---:|
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

### Reward and Infrastructure Validation

- Process-aware reward implemented as
  `0.8 answer + 0.15 evidence coverage + 0.05 format`.
- Offline replay confirms non-zero evidence variance, positive correlation
  with exact match, and answer-dominant reward mass.
- Project-local assistant-only DAPO reward manager uses the response mask for
  generated-token length and logs assistant length, total trajectory length,
  penalty, and search count.
- 91 unit/integration tests passed at the last full validation.

## Experiments in Progress

- GRPO with process-aware composite reward.
- Corrected DAPO with assistant-only overlong accounting.

These runs are not yet represented as completed results. Do not attribute the
Fresh DAPO collapse to overlong shaping until the corrected ablation is
finished and evaluated.

## Next Actions

1. Complete and evaluate the composite-reward GRPO checkpoint.
2. Complete and evaluate corrected assistant-only DAPO.
3. Compare task quality, protocol reliability, search distribution,
   useful/wasted retrieval, and generated length.
4. Update `experiments/results.md` only from stored evaluation artifacts.
5. Decide whether a component-level DAPO ablation is justified.

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
