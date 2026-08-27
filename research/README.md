# Research Track

MiniAgentRL’s research track asks:

> Can GRPO teach an LLM tool agent to solve multi-hop tasks while reducing unnecessary tool use?

The public learning layer is intentionally small. This track preserves the real system: Qwen3, native multi-turn verl rollouts, vLLM, FSDP, deterministic BM25 search, HotpotQA-derived environments, trajectory logging, and held-out evaluation.

## Current experimental line

1. **Strict Hotpot-MT** — a controlled multi-turn stress-test environment.
2. **Qwen3-8B** — the current formal research model.
3. **Vanilla task-only GRPO** — validate the learning signal before shaping.
4. **Tool-use analysis** — compare attempted, valid, executed, useful, and wasted calls.
5. **Cost-aware RL** — planned only after the vanilla gate is accepted.
6. **Natural Bridge-Hard** — a less-filtered secondary evaluation.

The active formal strict vanilla run is still in progress. Final M4 and cost-aware claims are therefore TBD.

## Evaluation sets

### Hotpot-MT Strict

This is not an unmodified HotpotQA benchmark. It is a deliberately controlled stress test using bridge-focused candidates, top-k 1 retrieval, bounded observations, and a three-executed-search budget. The strict candidate construction also uses a question-level information-availability filter. That filter is useful for studying behavior, but it must be disclosed when reporting results.

### Natural Bridge-Hard

This secondary set keeps official validation rows with type=bridge and level=hard without the strict answer-absence filter. It uses the same runtime limits as the strict environment, so the two evaluations are comparable while answering different questions.

The fixed 200-row artifact is stored outside Git at:

~~~text
/home/zfs01/jiangjr/efficienttool-rl-data/verl_hotpotqa_mt_natural_bridge_hard_val_200.parquet
~~~

SHA-256: 1835707b46734751610d42a6f5ebba8bb3098789f841fede1c88a63b3cbf5fdc

## Evidence so far

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Held-out ReAct baseline, 60 examples | EM 0.400, F1 0.506, executed searches 1.000 | Stable M2 baseline |
| Strict Qwen3-8B pilot, 200 examples | EM 0.215, F1 0.3344, average executed searches 1.345, P(search ≥ 2) 0.315 | Multi-search behavior exists before RL |
| Strict 8B pilot, exactly two vs one search | EM 0.5263 vs 0.0949 | Observed association, not causal proof |
| Strict 8B GRPO sanity | Validation EM/F1 0.240/0.3635 → 0.320/0.4324 in four updates | Technical learning-signal evidence only |

These values come from stored reports and are not a substitute for the active formal run. No M5 improvement or efficiency claim has been accepted.

## Reproducibility map

- [Current status and gates](../PROGRESS.md)
- [Environment and validated versions](../docs/environment_report.md)
- [M2 data policy](../docs/m2_data_policy.md)
- [M3 sanity plan](../docs/m3_sanity_plan.md)
- [M4 experiment plan](../docs/m4_plan.md)
- [Debug history](../docs/debug_log.md)
- [Strict training config](../configs/qwen8b_grpo_hotpot_mt_strict.yaml)
- [Strict tool config](../configs/tool_config/hotpot_multi_turn.yaml)
- [Training entry point](../scripts/run_ppo_m3.py)
- [Local ReAct evaluator](../scripts/evaluate_react.py)
- [Native rollout analyzer](../scripts/analyze_verl_rollouts.py)

Large models, normalized datasets, checkpoints, and run outputs intentionally live outside this checkout. Every report should record the commit, seed, split, config, framework versions, and artifact fingerprint.

## Next gate

Let vanilla strict Qwen3-8B finish. Then compare the held-out checkpoint with the matched baseline using answer quality, executed/useful/wasted search behavior, turns, tokens, and search-count distributions. Only if that evidence is accepted should the project choose and run a cost-aware reward sweep.
