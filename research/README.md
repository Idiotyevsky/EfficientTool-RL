# Research Track

MiniAgentRL’s research track asks:

> Can GRPO teach an LLM tool agent to solve multi-hop tasks while reducing unnecessary tool use?

This track documents the implementation used for Qwen3, native multi-turn verl rollouts, vLLM, FSDP, deterministic BM25 search, HotpotQA-derived environments, trajectory logging, and held-out evaluation.

## Current experimental line

1. **Strict Hotpot-MT** — a controlled multi-turn stress-test environment.
2. **Qwen3-8B** — the current research model.
3. **Vanilla task-only GRPO** — validate the learning signal before shaping.
4. **Tool-use analysis** — compare attempted, valid, executed, useful, and wasted calls.
5. **Cost-aware RL** — planned after the vanilla baseline is finalized.
6. **Natural Bridge-Hard** — a less-filtered secondary evaluation.

The Qwen3-8B vanilla GRPO baseline comparison is complete on the matched Natural Bridge-Hard evaluation. Cost-aware training is the next stage and has no reported result yet.

## Evaluation sets

### Hotpot-MT Strict

This is not an unmodified HotpotQA benchmark. It is a deliberately controlled stress test using bridge-focused candidates, top-k 1 retrieval, bounded observations, and a three-executed-search budget. The strict candidate construction also uses a question-level information-availability filter. That filter is useful for studying behavior, but it must be disclosed when reporting results.

### Natural Bridge-Hard

This secondary set keeps official validation rows with type=bridge and level=hard without the strict answer-absence filter. It uses the same runtime limits as the strict environment, so the two evaluations are comparable while answering different questions.

The fixed 200-row secondary artifact is kept outside this Git checkout:

~~~text
verl_hotpotqa_mt_natural_bridge_hard_val_200.parquet
~~~

SHA-256: 1835707b46734751610d42a6f5ebba8bb3098789f841fede1c88a63b3cbf5fdc

## Latest vanilla baseline

Natural Bridge-Hard · 200 examples · Qwen3-8B Base → Step 62

| Metric | Base | Step 62 |
| --- | ---: | ---: |
| EM | 32.5% | 51.5% |
| F1 | 42.03% | 62.53% |
| Completion | 93.5% | 97.5% |
| Invalid action | 10.06% | 0.17% |
| Executed search | 1.335 | 1.960 |
| Multi-search rate | 31.5% | 86.0% |
| Useful search | 0.965 | 1.445 |
| Wasted search | 0.370 | 0.515 |
| Tool efficiency | 72.28% | 73.72% |

The result is a vanilla baseline observation, not a cost-aware result: task quality and multi-step retrieval improved, while the absolute number of wasted calls also increased.

## Evidence so far

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Held-out ReAct baseline, 60 examples | EM 0.400, F1 0.506, executed searches 1.000 | Reference baseline |
| Strict Qwen3-8B pilot, 200 examples | EM 0.215, F1 0.3344, average executed searches 1.345, P(search ≥ 2) 0.315 | Pilot behavior evidence |
| Strict 8B pilot, exactly two vs one search | EM 0.5263 vs 0.0949 | Observed association, not causal proof |
| Strict 8B GRPO sanity | Validation EM/F1 0.240/0.3635 → 0.320/0.4324 in four updates | Small-scale training check |
| Natural Bridge-Hard vanilla GRPO, 200 examples | EM 0.325 → 0.515, F1 0.4203 → 0.6253, multi-search 0.315 → 0.860 | Stronger task quality and more active multi-step retrieval |

The completed rows come from stored reports. Pilot and sanity results describe system behavior; the vanilla comparison now has a matched Natural Bridge-Hard result, while cost-aware training has not started.

## Reproducibility map

- [Experiment status and history](../PROGRESS.md)
- [Environment and versions](../docs/environment_report.md)
- [Debug history](../docs/debug_log.md)
- [Strict training config](../configs/qwen8b_grpo_hotpot_mt_strict.yaml)
- [Strict tool config](../configs/tool_config/hotpot_multi_turn.yaml)
- [GRPO training entry](../scripts/run_ppo_m3.py)
- [Local ReAct evaluator](../scripts/evaluate_react.py)
- [Native rollout analyzer](../scripts/analyze_verl_rollouts.py)

Large models, normalized datasets, checkpoints, and run outputs intentionally live outside this checkout. Every report should record the commit, seed, split, config, framework versions, and artifact fingerprint.

## Next steps

Use the completed vanilla baseline to choose a cost-aware objective that preserves useful exploration while reducing wasted calls. Report task quality together with executed/useful/wasted search behavior, turns, tokens, and search-count distributions.
