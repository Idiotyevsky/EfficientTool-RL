# Composite Reward Pilot — Offline Distribution Validation

The process-aware composite reward
(`R = 0.8 * answer + 0.15 * evidence + 0.05 * format`) was validated offline
against stored fixed-policy trajectories before any training run. The script
is `scripts/validate_composite_reward.py`; the raw reports live next to this
file. No training configuration was changed by this validation.

## Components

- `answer`: existing task-only `0.5 * EM + 0.5 * F1` (`rewards/task.py`).
- `evidence`: document-level supporting-evidence coverage
  `|retrieved gold titles| / |gold supporting titles|` over successful search
  responses. Saturating and duplicate-insensitive; gold titles come only from
  evaluation-only `extra_info.supporting_titles`.
- `format`: mean of three binary checks — valid single terminal answer, no
  malformed/unknown tool calls, trajectory ends with the answer.

## Evidence

### Base Qwen3-8B (strict bridge-hard 200, one rollout each)

- answer mean 0.2747 (std 0.3986); evidence coverage mean 0.5575 (std 0.2407);
  format mean 0.9483 (std 0.1829); total score mean 0.3508 (std 0.3418).
- corr(evidence coverage, EM) = **0.532**; corr(searches, EM) = 0.345.
- aggregate share of total: answer 0.627, evidence 0.238.
- evidence reward on EM==0 rows: mean 0.490 — wrong answers that retrieved
  gold evidence still receive process credit, capped at 0.15 + 0.05.

### Vanilla GRPO Step-62 (strict train 2000 x 4 rollouts)

- answer mean 0.6597 (std 0.4488) — matches the stored fixed-policy
  EM 0.6255 / F1 0.6939 (0.5 * EM + 0.5 * F1 = 0.6597), confirming the replay.
- evidence coverage mean 0.8100 (std 0.2892); format mean 0.9872 (std 0.0937);
  total score mean 0.6986 (std 0.3895).
- corr(evidence coverage, EM) = **0.585**; corr(searches, EM) ≈ 0.
- aggregate share of total: answer 0.755, evidence 0.174 — the answer
  component dominates as intended.
- executed searches mean 2.0185 (max 3, budget-limited).

## Interpretation

1. The evidence component carries real variance (std ~0.24–0.29) and is
   positively correlated with task success on both policies, so it adds a
   usable gradient signal rather than a constant offset.
2. The format component is near ceiling for both policies (mean >= 0.948);
   with a 0.05 weight it stays auxiliary and cannot dominate training.
3. Evidence credit on wrong-answer rows (mean ~0.49–0.59 coverage) is the
   intended process supervision; its contribution is bounded by the 0.15
   weight. If future runs show answer under-optimization, lower
   `evidence_weight` first.
4. Saturating coverage prevents reward from increasing by repeating searches;
   `useful/wasted` counters are logged alongside for reward-hacking checks.

The pilot weights 0.8 / 0.15 / 0.05 are kept as the default training
configuration (`configs/grpo/qwen8b_hotpot_mt_strict_composite.yaml`).
