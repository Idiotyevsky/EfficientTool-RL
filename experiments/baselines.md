# Baselines

All numbers below are reproduced from stored experiment artifacts; none are
extrapolated. Artifacts live outside the Git checkout under
`/home/zfs01/jiangjr/efficienttool-rl-{runs,data}` (recorded in `PROGRESS.md`).

## Evaluation sets

Two fixed HotpotQA-derived evaluation sets share identical runtime limits
(top-k=1 retrieval, 384-token observations, three-executed-search budget,
five assistant turns), so results are directly comparable:

- **Hotpot-MT Strict (train 2000 / val 100)** — controlled stress test:
  bridge-focused candidates, top-k=1, bounded observations, a
  question-level information-availability filter. Useful for studying
  behavior; the filter must be disclosed when reporting.
- **Natural Bridge-Hard (200)**: official validation rows with
  `type=bridge`, `level=hard`, no question-level filtering. The primary
  held-out comparison set. Parquet artifact
  `verl_hotpotqa_mt_natural_bridge_hard_val_200.parquet`, SHA-256
  `1835707b46734751610d42a6f5ebba8bb3098789f841fede1c88a63b3cbf5fdc`.

Large models, normalized datasets, checkpoints, and run outputs live outside
the Git checkout; every report records commit, seed, split, config, framework
versions, and artifact fingerprints.

## Primary held-out evaluation: Natural Bridge-Hard (200 examples)

Secondary HotpotQA evaluation: official validation rows with `type=bridge`,
`level=hard`, no question-level filtering. Fixed artifact
`verl_hotpotqa_mt_natural_bridge_hard_val_200.parquet`,
SHA-256 `1835707b46734751610d42a6f5ebba8bb3098789f841fede1c88a63b3cbf5fdc`.
Runtime limits identical to the strict environment: top-k=1 retrieval,
384-token observations, three executed-search budget, five assistant turns.

| Metric | Qwen3-8B Base | + Vanilla GRPO (step 62) |
|---|---:|---:|
| EM | 32.5% | 51.5% |
| F1 | 42.03% | 62.53% |
| Completion rate | 93.5% | 97.5% |
| Invalid action rate | 10.06% | 0.17% |
| Executed searches / episode | 1.335 | 1.960 |
| Multi-search rate | 31.5% | 86.0% |
| Useful searches / episode | 0.965 | 1.445 |
| Wasted searches / episode | 0.370 | 0.515 |
| Tool efficiency (useful/executed) | 0.7228 | 0.7372 |

Interpretation recorded with the original run: vanilla task-only GRPO
improves task quality and encourages more multi-step retrieval; both useful
and wasted searches increase, so this is **not** a cost-aware result.

## Training environment (shared by both methods)

- Model: Qwen3-8B (bf16), FSDP full sharding with param/optimizer offload.
- Rollout: vLLM 0.11.0 async server, group size n=4, temperature 0.8,
  top-p 0.95, max 5 assistant turns, max response 1024 tokens.
- Search tool: deterministic per-trajectory BM25 over the sample's own
  distractor passages; answers are never placed in prompts or tool kwargs.
- Reward (vanilla GRPO): `0.5 * EM + 0.5 * token-F1` on the strict
  `<answer>` span, no shaping terms.
- Data: strict Hotpot-MT train 2000 / val 100 parquet artifacts (SHA-256
  recorded in `PROGRESS.md`).

## Qwen3-8B vanilla GRPO training diagnostics (strict sanity, 128 prompts x 4)

- mean rollout reward 0.3039 (std 0.4383); non-trivial reward groups 36.7%;
  zero-variance groups 63.3%.
- Actor grad norms non-zero at every step: 4.04 -> 2.54.
- Executed searches averaged 1.502; multi-search episodes 43.9%; useful
  second searches 65.8%.
- Validation (strict 100) from step 0 to step 4: EM 0.240 -> 0.320,
  F1 0.3635 -> 0.4324 (technical sanity evidence, not the headline claim).

## Zero-variance motivation

On the 2,000-prompt vanilla GRPO run the zero-variance group ratio reached
0.684 — two thirds of generated groups carried no learning signal because
all four rollouts received the same reward. This is the concrete motivation
for the DAPO dynamic-sampling comparison on this task.
