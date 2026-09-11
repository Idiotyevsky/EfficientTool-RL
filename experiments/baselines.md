# ToolAgentLab Baselines

All values are reproduced from stored artifacts. Models, normalized datasets,
checkpoints, and large trajectory files are kept outside Git; public reports
record stable artifact names and fingerprints rather than machine-local paths.

## Evaluation Protocol

Two HotpotQA-derived sets share the same runtime limits: top-1 retrieval,
384-token observations, three executed searches, and five assistant turns.

- **Hotpot-MT Strict (train 2,000 / validation 100):** controlled bridge-focused
  stress test with a question-level information-availability filter.
- **Natural Bridge-Hard (200):** official validation rows with `type=bridge`
  and `level=hard`, without question-level filtering. This is the primary
  held-out comparison. Artifact
  `verl_hotpotqa_mt_natural_bridge_hard_val_200.parquet`, SHA-256
  `1835707b46734751610d42a6f5ebba8bb3098789f841fede1c88a63b3cbf5fdc`.

## Natural Bridge-Hard Results

| Metric | Qwen3-8B Base | Vanilla GRPO | Fresh DAPO |
|---|---:|---:|---:|
| EM | 32.5% | 51.5% | 33.0% |
| F1 | 42.03% | 62.53% | 41.83% |
| Completion | 93.5% | 97.5% | 100% |
| Invalid action | 10.06% | 0.17% | 0% |
| Searches / episode | 1.335 | 1.960 | 1.100 |
| Multi-search | 31.5% | 86.0% | 10.0% |
| Useful / episode | 0.965 | 1.445 | 0.880 |
| Wasted / episode | 0.370 | 0.515 | 0.220 |
| Useful / executed | 72.28% | 73.72% | 80.0% |

## Shared Training Environment

- Qwen3-8B, bf16, FSDP full sharding with parameter/optimizer offload.
- vLLM 0.11.0 async rollout, group size 4, temperature 0.8, top-p 0.95.
- Strict one-action multi-turn protocol, maximum five assistant turns.
- Deterministic per-trajectory BM25; answer/supporting metadata is never placed
  in prompts or tool kwargs.
- Task reward: `0.5 EM + 0.5 token-F1`.
- Strict training split: 2,000 prompts; validation: 100 prompts; seed 42.

## Vanilla GRPO Diagnostics

A 128-prompt × 4-rollout sanity run produced mean reward 0.3039 (std 0.4383),
36.7% non-trivial reward groups, non-zero actor gradients at every update, and
43.9% multi-search trajectories. Strict validation changed from EM 0.240 /
F1 0.3635 to EM 0.320 / F1 0.4324 over four updates.

The full 2,000-prompt run reached a 0.684 zero-variance group ratio, motivating
the DAPO dynamic-sampling comparison.

## Fresh DAPO Qualification

The Fresh DAPO row is a completed recipe-level result. The configured overlong
buffer was not effective in that run because async `rm_scores` bypassed the
stock shaping path. It must not be used as evidence for or against agent-aware
overlong shaping. See [DAPO diagnosis](../analysis/dapo_diagnostics/README.md).
