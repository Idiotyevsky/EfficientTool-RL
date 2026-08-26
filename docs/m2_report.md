# M2 Validation Report

## Decision

**PASS — accepted by Sol on 2026-08-26.**

## Data and Retrieval

HotpotQA distractor validation was normalized to 7,405 JSONL records on ZFS.
The 46,428,993-byte artifact has SHA-256
`c878eb9b73efd7f660dc8bc5fcf0d4a7316122608a0a6846d2da638b56d3739e`.
Search receives only passage titles/text; answers and supporting labels remain
offline evaluation data. Deterministic Okapi BM25 returns bounded, structured
results with corpus-order tie breaking.

## Baseline Evidence

The frozen Qwen3-1.7B run on held-out indices 120–179 produced:

| Metric | Value |
|---|---:|
| Episodes / completion | 60 / 100% |
| Exact Match | 0.400 |
| Token F1 | 0.506 |
| Average search calls | 1.000 |
| Average turns | 2.017 |
| Average generated tokens | 45.983 |
| Invalid-action rate | 0.0083 |

All 31 deterministic tests pass. Twenty successes and twenty failures were
manually reviewed; full behavioral categories are recorded in
`analysis/react_failure_analysis.md`.

## Reproducibility Check

Indices 100–119 were run twice with identical model, code, seed, device, and
configuration. Both runs produced byte-identical trajectories and metrics:

- trajectories SHA-256: `614c24f9803085200d7c820f9dd699cee1423d6af244f2bf60ad106325ad2246`;
- metrics SHA-256: `ad583d1d377d8ee852d76d464fa3b0d3ebb125fa35aec25f3999d241262b76f1`.

## Gate Review

- Metrics are reproducible: PASS.
- Trajectories are structurally inspectable: PASS.
- Search observations enter subsequent model context: PASS.
- Answer/support labels are excluded from tool observations: PASS.
- Failures are categorized and manually reviewed: PASS.

M2 validates the baseline system; it does not claim competitive benchmark
performance. GRPO work may begin only with M3 sanity checks.
