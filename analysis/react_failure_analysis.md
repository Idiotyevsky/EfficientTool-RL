# ReAct Baseline Failure Analysis

## Scope

The frozen Qwen3-1.7B ReAct baseline was evaluated on HotpotQA distractor
validation indices 120–179. Indices 0–99 are reserved for prompt development
and excluded. The run completed 60/60 episodes with EM 0.400, token F1 0.506,
and 1.0 search call per episode.

Twenty successful and twenty failed trajectories were manually reviewed using
the question, query, retrieved titles, reference, and model answer. The stored
analysis covers all 60 trajectories.

## Observed Failure Modes

Automated evidence-title diagnostics categorized the 36 strict-EM failures:

- 18 retrieved both supporting titles but produced a wrong or mismatched answer;
- 15 retrieved only one supporting title;
- 2 retrieved neither supporting title;
- 1 emitted untagged text before recovering to a final answer.

Manual review confirms genuine reasoning errors, including choosing the wrong
tennis ranking, confusing a game with its venue, and reversing publication
dates. It also reveals evaluation mismatches: `Oxford` versus `University of
Oxford`, `Richmond` versus `Richmond River`, and minimal spans versus explanatory
sentences. These remain EM failures; F1 and manual categories are reported to
avoid hiding the distinction.

## Behavioral Findings

- Search-count distribution: `{1: 60}`.
- Duplicate-query rate: 0.
- Early-answer-without-search rate: 0.
- Average supporting-title recall: 0.775.
- Invalid-action rate: 0.0083.

The baseline consistently uses the tool, but it nearly always answers after one
query. The dominant improvement opportunities are better second-hop querying
and reasoning over already retrieved evidence, not merely forcing tool use.

Evidence is stored under `outputs/react_hotpotqa_heldout60_seed42/analysis/`.
