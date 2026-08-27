# Chapter 08 — Efficient Tool Use

## What you will learn

You will:

1. distinguish attempted, valid, executed, useful, and wasted calls;
2. run the production trajectory analyzer on a controlled example;
3. see why fewer searches is not the same as better tool use.

## See the final effect first

Run:

~~~bash
PYTHONPATH=src python examples/08_efficiency_metrics.py
~~~

The deterministic demonstration compares one path that finds two supporting titles with one that repeats the first query. You should see a summary like:

~~~text
{
  "attempted_tool_call_count": 4,
  "valid_tool_call_count": 4,
  "executed_search_call_count": 4,
  "useful_search_call_count": 3,
  "wasted_search_call_count": 1,
  "tool_efficiency": 0.75,
  "duplicate_query_rate": 0.25
}
~~~

Then it prints per-episode useful and wasted counts. If these values appear, the production analyzer—not a tutorial-only formula—has classified the trajectories.

## 1. Five different counts

| Metric | Question it answers |
| --- | --- |
| Attempted | Did the model emit a tool-call opening? |
| Valid | Did the action parse as a tool call? |
| Executed | Did the environment accept and run it? |
| Useful | Did the search add a previously unseen supporting title? |
| Wasted | Did an executed search add no new supporting title? |

These are intentionally not synonyms. A malformed call can be attempted but not valid. A valid call can be rejected by an unknown-tool or exhausted-budget boundary. An executed search can be correct but redundant.

For a supporting-title set \(G\) and results \(D_t\), the analyzer tracks:

\[
S_t = S_{t-1} \cup (D_t \cap G)
\]

A search is useful when \(|S_t| > |S_{t-1}|\). Wasted calls are executed calls that do not add a new supporting title.

## 2. Why not minimize search count directly?

Consider two episodes:

~~~text
A: search evidence A → search evidence B → correct answer
B: guess without search → wrong answer
~~~

A has more calls but better task behavior. A cost objective that blindly rewards zero calls can teach under-search. The project therefore records task quality alongside executed cost and inspects accuracy conditioned on search count.

Duplicate queries, premature answers, long trajectories, and repeated no-evidence searches are more informative targets than necessary evidence gathering.

## 3. Where do useful labels come from?

Supporting titles are available to offline analysis because HotpotQA provides them. They are never inserted into the model observation. The agent sees only search results.

For real stored trajectories, use:

~~~bash
PYTHONPATH=src python scripts/analyze_trajectories.py \
  --data /path/to/hotpotqa_distractor_validation.jsonl \
  --trajectories /path/to/react_run/trajectories.jsonl \
  --output-dir /path/to/react_run/analysis
~~~

The resulting behavior_metrics.json includes search-count distributions, duplicate-query rate, useful/wasted calls, early answers, and accuracy by search count.

## 4. Strict versus natural evaluation

**Hotpot-MT Strict** is a controlled multi-turn stress test: bridge-focused candidates, top-k 1, bounded observations, and a three-search budget. Its strict candidate filter must be disclosed.

**Natural Bridge-Hard** keeps official validation bridge/hard rows without the strict answer-absence filter. It is a secondary check against conclusions that depend only on the controlled set.

Do not merge the two populations into one score.

## Common problems

### Raw output contains more tool tags than executed calls

Count successful tool responses or the structured tool_executed field. Raw strings measure attempts, not environment work.

### Every search is labeled wasted

Check that the analyzer has the matching example IDs and supporting titles. A trajectory-only file cannot infer usefulness without the corresponding dataset metadata.

### Efficiency improves because accuracy collapses

That is under-search, not a successful efficiency result. Always report EM/F1, completion, executed calls, and search-count distributions together.

### A repeated query is not exactly identical

The analyzer normalizes queries for duplicate accounting, but near-duplicate semantic queries may need a separate analysis rule. Do not silently change the official metric while interpreting results.

## Exercises

1. Change the second query in examples/08_efficiency_metrics.py to Analytical Engine and watch wasted calls fall.
2. Add a third repeated search and observe the count distribution.
3. Make the first action malformed. Which metric changes?
4. Compare a correct two-search path with an incorrect zero-search path and explain which is more useful to the task.

## Checkpoint

You can finish the course when you can explain:

- why executed is the cost measure;
- why useful is an offline evidence-increment label;
- why a necessary second search should not be penalized like a duplicate;
- why cost-aware RL must wait for a valid vanilla baseline.

> **Research note.** The active formal strict Qwen3-8B vanilla run is still the gate for cost-aware RL. No M5 efficiency improvement is claimed until held-out task quality and executed/useful/wasted behavior are measured together.

## Further reading

See the [research index](../research/README.md), [trajectory analysis implementation](../src/efficienttool_rl/evaluation/trajectory_analysis.py), and [current status](../PROGRESS.md).
