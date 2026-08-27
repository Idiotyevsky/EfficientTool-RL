# Chapter 04 — ReAct on HotpotQA

## What you will learn

You will:

1. run the real local evaluator on a bounded HotpotQA slice;
2. connect question data, per-example passages, BM25, and answer metrics;
3. understand what a training-free ReAct baseline can and cannot tell you.

## See the final effect first

You need a normalized validation JSONL file and a local Qwen checkpoint. If the data is not prepared yet:

~~~bash
pip install -e ".[data]"
python scripts/prepare_hotpotqa.py \
  --split validation \
  --output-dir /path/to/efficienttool-rl-data
~~~

Then run one bounded example:

~~~bash
pip install -e ".[test,hf]"
PYTHONPATH=src python examples/03_react_hotpot.py \
  --data /path/to/efficienttool-rl-data/hotpotqa_distractor_validation.jsonl \
  --model /path/to/Qwen3-1.7B \
  --limit 1 \
  --top-k 3
~~~

The output is one JSON record. Values depend on the checkpoint and selected row:

~~~text
{
  "id": "...",
  "question": "...",
  "answer": "...",
  "exact_match": 0.0 or 1.0,
  "f1": 0.0 ... 1.0,
  "attempted_tool_calls": ...,
  "valid_tool_calls": ...,
  "executed_search_calls": ...,
  "termination_reason": "final_answer"
}
~~~

If the record prints, the local policy, agent loop, search environment, and answer metric have all crossed one episode boundary.

## 1. What is ReAct here?

This project’s ReAct baseline is inference-only:

~~~text
question
  → Qwen generates a search action
  → local BM25 returns passages
  → observation enters the next context
  → Qwen answers
~~~

There is no policy update. The purpose is to measure a stable starting point before GRPO changes the policy.

Each HotpotQA row contains a question, answer, context passages, supporting facts, type, and level. The local evaluator gives the model only the question and retrieved observations. Gold answer and supporting titles are used offline for scoring and analysis.

## 2. Why search each example’s local passages?

The initial environment is deterministic and offline. BM25 indexes the passages supplied for the current example, so the run is reproducible and does not depend on a live web API. This is a controlled research environment, not a web-search product.

The evaluator exposes useful limits:

- max-turns;
- max-search-calls;
- top-k;
- maximum observation tokens.

For the controlled multi-turn profile, use bridge-focused rows with top-k 1 and bounded observations. For a general ReAct baseline, top-k 3 is also useful. Record the profile with the results; changing it changes the information available to the policy.

## 3. Read the evaluator in layers

The evaluator:

1. loads normalized rows with load_hotpotqa;
2. constructs a BM25Search for one example;
3. caps the model’s top_k request;
4. runs AgentRunner;
5. scores final_answer with answer_metrics;
6. stores trajectories, failures, and metrics when using scripts/evaluate_react.py.

The example file is a bounded console adapter. The production script adds output directories, SHA-256 data provenance, JSONL trajectories, and aggregate metrics.

## Common problems

### The data loader rejects the file

The loader expects normalized JSON or JSONL with context entries shaped like title plus sentence list. Use the repository preparation script rather than hand-writing a new schema.

### The model path works in another program but not here

The policy uses local files only. Pass the local Hugging Face directory and install the hf extra in the same interpreter used for the command.

### Every episode answers immediately

Inspect the raw output. The model may know the answer, ignore the tool instruction, or fail the strict tag. Try the controlled top-k 1 profile only when studying multi-turn behavior; do not silently mix metrics from different profiles.

### EM is zero but the answer looks close

HotpotQA metrics normalize punctuation/articles but still score the extracted final answer. Check that the model emitted one answer block and did not include an explanation.

## Exercises

1. Run limit 5 and save the console output.
2. Compare top-k 1 and top-k 3 on the same rows.
3. Run the production evaluator with a unique output directory and inspect trajectories.jsonl.
4. Add question-type bridge and levels medium hard filters, then record how the population changed.

## Checkpoint

You can continue when you can state:

- what the model sees initially;
- where passages come from;
- where gold answers are used;
- why this is a baseline rather than training;
- which command stores inspectable trajectories.

## Next

[Chapter 05 — Rollouts, Environments, and Reward](05_rollouts_rewards_environment.md) turns the episode record into an RL training signal.
