# Chapter 02 — Qwen’s First Real Tool Call

## What you will learn

You will:

1. pass a real search schema into a Qwen chat template;
2. observe a model-generated action on a tiny local corpus;
3. separate model behavior from parser and environment behavior.

This is the first chapter that loads a model. It does not use HotpotQA, Ray, verl, or distributed training.

## See the final effect first

Install the Hugging Face extras and point the example at a local checkpoint:

~~~bash
pip install -e ".[test,hf]"

PYTHONPATH=src python examples/02_real_qwen_tool_calling.py \
  --model /path/to/Qwen3-1.7B \
  --device cuda:0
~~~

The model output is stochastic or model-version dependent, so the following is representative rather than an exact golden transcript:

~~~text
Real model episode (the policy below is not scripted).
Question: Use search before answering: what machine did Ada Lovelace write notes about?

Turn 1 model output:
<tool_call>{"name":"search","arguments":{"query":"Ada Lovelace notes","top_k":1}}</tool_call>
Parsed action:
{"kind": "tool_call", "name": "search", ...}
Observation:
{"ok": true, "tool": "search", "result": [...]}

Turn 2 model output:
<answer>Analytical Engine</answer>
~~~

The important checkpoint is not the exact wording. It is seeing a real generated action parsed as a tool call and followed by a structured observation. A direct answer is also useful evidence; troubleshoot it instead of silently calling it a successful tool-use run.

## 1. Where does the model learn about search?

The local policy is [TransformersToolPolicy](../src/efficienttool_rl/policies/transformers_policy.py). Its generate method:

1. copies the current role/content messages;
2. supplies SEARCH_TOOL_SCHEMA as the tools argument;
3. calls the tokenizer’s apply_chat_template;
4. disables Qwen thinking when the template supports that option;
5. generates text from the resulting prompt;
6. decodes only the newly generated tokens.

The schema therefore reaches the model through the chat template. Merely having a Python function named search is not enough.

## 2. What does the example add?

The example uses three local passages:

~~~text
Ada Lovelace      → wrote notes on Babbage’s Analytical Engine
Charles Babbage   → designed the Analytical Engine
Alan Turing       → unrelated comparison passage
~~~

It constructs the real BM25Search and the real AgentRunner. The policy is TransformersToolPolicy; there is no scripted response list. The only small convenience is a bounded_search adapter that caps the requested top_k, exactly like the local evaluator.

The first action and its observation are still handled by AgentRunner. This means the lesson teaches the same episode boundary used later by ReAct and GRPO.

## 3. Why might the model answer immediately?

A model can:

- know the answer from pretraining;
- ignore the instruction to search;
- fail to emit the strict tag;
- emit a format that the tokenizer or parser does not recognize.

These are different failures. Look at the printed model output, parsed action, and termination reason in that order. Do not infer “the tool is broken” merely because executed_search_calls is zero.

## Common problems

### The checkpoint path is rejected

The policy intentionally uses local_files_only=True. Download or prepare a Hugging Face-format checkpoint first, then pass its directory, not a remote model identifier.

### The model answers without a tool call

Try the exact prompt, use the default greedy decoding first, and inspect whether the tools schema is present in the chat template. You can also run with --sample, but sampling changes reproducibility.

### The output is a naked JSON object

That is a protocol-format failure. Keep the raw output as evidence. Do not add a second parser that silently repairs it for the tutorial.

### CUDA memory is insufficient

Reduce max-new-tokens and use a smaller local checkpoint. This is an inference-only lesson; do not change the formal 8B training configuration to make it fit.

## Exercises

1. Change the question to ask who designed the Analytical Engine.
2. Set max-search-calls to 1 and observe the budget boundary.
3. Run once greedily and once with --sample. Compare action formatting and termination.
4. Add a passage and test whether the query changes the observation.

## Checkpoint

You are ready when you can point to:

- the line that gives the model its tool schema;
- the line that calls the real model;
- the line where AgentRunner parses the generated text;
- the output field that tells you whether a search actually executed.

> **Research note.** The small-corpus lesson demonstrates capability, not benchmark performance. Strict Hotpot-MT behavior and Qwen3-8B pilot evidence belong to the research track.

## Next

[Chapter 03 — Multi-turn Agents](03_multiturn_agent.md) makes the observation part of the next state.
