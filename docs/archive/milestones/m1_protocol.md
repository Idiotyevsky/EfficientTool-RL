# M1 Action and Trajectory Protocol

**Status:** Approved by Sol for M1 implementation.

## Model Actions

Each model turn must contain exactly one case-sensitive action block. Reasoning
text may appear outside the block, but a second action block is invalid.

```text
<tool_call>
{"name": "search", "arguments": {"query": "..."}}
</tool_call>
```

```text
<answer>
final answer text
</answer>
```

Tool payloads must be JSON objects with a non-empty string `name` and an object
`arguments`. Final answers must be non-empty. Plain text, malformed tags,
malformed JSON, missing fields, and multiple actions become structured
`InvalidAction` values; they never crash an episode.

Unknown tools are syntactically valid calls but are rejected by the registry.
Repeated calls across turns are allowed until the configured tool budget.

## Episode Semantics

An episode starts with system and user messages. After each tool call, the
structured observation is appended to the conversation before the next model
turn. Invalid actions receive a structured correction observation and consume
a turn. A tool call beyond the budget terminates the episode without executing
the tool.

Termination reasons are:

- `final_answer` — a valid answer was produced;
- `max_turns` — no answer before the turn budget;
- `max_tool_calls` — the policy attempted an over-budget tool call.

## Trajectory Schema

Each episode records its ID, original prompt, ordered steps, final answer,
termination reason, executed tool-call count, and invalid-action count. Each
step records the turn index, raw model output, parsed action, structured
observation, and whether that step terminated the episode. JSONL is the
canonical on-disk representation.
