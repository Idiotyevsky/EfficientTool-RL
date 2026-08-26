# M1 Validation Report

## Decision

**PASS — accepted by Sol on 2026-08-26.**

The minimal agent executes bounded multi-turn episodes, handles invalid model
actions without crashing, dispatches injected tools, and records each step as
structured data.

## Validation

```bash
python -m pytest -q
```

Result: `18 passed`.

The deterministic tests cover valid calls, malformed JSON, unknown tools,
missing arguments, multiple/repeated calls, plain text, final answers, tool
budgets, maximum turns, tool exceptions, and JSONL trajectory output.

The real-model smoke test used local `Qwen3-1.7B` on an idle remote GPU with
offline model loading. Its two actions were:

```text
<tool_call>
{"name": "search", "arguments": {"query": "verification token"}}
</tool_call>
<answer>M1_AGENT_OK</answer>
```

Observed outcome: `final_answer=M1_AGENT_OK`, `tool_calls=1`,
`invalid_actions=0`, and `termination_reason=final_answer`.

## Gate Review

- Multi-turn episodes work: PASS.
- Malformed actions do not crash the process: PASS.
- Tool interaction is logged structurally: PASS.
- Unit tests pass: PASS.

## Scope Boundary

The search backend remains a deterministic smoke-test stub. HotpotQA loading,
BM25 retrieval, baseline evaluation, and benchmark metrics belong to M2.
