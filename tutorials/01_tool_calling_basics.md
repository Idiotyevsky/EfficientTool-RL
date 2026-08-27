# Chapter 01 — From Text to a Tool Call

## What you will learn

You will:

1. see why a tool call is still model-generated text;
2. trace that text through the parser and BM25 search tool;
3. handle malformed actions without crashing an episode.

## See the final effect first

Run:

~~~bash
PYTHONPATH=src python examples/01_tool_calling.py
~~~

You should see a sequence like:

~~~text
Model output (simulated):
<tool_call>{"name":"search","arguments":{"query":"Ada Lovelace","top_k":1}}</tool_call>

Parsed action:
{
  "kind": "tool_call",
  "name": "search",
  "arguments": {
    "query": "Ada Lovelace",
    "top_k": 1
  }
}

Search observation:
[
  {
    "title": "Ada Lovelace",
    "passage": "...",
    "score": 1.92419407
  }
]
~~~

If you see a parsed tool_call and a passage, the parser and deterministic tool are connected. This example labels the model output as simulated: it teaches the protocol without pretending that a model generated the string.

## 1. Why not let the model call Python directly?

A language model produces tokens. It does not automatically execute a function. The application gives it a tool description, asks for a structured action, parses the generated text, and then performs the side effect on the model’s behalf.

The project protocol has two terminal shapes:

~~~text
<tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>
<answer>final answer span</answer>
~~~

The outer tags identify the action boundary. The JSON identifies the tool name and arguments. The distinction matters because ordinary prose, a JSON fragment, and an executable action are not equivalent.

## 2. What does the schema do?

The search schema tells a Qwen-compatible chat template that a function named search exists. Its arguments include a required string query and an optional top_k integer. The schema is a description, not execution. The actual implementation is BM25Search.tool.

The path is:

~~~text
model tokens
  → tagged text
  → parse_action()
  → ToolCall(name, arguments)
  → BM25Search.tool(arguments)
  → list of result dictionaries
~~~

The parser does not search. The search tool does not parse model prose. Keeping these jobs separate makes failures observable.

## 3. Read the smallest code path

The example supplies one clearly labeled teaching input:

~~~python
SIMULATED_MODEL_OUTPUT = (
    '<tool_call>{"name":"search","arguments":'
    '{"query":"Ada Lovelace","top_k":1}}</tool_call>'
)
~~~

The real parser is then called:

~~~python
action = parse_action(SIMULATED_MODEL_OUTPUT)
~~~

A successful parse creates a ToolCall. The example checks that type before passing the arguments to the real search implementation:

~~~python
if not isinstance(action, ToolCall):
    raise RuntimeError(...)
observation = search.tool(action.arguments)
~~~

The result is a JSON-serializable observation. In a real episode, that observation will be appended to the next model context; Chapter 03 shows the loop.

## 4. What happens on bad output?

parse_action returns an InvalidAction instead of raising for common model mistakes:

- no action tag;
- an unclosed tag;
- multiple actions in one turn;
- malformed JSON;
- missing name;
- missing or non-object arguments.

Unknown tool names are still parsed as ToolCall and become a structured unknown-tool observation when AgentRunner dispatches them. This lets evaluation distinguish syntax errors from environment errors.

## Common problems

### The model emits JSON without the tool_call tag

The strict protocol does not treat naked JSON as executable. This prevents arbitrary prose from silently becoming an action. Prompt/template work belongs in the real model lesson.

### The tool returns no result

Check the query type and top_k value. BM25Search validates both and raises a readable error that AgentRunner converts into an observation.

### One generation contains two actions

The project allows exactly one action per turn. Multiple actions are marked invalid so the trajectory boundary remains unambiguous.

## Exercises

1. Change top_k from 1 to 2 and compare the observation.
2. Replace the closing tag with a typo and inspect the InvalidAction fields.
3. Change the tool name to calculator and run the parser. What changes only after dispatch?
4. Add a second tool-call block and explain why it is rejected.

## Checkpoint

You are ready when you can explain:

- the schema describes capability but does not execute anything;
- parse_action converts text into a typed action;
- BM25Search produces an observation;
- invalid model output is data for analysis, not a process crash.

## Next

[Chapter 02 — Real Qwen Tool Calling](02_real_qwen_tool_calling.md) replaces the simulated string with a real local model.
