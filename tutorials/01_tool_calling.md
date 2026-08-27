# 1. Tool Calling

A tool agent turns a model’s text into an action that an environment can execute. In this project the action protocol is deliberately strict:

~~~text
<tool_call>{"name":"search","arguments":{"query":"Ada Lovelace"}}</tool_call>
~~~

The environment returns a structured observation. The agent can then continue or terminate with:

~~~text
<answer>Analytical Engine</answer>
~~~

## What to inspect

- [The parser](../src/efficienttool_rl/protocol.py) accepts exactly one tagged action.
- [The BM25 tool](../src/efficienttool_rl/tools/search.py) searches a deterministic local passage collection.
- [The tool schema](../src/efficienttool_rl/protocol.py) documents the available search arguments.
- [The first example](../examples/01_tool_calling.py) shows the whole protocol without a model.

Run it with:

~~~bash
PYTHONPATH=src python examples/01_tool_calling.py
~~~

Malformed JSON, unknown tools, missing arguments, and plain text become structured invalid actions or tool-error observations. They should be counted and analyzed, not allowed to crash an episode. This is why parsing and tool execution are separate from model generation.

## Exercise

Change the small corpus or query in the example. Observe how the parsed action stays the same while the deterministic search observation changes.
