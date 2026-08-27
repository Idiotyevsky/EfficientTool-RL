# 2. Multi-turn Agents

A multi-turn loop passes the tool observation back into the model context:

~~~text
question
  → search #1
  → observation #1
  → search #2
  → observation #2
  → final answer
~~~

The important lesson from this project is:

> Multi-turn support in code does not guarantee multi-turn behavior from the policy.

An earlier retrieval setup returned top-k 3 passages. One search often exposed enough evidence, so the model usually answered immediately. The controlled Hotpot-MT environment instead uses bridge questions, top-k 1, bounded observations, and a three-search budget. This creates information demand for later searches without forcing the model to call the tool a fixed number of times.

Run the transparent loop demonstration:

~~~bash
PYTHONPATH=src python examples/02_multiturn_agent.py
~~~

The example uses the real AgentRunner and search tool with a scripted policy adapter. Its trajectory is clearly labeled as a teaching path, not a language-model prediction.

In the fixed Qwen3-8B strict pilot, 31.5% of episodes executed at least two searches. Exactly-two-search episodes had 52.63% EM versus 9.49% after one search. This is a pilot association, not causal evidence and not a finished GRPO result.

For the research interpretation, distinguish the controlled **Hotpot-MT Strict** set from the less-filtered **Natural Bridge-Hard** secondary set in the [research index](../research/README.md).
