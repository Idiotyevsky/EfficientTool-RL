# Chapter 03 — From One Tool Call to a Multi-turn Agent

## What you will learn

You will:

1. see exactly how an observation becomes the next model input;
2. map messages to state, action, observation, and transition;
3. understand why code-level multi-turn support does not guarantee multi-turn behavior.

## See the final effect first

Run:

~~~bash
PYTHONPATH=src python examples/02_multiturn_agent.py
~~~

The example uses a scripted policy so the path is reproducible and visible:

~~~text
Recorded trajectory demonstration (not model output):

Turn 1: <tool_call>...</tool_call>
Observation: Ada Lovelace ...

Turn 2: <tool_call>...</tool_call>
Observation: Analytical Engine ...

Turn 3: <answer>Analytical Engine</answer>

Episode summary:
{
  "termination_reason": "final_answer",
  "attempted_tool_calls": 2,
  "valid_tool_calls": 2,
  "executed_search_calls": 2
}
~~~

This is a loop demonstration, not a claim that a language model independently chose this path.

## 1. The state transition

For one episode:

~~~text
s0 = question
a0 = search("Ada Lovelace")
o0 = search results
s1 = question + a0 + o0
a1 = search("Analytical Engine")
o1 = search results
s2 = question + a0 + o0 + a1 + o1
a2 = final answer
~~~

The environment does not teleport the model to a new prompt. It appends the previous assistant action and tool observation to the message history. That history is the next state.

## 2. The key code path

AgentRunner starts with:

~~~python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
]
~~~

At each turn it asks the policy for exactly one response:

~~~python
model_output = canonicalize_action_text(
    self.policy.generate(tuple(messages))
)
action = parse_action(model_output)
~~~

For a tool call, the tool is executed and the observation is appended:

~~~python
messages.append({"role": "assistant", "content": model_output})
messages.append({
    "role": "tool",
    "content": json.dumps(observation, ensure_ascii=False, sort_keys=True),
})
~~~

The next loop iteration calls policy.generate with the longer message sequence. That is the entire mechanical reason this is multi-turn.

## 3. Why use a scripted policy here?

The ScriptedPolicy in the example emits three fixed strings and ignores messages. That sounds artificial, but it isolates the loop mechanics. Chapter 02 already demonstrated a real Qwen policy; combining model uncertainty and state-transition debugging in one first example makes failures harder to interpret.

When you are ready, replace only the policy adapter with TransformersToolPolicy. Keep AgentRunner, the tool, the protocol, and the trajectory schema unchanged.

## 4. Capability versus behavior

An environment can permit five turns while a policy answers after one. Earlier project pilots often behaved this way because top-k 3 retrieval exposed too much evidence in one call. The controlled Hotpot-MT profile uses bridge-focused questions, top-k 1, bounded observations, and an explicit search budget to create genuine information demand for later calls.

The environment should create a reason to continue. A reward should not merely force a fixed number of searches.

## Common problems

### The second observation never appears

Check the first action: if it is a final answer or InvalidAction, the loop has no successful tool transition to continue from.

### The episode ends after too many calls

AgentConfig.max_tool_calls is an execution budget. A request after the budget becomes a structured tool_budget_exhausted observation and the episode terminates.

### The model emits several actions at once

The protocol allows one action per generation boundary. This protects the meaning of a turn and makes executed-call accounting reliable.

## Exercises

1. Change AgentConfig.max_tool_calls from 3 to 1. What termination reason appears?
2. Make the second query identical to the first. The trajectory still works; Chapter 08 will measure it as waste.
3. Add a print of len(messages) inside the scripted policy. How does it grow?
4. Replace the final answer with malformed XML and inspect the stored invalid action.

## Checkpoint

Explain, without looking at the code:

- what belongs to state;
- what is an action;
- where the observation is created;
- which two append calls create the next state;
- why two allowed turns do not prove that a policy will use two turns.

## Next

[Chapter 04 — ReAct on HotpotQA](04_react_hotpot.md) replaces the tiny corpus with a bounded benchmark episode.
