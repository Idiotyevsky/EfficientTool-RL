# Chapter 05 — Rollouts, Environments, and Reward

## What you will learn

You will:

1. inspect one complete EpisodeResult;
2. connect the trajectory to task reward;
3. distinguish environment evidence from offline gold-based scoring.

## See the final effect first

Run:

~~~bash
PYTHONPATH=src python examples/05_trajectory_reward.py
~~~

The output is verbose by design. Near the end you should find:

~~~text
Task reward (0.5 EM + 0.5 token F1):
{
  "em": 1.0,
  "f1": 1.0,
  "score": 1.0,
  "valid_answer": 1.0
}

Behavior summary:
{
  "avg_executed_search_calls": 1.0,
  "completion_rate": 1.0,
  ...
}
~~~

The exact JSON order is not important. The checkpoint is that one episode has both a structured trajectory and a score.

## 1. What is a rollout?

A rollout is one sampled interaction between a policy and an environment. In this project its minimum record contains:

~~~text
prompt
steps:
  model_output
  parsed action
  observation
  tool_executed
final_answer
termination_reason
attempted / valid / executed counts
~~~

Several rollouts for one prompt form the group used by GRPO. The environment is not just a function that returns text: it defines which actions are accepted, what observations are visible, and when an episode ends.

## 2. How is task reward extracted?

The approved vanilla task reward is:

~~~text
R_task = 0.5 × EM + 0.5 × token F1
~~~

The reward adapter first extracts exactly one terminal answer block. It removes native tool-call and tool-response scaffolding, then compares the answer span with the reference.

This is why a correct-looking answer in ordinary prose can still receive zero: the protocol requires a final answer block. Conversely, the search environment never receives the reference answer. Gold data is an offline scoring input, not an observation.

## 3. Code-to-concept map

| Concept | Project object |
| --- | --- |
| Policy action | ToolCall, FinalAnswer, or InvalidAction |
| One transition | EpisodeStep |
| Full rollout | EpisodeResult |
| Task score | rewards/task.py |
| Aggregate behavior | evaluation/metrics.py |
| Stored trajectory | JsonlTrajectoryWriter |

AgentRunner catches tool exceptions and converts them into observations. That design is important for RL: one malformed action should produce a measurable low-signal trajectory, not terminate the whole training worker.

## Common problems

### Reward is zero for a correct-looking response

Print the exact response passed to task_reward. Check for exactly one answer block, a non-empty answer, and no extra scaffolding that violates the extraction contract.

### The episode ends at max_turns

The policy did not emit a terminal answer within the turn budget. Treat this as a completion failure and inspect the last action, not only the numeric reward.

### Search error and parser error look identical

They should not. InvalidAction describes parsing; a tool-error observation describes execution. The stored action and observation fields let you separate them.

## Exercises

1. Change the scripted answer to London and compare EM, F1, and score.
2. Set max_tool_calls to zero. What happens to the first search?
3. Replace the query with malformed arguments and inspect the error observation.
4. Write the EpisodeResult to JSONL with JsonlTrajectoryWriter, then read it back.

## Checkpoint

Explain:

- why a rollout contains more than the final answer;
- why reward code must understand the output protocol;
- why gold references may be used by reward code but not by the search tool;
- why a tool exception should become evidence.

## Next

[Chapter 06 — GRPO for Agents](06_grpo_for_agents.md) uses several rewarded rollouts to produce a relative learning signal.
