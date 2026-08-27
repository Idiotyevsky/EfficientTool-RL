# 3. Rollouts and Environments

An Agent RL episode is an interaction sequence:

| Concept | In this project |
| --- | --- |
| State | The question plus prior assistant actions and tool observations |
| Action | One parsed tool call or one final answer |
| Observation | A structured search result or structured error |
| Transition | The environment appends the observation to the next context |
| Episode end | Final answer, tool budget, or turn budget |

[AgentRunner](../src/efficienttool_rl/agent.py) implements this loop. Its EpisodeResult stores every step and separates:

- attempted tool calls;
- valid parsed calls;
- executed tool calls;
- executed search calls;
- invalid actions;
- termination reason.

This separation matters. A model can emit two tool-call tags, produce one malformed action, or request a tool after the budget is exhausted. Counting raw strings as executed work would mismeasure cost.

The environment is also part of the experiment. BM25 is deterministic, passages are local, observations are bounded, and search budgets are explicit. Changing top-k or observation length changes the information available to the policy and therefore changes the research question.

The [ReAct evaluator](../scripts/evaluate_react.py) stores trajectories and answer metrics. The [trajectory analyzers](../src/efficienttool_rl/evaluation/) turn those records into behavior statistics. Inspect trajectories as well as aggregate reward.
