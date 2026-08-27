# 4. GRPO for Agents

GRPO uses multiple rollouts for the same prompt and compares their rewards within the group:

~~~text
prompt
  ├── rollout 1 → reward
  ├── rollout 2 → reward
  ├── rollout 3 → reward
  └── rollout 4 → reward
~~~

The intuition is simple: a rollout that scores above its group mean receives a positive relative signal, while one below the mean receives a negative signal. In this project, the initial task-only reward is:

~~~text
R_task = 0.5 × EM + 0.5 × token F1
~~~

The group-relative signal disappears when every rollout receives the same reward. These zero-variance groups are therefore logged explicitly. A technical GRPO sanity run also checks trajectory diversity, non-trivial reward variance, non-zero gradients, parameter changes, and held-out evaluation changes.

Run the numeric demonstration:

~~~bash
PYTHONPATH=src python examples/04_grpo_concepts.py
~~~

It reuses the real task reward but does not run an optimizer or update model weights. Full training uses the native verl/vLLM path described in the [research index](../research/README.md). Agent RL is expensive because each update requires multiple generated trajectories, tool interactions, and long-tail rollout latency before the policy can be updated.

The strict Qwen3-8B four-update run changed validation EM/F1 from 0.240/0.3635 to 0.320/0.4324. This is technical sanity evidence only; the formal vanilla experiment must finish before making a task-improvement claim.
