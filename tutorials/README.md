# MiniAgentRL Hands-on Course

This course turns the repository into a guided 00→08 path. Each chapter has a runnable checkpoint, expected output, troubleshooting, an exercise, and a short bridge to the production implementation.

## Recommended order

1. [00 — Environment and the Agent RL map](00_environment_and_map.md)
2. [01 — From text to a Tool Call](01_tool_calling_basics.md)
3. [02 — Qwen’s first real Tool Call](02_real_qwen_tool_calling.md)
4. [03 — From Tool Call to Multi-turn Agent](03_multiturn_agent.md)
5. [04 — ReAct on HotpotQA](04_react_hotpot.md)
6. [05 — Rollouts, Environments, and Reward](05_rollouts_rewards_environment.md)
7. [06 — GRPO: from reward to policy update](06_grpo_for_agents.md)
8. [07 — Run a real GRPO smoke](07_grpo_smoke.md)
9. [08 — Efficient Tool Use](08_efficient_tool_use.md)

Chapters 00, 01, 03, 05, 06, and 08 run without a model. Chapter 02 uses a local Qwen checkpoint. Chapter 04 uses a local checkpoint plus normalized HotpotQA. Chapter 07 uses the real verl/Ray stack and may allocate a GPU.

The older 01–05 filenames are retained as compatibility entry points. They redirect to the canonical chapters above.
