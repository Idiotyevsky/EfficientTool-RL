# M3 GRPO Sanity Plan

## Framework Decision

Reuse verl 0.7's native `experimental.agent_loop.ToolAgentLoop` and SGLang
multi-turn rollout path. No separate Agent-R1 checkout is present, and no
distributed RL component will be reimplemented.

The installed editable package points to the clean `verl/` subtree inside
THUNLP/OPD commit `4532fd35ccfdde82adc918b265e4c964534e83d1`. The enclosing
OPD repository has unrelated user changes; this project must not modify it.

## Data and Tool Boundary

The approved train smoke set contains 500 normalized HotpotQA distractor rows
with SHA-256
`b73d78fac545d8f88bb07e4971913a3ea5fa53f304e0f99a1b5bde3f7e78f003`.
Each verl record will pass only its ten passage titles/text through
`extra_info.tools_kwargs.search.create_kwargs`. Gold answers remain solely in
`reward_model.ground_truth` and never enter the tool state.

## Reward

M3 uses only:

```text
R_task = 0.5 * EM + 0.5 * token_F1
```

Malformed, untagged, empty, or multiple answers receive zero. Search count and
token count have no reward effect in M3. Cost penalties remain forbidden until
vanilla GRPO passes.

## Bounded First Run

- model: Qwen3-1.7B;
- prompts: 8–32 before the 500-prompt smoke;
- group size: 4;
- max assistant turns: 5;
- max generated tokens: 1024;
- temperature: 0.8;
- task reward only;
- no main run until rollout diversity, non-zero group variance, reward ordering,
  parameter updates, and held-out behavioral change are demonstrated.

## Native verl Reward Boundary

Native multi-turn dumps include tool observations and role markers in the
response string. The task reward removes only that trajectory scaffolding,
then requires exactly one terminal `<answer>...</answer>` block. Stored reward
diagnostics can be reproduced with:

```bash
python scripts/analyze_verl_rollouts.py --rollouts /path/to/rollouts/1.jsonl
```
