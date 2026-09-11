# ToolAgentLab Composite Reward Pilot

The process-aware reward
(`R = 0.8 answer + 0.15 evidence + 0.05 format`) was audited offline on stored
fixed-policy trajectories before training. The replay script is
[`scripts/validate_composite_reward.py`](../../scripts/validate_composite_reward.py);
raw JSON reports live beside this file.

## Components

- **Answer:** `0.5 EM + 0.5 F1`.
- **Evidence:** unique gold supporting-document coverage over successful search
  responses. It is saturating and duplicate-insensitive.
- **Format:** mean of valid terminal answer, no malformed/unknown call, and
  trajectory termination with the answer.

Supporting titles come from evaluation-only metadata and never enter prompts or
tool kwargs.

## Offline Evidence

| Stored policy | Answer mean | Evidence mean | Format mean | corr(evidence, EM) | Answer share |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Base | 0.2747 | 0.5575 | 0.9483 | 0.532 | 0.627 |
| Vanilla GRPO step 62 | 0.6597 | 0.8100 | 0.9872 | 0.585 | 0.755 |

Evidence carries usable variance and correlates positively with task success.
Format remains near ceiling and auxiliary. Answer reward remains the largest
aggregate contribution.

Evidence credit on incorrect answers is intentional process supervision and is
capped by its 0.15 weight. Repeated retrieval cannot increase coverage.
