---
title: 指标定义
description: Task quality 与 Tool behavior 的统一统计口径。
---

# 指标定义

Efficiency claim 必须同时有任务质量与行为成本；单独报告“平均 search 下降”无法排除 under-search。

## Answer quality

- **EM**：规范化后的 exact match。
- **Token F1**：预测与 reference token overlap。
- **Task reward**：vanilla objective 中为 $0.5\,EM+0.5\,F1$。
- **Valid-answer rate**：response 是否满足 exactly one terminal answer block。

## Tool accounting

| Metric | 精确定义 |
| --- | --- |
| attempted | generated text 中 `<tool_call>` opening 数 |
| valid | parser 得到合法 `ToolCall` 的数量 |
| executed | environment 实际接受并运行的 search 数 |
| useful | executed search 新增尚未发现的 gold supporting title |
| wasted | executed 但未新增 supporting title |

$$
Tool\ Efficiency = \frac{N_{useful}}{\max(N_{executed},1)}
$$

## Behavior diagnostics

- `P(search ≥ 2)` 与 `P(search ≥ 3)`；
- duplicate-query rate；
- early-answer / zero-search rate；
- second-search useful rate；
- successful multi-turn episode rate；
- accuracy conditioned on executed-search count；
- generated tokens、assistant turns 与 malformed actions。

## GRPO diagnostics

- reward mean/std 与 per-prompt group variance；
- `zero_variance_group_ratio`；
- actor loss、KL statistics、grad norm；
- rollout generation time 与 update time；
- before/after held-out evaluation；
- checkpoint 与 parameter update evidence。

分析实现位于 [`trajectory_analysis.py`](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/src/efficienttool_rl/evaluation/trajectory_analysis.py) 与 [`verl_analysis.py`](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/src/efficienttool_rl/evaluation/verl_analysis.py)。在交互界面中直观看五种行为：[Trajectory Explorer](/playground/trajectories)。
