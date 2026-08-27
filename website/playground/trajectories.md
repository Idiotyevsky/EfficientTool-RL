---
title: Trajectory Explorer
description: 用真实 Agent 数据结构理解多轮工具行为。
---

# Trajectory Explorer

汇总指标会告诉你“平均调用了几次工具”，Trajectory 会告诉你“为什么”。这里的五条轨迹使用项目 `AgentRunner`、parser 和行为指标的语义，但均明确标注为 **Teaching example**，不是模型预测或实验结果。

<TrajectoryExplorer />

## 如何读这些数字

`attempted` 只说明模型发出了 `<tool_call>` 开头；`valid` 要求 action 能被解析；`executed` 才代表环境真的接受并运行了工具。`useful` 与 `wasted` 是离线证据分析：一次搜索是否增加了尚未发现的 supporting title。

想看这些定义如何从真实 JSONL 计算，请继续 [08 · Efficient Tools](/learn/08-efficient-tools) 或直接阅读 [`trajectory_analysis.py`](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/src/efficienttool_rl/evaluation/trajectory_analysis.py)。
