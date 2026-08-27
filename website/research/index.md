---
title: Research Track
description: Strict Hotpot-MT、Qwen3-8B、multi-turn GRPO 与工具效率研究总览。
---

# Research Track

Learn Track 解释系统如何工作；Research Track 回答一个可检验的问题：**GRPO 能否提升多跳 Tool Agent，同时减少没有信息增益的工具调用？**

<div class="lesson-meta">
  <MetricPill label="model" value="Qwen3-8B" tone="agent" />
  <MetricPill label="trainer" value="verl + vLLM" />
  <MetricPill label="environment" value="Strict Hotpot-MT" tone="tool" />
  <MetricPill label="vanilla GRPO" value="IN PROGRESS" />
  <MetricPill label="cost-aware RL" value="PLANNED" tone="negative" />
</div>

## 当前研究链

<ConceptFlow :items="[
  { label: 'Strict Hotpot-MT', detail: 'controlled stress test', tone: 'tool' },
  { label: 'Qwen3-8B', detail: 'non-trivial tool behavior', tone: 'agent' },
  { label: 'Vanilla GRPO', detail: 'task-only reward' },
  { label: 'Behavior Audit', detail: 'executed / useful / wasted' },
  { label: 'Cost-aware RL', detail: 'next stage' },
  { label: 'Natural Bridge-Hard', detail: 'secondary evaluation' }
]" />

Qwen3-8B 的 vanilla GRPO 对照实验正在进行。完成基线评估后，再根据任务质量与工具行为决定 cost-aware objective；pilot 与 sanity check 会明确标注为阶段性证据。

## 两个评估集回答不同问题

### Strict Hotpot-MT

受控 multi-turn stress test：bridge-focused candidate、top-k 1、384-token bounded Observation、最多三次 executed search，并使用 question-level information-availability filter。它**不是**未修改的标准 HotpotQA benchmark。

### Natural Bridge-Hard

保留官方 validation 中 `type=bridge`、`level=hard` 的 row，不使用 strict answer-absence filter；运行时 tool budget 与 top-k 保持一致。它用于检查结论是否只依赖 controlled filter。

## 已有证据的边界

| Evidence | Stored result | 合理解释 |
| --- | --- | --- |
| Held-out ReAct baseline · 60 examples | EM 0.400 · F1 0.506 · avg search 1.000 | Reference baseline |
| Strict Qwen3-8B pilot · 200 examples | EM 0.215 · F1 0.3344 · P(search≥2) 31.5% | RL 前已存在 multi-search behavior |
| Strict 8B four-update sanity | EM/F1 0.240/0.3635 → 0.320/0.4324 | Small-scale training check |
| Qwen3-8B vanilla GRPO | In progress | Full baseline comparison |
| Cost-aware sweep | Planned | Starts after vanilla baseline evaluation |

## 审计入口

- [实验状态与证据](/research/experiments)
- [指标定义](/research/metrics)
- [环境与复现](/research/environment)
- [Raw PROGRESS.md](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/PROGRESS.md)
- [Raw debug log](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/docs/debug_log.md)

研究记录保留失败运行与 evaluator alignment 过程，方便追踪结果如何得到。
