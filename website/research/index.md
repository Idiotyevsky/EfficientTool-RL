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
  <MetricPill label="vanilla GRPO" value="COMPLETED" tone="positive" />
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

Qwen3-8B 的 vanilla GRPO baseline comparison 已完成。Natural Bridge-Hard 上的结果显示：任务质量提升的同时，Agent 也更频繁地进行多步检索。Cost-aware Tool RL 是下一阶段，尚未发布结果。

## Latest vanilla baseline

Natural Bridge-Hard · 200 examples · Qwen3-8B Base → Step 62

| Metric | Base | Step 62 |
| --- | ---: | ---: |
| EM | 32.5% | 51.5% |
| F1 | 42.03% | 62.53% |
| Completion | 93.5% | 97.5% |
| Invalid action | 10.06% | 0.17% |
| Executed search | 1.335 | 1.960 |
| Multi-search rate | 31.5% | 86.0% |
| Useful search | 0.965 | 1.445 |
| Wasted search | 0.370 | 0.515 |
| Tool efficiency | 72.28% | 73.72% |

这是 vanilla baseline 的任务与行为结果，不是 cost-aware 结果。虽然 useful-call 占比略有上升，但 wasted calls 的绝对数量也增加了；下一步问题是能否保留有效探索，同时减少不必要的调用。

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
| Natural Bridge-Hard vanilla GRPO · 200 examples | EM 0.325 → 0.515 · F1 0.4203 → 0.6253 · multi-search 31.5% → 86.0% | Stronger task quality and more active multi-step retrieval |
| Qwen3-8B vanilla GRPO | Completed | Baseline behavior comparison |
| Cost-aware sweep | Planned | Starts after vanilla baseline evaluation |

## 审计入口

- [实验状态与证据](/research/experiments)
- [指标定义](/research/metrics)
- [环境与复现](/research/environment)
- [Raw PROGRESS.md](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/PROGRESS.md)
- [Raw debug log](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/docs/debug_log.md)

研究记录保留失败运行与 evaluator alignment 过程，方便追踪结果如何得到。
