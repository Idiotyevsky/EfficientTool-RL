---
title: 08 · Efficient Tool Use
description: 区分 attempted、valid、executed、useful 与 wasted 工具调用。
---

# 08 · Efficient Tool Use

## GRPO 把 Agent 训强以后，发生了什么？

在 Natural Bridge-Hard 的 200 条样本上，Qwen3-8B 的 vanilla GRPO checkpoint 将 EM 从 **32.5% 提升到 51.5%**，F1 从 **42.03% 提升到 62.53%**。

与此同时，平均 executed search 从 **1.335 增加到 1.960**，multi-search rate 从 **31.5% 增加到 86.0%**。更多搜索确实带来了更多 useful evidence：

$$
0.965\rightarrow1.445
$$

但 wasted search 也从：

$$
0.370\rightarrow0.515
$$

这是一条 vanilla baseline 结果，不是 cost-aware 结果。它提出了一个更具体的问题：

> **能不能保留 GRPO 学到的有效多步探索，同时减少不必要的工具调用？**

因此，本章先把工具行为拆成可测量的几种 count，再讨论如何避免把 under-search 误判成效率提升。

<div class="lesson-meta">
  <MetricPill label="runtime" value="CPU" />
  <MetricPill label="analyzer" value="actual implementation" tone="positive" />
  <MetricPill label="focus" value="measurement" tone="positive" />
</div>

## 五种 count 不是同义词

<ConceptFlow :items="[
  { label: 'Attempted', detail: 'emitted opening tag' },
  { label: 'Valid', detail: 'parseable ToolCall' },
  { label: 'Executed', detail: 'environment ran it', tone: 'tool' },
  { label: 'Useful', detail: 'new supporting title', tone: 'positive' },
  { label: 'Wasted', detail: 'executed, no new evidence' }
]" />

运行 deterministic demo：

```bash
PYTHONPATH=src python examples/08_efficiency_metrics.py
```

示例摘要：

```json
{
  "attempted_tool_call_count": 4,
  "valid_tool_call_count": 4,
  "executed_search_call_count": 4,
  "useful_search_call_count": 3,
  "wasted_search_call_count": 1,
  "tool_efficiency": 0.75,
  "duplicate_query_rate": 0.25
}
```

## useful 是怎样定义的？

令 gold supporting title 集为 $G$，第 $t$ 次搜索返回 title 集 $D_t$，累计已发现集合：

$$
S_t=S_{t-1}\cup(D_t\cap G)
$$

若 $|S_t|>|S_{t-1}|$，该 executed search 带来新证据，记为 useful；否则记为 wasted。Supporting titles 只供离线 analyzer 使用，绝不进入模型 Observation。

## 少调用为什么可能更差？

<div class="compare-panel">
  <article class="is-new"><span class="section-kicker">NECESSARY EXPLORATION</span><h3>2 searches · correct</h3><p>两次分别补齐两个 supporting title；调用多，但对任务有效。</p></article>
  <article><span class="section-kicker">UNDER-SEARCH</span><h3>0 searches · wrong</h3><p>成本为零却猜错，不能称为 efficient Agent。</p></article>
</div>

因此每次 efficiency comparison 都必须同时报告 task quality、executed/useful/wasted calls 与 search-count distribution。

## 在 Explorer 中看行为

<TrajectoryExplorer />

## Strict 与 Natural 不混报

- **Hotpot-MT Strict**：controlled multi-turn stress test，包含信息可用性筛选、top-k 1 与三次 executed-search budget。
- **Natural Bridge-Hard**：不使用 strict answer-absence filter 的次级评估。

本章先解决“如何测量工具效率”。Cost-aware 训练仍在后续实验阶段，结果尚未发布。

## 动手改一下

1. 把 efficiency demo 第二个 query 改为 `Analytical Engine`，观察 wasted 降低。
2. 增加第三个重复 query，解释 attempted/valid/executed/useful/wasted 的变化。
3. 比较正确两搜与错误零搜，说明为什么不能只优化 `N_search`。

<LearningCheckpoint>

- 为什么 executed 才接近真实环境成本？
- useful 为什么只能离线计算？
- 必要第二跳与重复 query 应该受到相同惩罚吗？
- 为什么在没有完整对照实验前，不能仅凭更少的搜索次数声称 Agent 更高效？

</LearningCheckpoint>

课程到这里结束。若要复现实验与审计证据，进入 [Research Track](/research/)。
