---
title: 08 · Efficient Tool Use
description: 区分 attempted、valid、executed、useful 与 wasted 工具调用。
---

# 08 · Efficient Tool Use

“搜索越少越好”会奖励不搜索就猜。真正的问题是：Agent 能否保留必要的多跳检索，同时减少没有信息增益的执行成本？

<div class="lesson-meta">
  <MetricPill label="runtime" value="CPU" />
  <MetricPill label="analyzer" value="production code" tone="positive" />
  <MetricPill label="cost-aware result" value="TBD" tone="negative" />
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

已验证预期摘要：

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

Cost-aware objective 必须等待 formal vanilla gate；当前结果保持 **TBD**，不在教程中预演成功结论。

## 动手改一下

1. 把 efficiency demo 第二个 query 改为 `Analytical Engine`，观察 wasted 降低。
2. 增加第三个重复 query，解释 attempted/valid/executed/useful/wasted 的变化。
3. 比较正确两搜与错误零搜，说明为什么不能只优化 `N_search`。

<LearningCheckpoint>

- 为什么 executed 才接近真实环境成本？
- useful 为什么只能离线计算？
- 必要第二跳与重复 query 应该受到相同惩罚吗？
- 为什么 M5 结果仍必须写 TBD？

</LearningCheckpoint>

课程到这里结束。若要复现实验与审计证据，进入 [Research Track](/research/)。
