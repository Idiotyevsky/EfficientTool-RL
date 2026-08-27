---
title: 实验与状态
description: 已完成、进行中和明确未开始的实验线。
---

# 实验与状态

状态以仓库中的 `PROGRESS.md` 和存储 artifact 为准；本页提供面向读者的稳定摘要，不复制完整 debug log。

## 已完成

- deterministic BM25 search、strict action protocol 与 structured trajectory logging；
- training-free held-out ReAct baseline 与 byte-identical repeat；
- native verl multi-turn integration、reward parsing 修复与 actor parameter delta 验证；
- original canonical environment 的 2,000-example vanilla GRPO retraining；
- Strict Hotpot-MT 的 Qwen3-1.7B/4B/8B pilots；
- Qwen3-8B strict four-update sanity gate；
- Natural Bridge-Hard fixed secondary artifact。

## 正在进行

Formal Strict Hotpot-MT · Qwen3-8B · task-only GRPO · 2,000 training examples。该 run 使用四张 A6000-class GPU；包装改造没有修改 config、runtime directory、checkpoint 或 trainer process。

最终需要在匹配 held-out protocol 下比较：EM、F1、executed/useful/wasted search、turns、tokens 与 search-count distribution。

## 明确未开始

Cost-aware reward implementation、lambda sweep、Pareto curve 与 final M5 claim。只有 formal vanilla gate 被 Sol 接受后，才能选择 reward semantics 并启动。

## 为什么保留失败证据？

早期 run 曾出现 all-zero reward、response clipping 与 native evaluator protocol mismatch。它们没有被删除或包装成成功，而是推动了 answer parser 修复、canonical loop alignment 与更严格的 sanity gate。

<div class="research-note"><strong>当前结论</strong><br>系统已证明能产生真实 multi-turn rollout、task reward、non-zero gradient 与 actor update；formal strict task-improvement 与 cost-efficiency trade-off 仍是 TBD。</div>

完整时间线：[`PROGRESS.md`](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/PROGRESS.md) · 失败调查：[`docs/debug_log.md`](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/docs/debug_log.md)。
