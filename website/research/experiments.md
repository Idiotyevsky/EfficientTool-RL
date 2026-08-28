---
title: 实验与状态
description: vanilla GRPO 基线与下一阶段 cost-aware 实验摘要。
---

# 实验与状态

状态以仓库中的 `PROGRESS.md` 和存储 artifact 为准；本页提供面向读者的稳定摘要，不复制完整 debug log。

## 已完成

- deterministic BM25 search、strict action protocol 与 structured trajectory logging；
- training-free held-out ReAct baseline 与 byte-identical repeat；
- native verl multi-turn integration、reward parsing 修复与 actor parameter delta 验证；
- original canonical environment 的 2,000-example vanilla GRPO retraining；
- Strict Hotpot-MT 的 Qwen3-1.7B/4B/8B pilots；
- Qwen3-8B strict four-update sanity check；
- Natural Bridge-Hard fixed secondary artifact；
- Qwen3-8B vanilla GRPO baseline 与 Natural Bridge-Hard matched evaluation。

## 下一阶段

vanilla GRPO baseline 已完成。Natural Bridge-Hard 的 200 条样本上，EM 从 32.5% 提升到 51.5%，F1 从 42.03% 提升到 62.53%，multi-search rate 从 31.5% 提升到 86.0%。

更强的多步探索同时带来更高的绝对工具成本：useful search 从 0.965 增加到 1.445，wasted search 从 0.370 增加到 0.515。下一阶段将据此选择 cost-aware reward，并比较任务质量、有效探索与浪费调用。

## 明确未开始

Cost-aware reward implementation、lambda sweep 与 Pareto curve 尚未开始。

## 为什么保留失败证据？

早期运行曾出现 all-zero reward、response clipping 与 native evaluator protocol mismatch；这些记录推动了 answer parser 修复、canonical loop alignment 与更严格的 sanity check。

<div class="research-note"><strong>当前结论</strong><br>vanilla GRPO 已在 Natural Bridge-Hard 上带来更高任务质量和更强多步检索行为；cost-efficiency trade-off 尚未通过 cost-aware 对照实验验证。</div>

完整时间线：[`PROGRESS.md`](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/PROGRESS.md) · 失败调查：[`docs/debug_log.md`](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/docs/debug_log.md)。
