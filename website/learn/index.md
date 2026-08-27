---
title: Learn Track
description: MiniAgentRL 00→08 完整学习路线。
---

# Learn Track

这不是九篇互不相关的文章。每一课都会把上一课的对象向前推进：文本 action 变成工具执行，工具执行变成多轮 trajectory，trajectory 变成 Reward，grouped rollouts 最终变成一次真实参数更新。

<div class="lesson-meta">
  <MetricPill label="course" value="00 → 08" tone="agent" />
  <MetricPill label="first steps" value="CPU" />
  <MetricPill label="real model" value="Qwen3-1.7B" tone="tool" />
  <MetricPill label="trainer" value="verl + vLLM" />
</div>

<CourseMap />

## 两条资源边界

**CPU 路线**覆盖环境检查、parser、BM25、多轮状态、Reward、GRPO 相对优势与效率指标。你可以先建立完整 mental model，不必安装 Ray。

**GPU 路线**在 Chapter 02 让 Qwen3-1.7B 真实生成 action；Chapter 07 通过项目现有的 verl/vLLM pipeline 做一次真实 optimizer update。它们使用同一套训练链路；一次更新用于理解训练过程，完整性能对比另行记录在 Research Track。

## 推荐起点

- 第一次接触 Tool Calling：从 [00 · Start](/learn/00-start) 开始。
- 已经写过 ReAct Agent：直接读 [03 · Multi-turn](/learn/03-multiturn)，重点看 environment necessity。
- 已经懂 PPO/GRPO 公式：[06 · GRPO](/learn/06-grpo) 会把公式映射到真实 verl config。
- 想继续看训练链路：阅读 [07 · Real Update](/learn/07-grpo-smoke) 和 [Research Track](/research/)。

::: warning 课程与研究的边界
课程中的小规模运行用于理解系统与训练过程；完整实验结果单独收录在 Research Track。
:::
