---
title: 05 · Rollout & Reward
description: EpisodeResult 如何保存行为，以及 task reward 如何从终止答案产生。
---

# 05 · Rollout、Environment 与 Reward

只保存 final answer 会丢掉 Agent RL 最重要的证据：用了什么 query、工具是否执行、看见什么 Observation、为何终止。Rollout 是完整交互记录，不是一段最终文本。

<div class="lesson-meta">
  <MetricPill label="runtime" value="CPU" />
  <MetricPill label="trajectory" value="EpisodeResult" tone="agent" />
  <MetricPill label="reward" value="0.5 EM + 0.5 F1" tone="positive" />
</div>

## 先生成并评分一条 trajectory

```bash
PYTHONPATH=src python examples/05_trajectory_reward.py
```

预期能在输出中找到：

```text
Task reward (0.5 EM + 0.5 token F1):
{"em": 1.0, "f1": 1.0, "score": 1.0, "valid_answer": 1.0}

Behavior summary:
{"avg_executed_search_calls": 1.0, "completion_rate": 1.0, ...}
```

<ConceptFlow :items="[
  { label: 'Prompt', detail: 'initial state' },
  { label: 'EpisodeStep[]', detail: 'action + observation' },
  { label: 'Final Answer', detail: 'terminal span', tone: 'agent' },
  { label: 'Task Reward', detail: 'EM / token F1', tone: 'positive' },
  { label: 'Behavior Metrics', detail: 'calls / termination', tone: 'tool' }
]" />

## Vanilla reward contract

$$
R_{task}=0.5\,EM+0.5\,F1
$$

reward adapter 要求 exactly one terminal `<answer>...</answer>`。它会去掉 native tool-response/thinking scaffolding，再对答案 span 评分。普通文本回答中“看起来答对了”仍可能因为协议无效而得 0。

<div class="code-map">
  <div><span>One transition</span><code>EpisodeStep</code></div>
  <div><span>Full rollout</span><code>EpisodeResult</code></div>
  <div><span>Task reward</span><code>src/efficienttool_rl/rewards/task.py</code></div>
</div>

## 常见失败

- **正确答案 reward=0**：打印传入 reward 的完整 response，检查 answer block 数量与 native scaffolding。
- **max_turns 终止**：policy 没有在预算内输出 FinalAnswer；检查最后 action。
- **parser error 与 tool error 混淆**：前者在 action，后者在 Observation；不要只看 reward。

## 动手改一下

把 scripted answer 改成 `London`，比较 EM、F1、score；再把 `max_tool_calls=0`，观察 action、Observation 与 termination 如何变化。

<LearningCheckpoint>

- rollout 为什么必须保存 Observation？
- Gold reference 为什么可用于 reward，却不可进入 search tool？
- 一个 tool exception 为什么应该变成 trajectory evidence？

</LearningCheckpoint>

有了同一 Prompt 的多条 rewarded rollouts，才能进入 [06 · GRPO](/learn/06-grpo)。
