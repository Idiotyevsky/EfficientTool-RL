---
title: 03 · Multi-turn Agent
description: Observation 如何成为下一状态，以及为什么“支持多轮”不等于“发生多轮”。
---

# 03 · 从 Tool Call 到 Multi-turn Agent

一次 Tool Call 只证明应用能执行函数。Multi-turn Agent 还必须把 assistant action 和 tool Observation 放回上下文，让下一次生成基于新证据发生。

<div class="lesson-meta">
  <MetricPill label="runtime" value="CPU" />
  <MetricPill label="policy" value="scripted teaching fixture" />
  <MetricPill label="loop" value="real AgentRunner" tone="agent" />
  <MetricPill label="prerequisite" value="Chapter 02" />
</div>

## 先把 trajectory 摊开

下面的交互 UI 使用与 `AgentRunner` 一致的 trajectory 语义。默认成功轨迹来自 tiny-corpus scripted fixture，明确不是模型预测。

<TrajectoryExplorer />

也可以直接运行生产 loop：

```bash
PYTHONPATH=src python examples/02_multiturn_agent.py
```

预期摘要：

```text
Recorded trajectory demonstration (not model output)
Turn 1: <tool_call>...Ada Lovelace...</tool_call>
Observation: ... Analytical Engine ...
Turn 2: <tool_call>...Analytical Engine...</tool_call>
Observation: ... general-purpose mechanical computer ...
Turn 3: <answer>Analytical Engine</answer>

executed_search_calls: 2
termination_reason: final_answer
```

## 一次 state transition 到底加了什么？

初始 messages：

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question},
]
```

模型生成 action，环境执行工具后，关键是这两次 append：

```python
messages.append({"role": "assistant", "content": model_output})
messages.append({
    "role": "tool",
    "content": json.dumps(observation),
})
```

下一次 `policy.generate(messages)` 接收更长的 history。因此：

$$
s_{t+1} = s_t \oplus a_t \oplus o_t
$$

其中 `⊕` 不是数值相加，而是把 action 与 Observation 按 role 追加到上下文。

<ConceptFlow :items="[
  { label: 'State s₀', detail: 'system + question' },
  { label: 'Action a₀', detail: 'search(Ada)', tone: 'agent' },
  { label: 'Observation o₀', detail: 'new evidence', tone: 'tool' },
  { label: 'State s₁', detail: 's₀ + a₀ + o₀' },
  { label: 'Action a₁', detail: 'search(Engine)', tone: 'agent' }
]" />

<<< ../../examples/02_multiturn_agent.py{python}

## Signature lesson：Multi-turn 不是自动发生的

<div class="compare-panel">
  <article>
    <span class="section-kicker">OLD · EASY INFORMATION STRUCTURE</span>
    <h3>一次 Search(top_k=3)</h3>
    <p>一个 Observation 可能同时返回多篇 supporting passage。模型支持五轮，但第一轮已经看够证据，于是直接 Answer。</p>
    <code>Question → Search(top3) → Answer</code>
  </article>
  <article class="is-new">
    <span class="section-kicker">CONTROLLED · HOTPOT-MT</span>
    <h3>每次只返回 top_k=1</h3>
    <p>第一篇证据暴露 bridge entity；第二次 entity-specific search 才能补齐下一跳。</p>
    <code>Question → Search₁ → Observation₁ → Search₂ → Answer</code>
  </article>
</div>

必须分清三个概念：

<ConceptFlow :items="[
  { label: 'Capability', detail: 'loop 允许继续生成' },
  { label: 'Necessity', detail: '信息结构需要下一跳', tone: 'tool' },
  { label: 'Behavior', detail: 'policy 实际选择继续搜索', tone: 'agent' }
]" />

增加 `max_turns` 只增加 capability。环境是否限制一次检索的信息量决定 necessity；最终 trajectory 才说明 behavior。不要用 reward 强迫固定搜索次数来冒充自然多轮。

<div class="research-note"><strong>From the lab · pilot, not causal proof</strong><br>受控 Strict Hotpot-MT 的 Qwen3-8B pilot 记录到 P(search ≥ 2) = 31.5%。恰好两次搜索的 episode 与更高 EM 相关；正式 GRPO gate 仍以 Research Track 的 held-out evidence 为准。</div>

## 代码里对应哪里？

<div class="code-map">
  <div><span>State transition</span><code>AgentRunner.run()</code></div>
  <div><span>Budget</span><code>AgentConfig.max_tool_calls</code></div>
  <div><span>Stored step</span><code>EpisodeStep</code></div>
</div>

## 常见失败

### 第二个 Observation 永远不出现

先看第一轮 action。若是 FinalAnswer 或 InvalidAction，就没有成功 Tool transition 可供继续。

### 一轮吐出多个 action

项目坚持 one action per generation boundary。否则哪个 Observation 对应哪个 action、执行了几次工具都会变得含糊。

### 达到 tool budget

超预算请求不会执行，而会返回 `tool_budget_exhausted` 并终止。这就是 `attempted/valid` 可能高于 `executed` 的原因之一。

## 动手改一下

1. 将 `max_tool_calls` 从 3 改成 1，观察 termination reason。
2. 把第二个 query 改成与第一个完全相同，然后在 Explorer 中解释为什么它是 wasted。
3. 在 scripted policy 里打印每次 `len(messages)`，验证 state 如何增长。

<LearningCheckpoint>

- 哪两次 append 构造了下一状态？
- `max_turns=5` 为什么不证明 policy 会调用五次工具？
- capability、necessity、behavior 分别由什么证据支持？
- 为什么 top-k 会改变研究问题？

</LearningCheckpoint>

下一课把 tiny corpus 换成 bounded HotpotQA episode： [04 · ReAct + HotpotQA](/learn/04-react-hotpot)。
