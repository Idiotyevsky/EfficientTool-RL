---
title: 02 · Real Qwen Tool Calling
description: 让本地 Qwen3-1.7B 第一次真实生成 Tool Call，并完成下一轮回答。
---

# 02 · Qwen 第一次真的调用 Tool

上一课人为提供 `<tool_call>`，目的是隔离 protocol。现在只替换一件东西：由 `TransformersToolPolicy` 调用真实 Qwen3-1.7B 生成 action。Parser、BM25、AgentRunner 全部保持不变。

<div class="lesson-meta">
  <MetricPill label="runtime" value="Local GPU" tone="tool" />
  <MetricPill label="model" value="Qwen3-1.7B" tone="agent" />
  <MetricPill label="training" value="No" />
  <MetricPill label="prerequisite" value="Chapter 01" />
</div>

## 先看这次什么是真的

<ConceptFlow :items="[
  { label: 'Question', detail: 'tiny local corpus' },
  { label: 'Qwen3-1.7B', detail: '真实 generate()', tone: 'agent' },
  { label: '<tool_call>', detail: '模型生成的文本', tone: 'tool' },
  { label: 'BM25 Search', detail: '应用执行' },
  { label: 'Observation', detail: '回到 messages' },
  { label: 'Qwen', detail: '第二次真实 generate()', tone: 'agent' },
  { label: '<answer>', detail: '模型生成的终止 action' }
]" />

运行：

```bash
pip install -e ".[test,hf]"
PYTHONPATH=src python examples/02_real_qwen_tool_calling.py \
  --model /path/to/Qwen3-1.7B \
  --device cuda:0
```

已验证运行的代表性输出如下；不同 checkpoint/生成设置的措辞可能不同：

```text
Real model episode (the policy below is not scripted).
Question: Use search before answering: what machine did Ada Lovelace write notes about?

Turn 1 model output:
<tool_call>{"name":"search","arguments":{"query":"Ada Lovelace notes","top_k":1}}</tool_call>
Parsed action:
{"kind":"tool_call", ...}
Observation:
{"ok":true,"tool":"search","result":[...]}

Turn 2 model output:
<answer>Analytical Engine</answer>
```

页面中的 “model output” 字段就是模型新生成的 token，不是 `ScriptedPolicy` response list。

## Tool schema 是怎样进入 Qwen 的？

`TransformersToolPolicy.generate()` 将当前 messages 与 `SEARCH_TOOL_SCHEMA` 一起交给 tokenizer：

```python
prompt = tokenizer.apply_chat_template(
    materialized_messages,
    tools=[SEARCH_TOOL_SCHEMA],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

模型看见的是 chat template 序列化后的协议描述，不是 Python function 对象。随后 `model.generate()` 产出 token；policy 只 decode 新生成部分，再交给原来的 AgentRunner。

<<< ../../src/efficienttool_rl/policies/transformers_policy.py{python}

## 为什么用 tiny corpus，而不是立刻上 HotpotQA？

这里只有三个 passage：Ada Lovelace、Charles Babbage、Alan Turing。数据、query 与答案都容易人工检查。若 action 格式错误，你只需要区分 model/template/protocol 三层，不必同时排查 benchmark loader、长上下文与检索 recall。

<div class="code-map">
  <div><span>Schema</span><code>protocol.SEARCH_TOOL_SCHEMA</code></div>
  <div><span>Model adapter</span><code>policies/transformers_policy.py</code></div>
  <div><span>Episode boundary</span><code>agent.AgentRunner.run</code></div>
</div>

## 如果 Qwen 直接回答，会发生什么？

直接 `<answer>` 是一个合法终止 action，但 `executed_search_calls = 0`。可能原因包括：模型从 pretraining 已知答案、system prompt 约束不足、schema 未进入 template。它是行为证据，不应该被伪装成一次成功 Tool Call。

按这个顺序排查：

1. 看原始 model output；
2. 看 parsed action；
3. 看 Observation 是否存在；
4. 看 episode summary 中 `executed_search_calls`。

## 常见失败

### checkpoint path 被拒绝

policy 使用 `local_files_only=True`。传入本地 Hugging Face checkpoint 目录，而不是远程模型 ID。

### 输出裸 JSON 或 malformed tag

保留 raw output。检查 Qwen chat template 是否支持并接收 `tools`，不要为教程增加宽松 parser。

### 显存不足

先减少 `--max-new-tokens` 或使用更小 checkpoint。这是 inference lesson；不要改 formal 8B 训练配置。

## 动手改一下

1. 把问题改成 “Who designed the Analytical Engine?”。
2. 分别 greedy 和 `--sample` 运行，比较 action 格式稳定性。
3. 设置 `--max-search-calls 0`，观察合法 Tool Call 如何被 budget 拒绝。

<LearningCheckpoint>

- 哪一行把 schema 交给模型？
- 哪些字符串是 Qwen 真正生成的？
- 为什么 `executed_search_calls = 0` 不能简单归因于 Tool 故障？
- 这次运行有没有更新模型参数？

</LearningCheckpoint>

下一课关注 Observation 如何改变下一轮 state： [03 · Multi-turn](/learn/03-multiturn)。
