---
title: 01 · Tool Calling
description: 模型生成的是文本；parser 将文本变成 action；应用才真正执行工具。
---

# 01 · 从文本到 Tool Call

模型不会“伸手调用 Python”。它只会继续生成 token。Tool Calling 的关键，是把 **generated string、parsed action、tool execution** 三件事明确分开。

<div class="lesson-meta">
  <MetricPill label="runtime" value="CPU" />
  <MetricPill label="example" value="01_tool_calling.py" tone="tool" />
  <MetricPill label="prerequisite" value="Chapter 00" />
</div>

## 先看结果

```bash
PYTHONPATH=src python examples/01_tool_calling.py
```

你会依次看到三段输出：

```text
Model output (simulated):
<tool_call>{"name":"search","arguments":{"query":"Ada Lovelace","top_k":1}}</tool_call>

Parsed action:
{"kind":"tool_call","name":"search","arguments":{"query":"Ada Lovelace","top_k":1}}

Search observation:
[{"title":"Ada Lovelace","passage":"... Analytical Engine.","score":1.924...}]
```

第一段是明确标注的 teaching input，不是假装由模型生成的实验数据；第二、三段调用的是生产 parser 与 BM25 tool。

## 三个对象，三种责任

<ConceptFlow aria-label="Tool Calling 的三个边界" :items="[
  { label: 'MODEL OUTPUT', detail: '一串 token / string', tone: 'agent' },
  { label: 'PARSED ACTION', detail: 'parse_action() → ToolCall' },
  { label: 'TOOL EXECUTION', detail: 'BM25Search.tool()', tone: 'tool' },
  { label: 'OBSERVATION', detail: 'JSON-serializable evidence' }
]" />

### 1. Schema 不是 Tool

`SEARCH_TOOL_SCHEMA` 只描述：有一个叫 `search` 的 function，它需要 `query`，可以接收 `top_k`。Chat template 会把这份描述给模型看，但 schema 本身不会检索任何内容。

### 2. Tool Call 不是 Tool Execution

模型输出：

```xml
<tool_call>{"name":"search","arguments":{"query":"Ada Lovelace"}}</tool_call>
```

仍然只是一串字符。应用调用 `parse_action()` 后，才得到类型化的 `ToolCall(name, arguments)`；只有 dispatcher 接受它并调用 `BM25Search.tool(arguments)`，外部世界才发生效果。

### 3. Observation 是下一轮输入

工具结果被包成 JSON-serializable Observation。当前示例打印它；Chapter 03 会把它追加回 messages，构成下一状态。

## 最小真实代码路径

下面不是另写的教程实现，而是页面直接导入仓库示例：

<<< ../../examples/01_tool_calling.py{python}

<div class="code-map">
  <div><span>Protocol</span><code>src/efficienttool_rl/protocol.py</code></div>
  <div><span>Parser result</span><code>ToolCall | FinalAnswer | InvalidAction</code></div>
  <div><span>Tool</span><code>src/efficienttool_rl/tools/search.py</code></div>
</div>

## 为什么 parser 不直接“尽量修好”输出？

Agent RL 需要知道模型到底做错了什么。裸 JSON、未闭合 XML、多 action 与未知 tool 是不同 failure mode。项目对常见生成错误返回 `InvalidAction`，而不是让整个 episode 崩溃；但不会悄悄把错误文本修成成功 action。

| 输出 | Parser 结果 | 是否执行 |
| --- | --- | --- |
| 一个完整 `<tool_call>` | `ToolCall` | 由环境继续判断 |
| 裸 JSON | `InvalidAction(no_action)` | 否 |
| 两个 action block | `InvalidAction(multiple_actions)` | 否 |
| `calculator` | 合法 `ToolCall` | dispatcher 返回 unknown tool |

## 常见失败

### 模型输出了 JSON，却没有 `<tool_call>`

在 strict protocol 中这不是 action。先检查 tool schema 与 system prompt 是否进入 chat template，不要添加第二个“宽松 parser”掩盖格式问题。

### 一次生成了两个 Tool Call

本项目每个 generation boundary 只允许一个 action。这样 turn、Observation 和 executed-call accounting 才不会含糊。

### Tool 抛异常

`AgentRunner` 会把异常转换成结构化 `tool_error` Observation。错误成为 trajectory evidence，而不是训练 worker 的进程崩溃。

## 动手改一下

1. 把 `top_k` 从 1 改成 2，比较 Observation。
2. 删除闭合 tag，打印 `InvalidAction.code`。
3. 把工具名改成 `calculator`：解释为什么 parse 成功、执行仍失败。

<LearningCheckpoint>

- Schema、Tool Call、Tool Execution 分别由谁负责？
- 为什么语言模型生成了合法 JSON 仍不代表函数已执行？
- malformed action 为什么应该被保存，而不是抛到进程最外层？

</LearningCheckpoint>

现在我们只用了模拟 emission。下一课把这段字符串换成 **Qwen3-1.7B 真正生成的 action**： [02 · Real Qwen](/learn/02-real-qwen)。
