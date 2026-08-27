---
title: 00 · Start
description: 检查环境，并建立 Agent RL 的完整地图。
---

# 00 · 从哪里开始？

先别装 Ray，也别下载 8B 模型。第一步是确认：你能导入真实项目包，并知道后续每一层增加了什么。

<div class="lesson-meta">
  <MetricPill label="runtime" value="CPU" />
  <MetricPill label="type" value="可执行" tone="positive" />
  <MetricPill label="prerequisite" value="Python 3.10+" />
</div>

## 先看结果

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
PYTHONPATH=src python examples/00_environment_check.py
```

成功时会看到：

```text
MiniAgentRL Learn Track environment check
Python:     3.x
Core package: PASS ({'kind': 'answer', ...})

Optional components:
  torch        available / not installed
  verl         available / not installed
Next step: run examples/01_tool_calling.py.
```

`Core package: PASS` 是本课 checkpoint。可选组件缺失并不妨碍 CPU 课程。

## 一张地图看完整条链

<ConceptFlow :items="[
  { label: 'Tool Calling', detail: '文本 → action', tone: 'tool' },
  { label: 'Multi-turn', detail: 'action → observation → state', tone: 'agent' },
  { label: 'Rollout', detail: '保存完整 episode' },
  { label: 'Reward', detail: 'EM / F1' },
  { label: 'GRPO', detail: 'group-relative update', tone: 'agent' },
  { label: 'Efficiency', detail: 'useful / wasted', tone: 'positive' }
]" />

## 代码从哪里读

<div class="code-map">
  <div><span>Action protocol</span><code>src/efficienttool_rl/protocol.py</code></div>
  <div><span>Agent loop</span><code>src/efficienttool_rl/agent.py</code></div>
  <div><span>GRPO training entry</span><code>scripts/run_ppo_m3.py</code></div>
</div>

课程示例直接复用核心实现，因此你在示例里看到的协议、Agent loop 和训练代码保持一致。

## 常见问题

### `efficienttool_rl` 无法导入

确认命令从仓库根目录执行，并保留 `PYTHONPATH=src`；或确认 editable install 使用的是当前 Python。

### 没有 torch / verl

先继续 CPU 课程。Chapter 02 才需要 Transformers，Chapter 07 才需要完整 verl/Ray/vLLM 环境。

## 动手改一下

分别用系统 Python 和虚拟环境 Python 运行 health check。比较 `Executable` 和 optional components；写下哪些课能在每个环境运行。

<LearningCheckpoint>

- 哪个文件把模型文本解析为 action？
- 哪个对象保存一整个 episode？
- 为什么 `verl not installed` 不代表 CPU 课程不能运行？

</LearningCheckpoint>

下一课不加载模型，先把一串“模型输出”真正变成 BM25 Observation： [01 · Tool Calling](/learn/01-tool-calling)。
