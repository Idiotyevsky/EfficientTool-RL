---
title: 04 · ReAct + HotpotQA
description: 在真实 Qwen 与本地 HotpotQA passage 上运行 bounded ReAct baseline。
---

# 04 · ReAct：从 tiny corpus 到 HotpotQA

现在保留同一个 Qwen policy、AgentRunner 与 BM25，只把三篇教学 passage 换成规范化 HotpotQA row。这个 inference-only baseline 是 RL 前的可测起点，不会更新参数。

<div class="lesson-meta">
  <MetricPill label="runtime" value="Local GPU" tone="tool" />
  <MetricPill label="data" value="HotpotQA JSONL" />
  <MetricPill label="training" value="No" />
</div>

## 跑一个 bounded episode

```bash
pip install -e ".[data,hf]"
python scripts/prepare_hotpotqa.py \
  --split validation \
  --output-dir /path/to/efficienttool-rl-data

PYTHONPATH=src python examples/03_react_hotpot.py \
  --data /path/to/efficienttool-rl-data/hotpotqa_distractor_validation.jsonl \
  --model /path/to/Qwen3-1.7B \
  --limit 1 \
  --top-k 3
```

成功输出包含 question/reference/final answer、EM/F1、turns、attempted/valid/executed calls 与 termination reason。具体分数依 checkpoint 和 row 而变，不应伪造为固定 golden value。

## Model 看见什么，Evaluator 又看见什么？

<div class="compare-panel">
  <article class="is-new"><span class="section-kicker">MODEL CONTEXT</span><h3>Question + retrieved observations</h3><p>Gold answer 与 supporting title 不进入 Observation。</p></article>
  <article><span class="section-kicker">OFFLINE EVALUATION</span><h3>Reference + stored trajectory</h3><p>EM/F1 与 useful-search 标签只在 episode 后计算。</p></article>
</div>

这种边界避免 answer leakage，也让检索环境保持 deterministic。生产 evaluator `scripts/evaluate_react.py` 还会保存 trajectory、failures、metrics 与数据 fingerprint。

## top-k 改变的不是“小参数”

`top_k=3` 常让一次搜索返回多篇 evidence，适合一般 baseline；`top_k=1` 会限制单次信息量，是 Strict Hotpot-MT controlled stress test 的一部分。不同 profile 的结果不能混成一个表格。

## 常见失败

- **loader 拒绝文件**：使用仓库 preparation script，不要手写另一套 context schema。
- **每题都直接回答**：看 raw action；可能是 pretraining knowledge、prompt 或 strict tag 行为。
- **EM=0 但看似相近**：检查 `<answer>` 中是否只包含最小答案 span。

## 动手改一下

在相同 5 条 row 上比较 `top-k=1` 与 `top-k=3`，记录 executed searches 与 EM/F1，但不要把两组 profile 合并平均。

<LearningCheckpoint>

- ReAct baseline 中有没有 optimizer step？
- Gold answer 在何时可见？
- 为什么 top-k=1 与 top-k=3 属于不同 information structure？

</LearningCheckpoint>

下一课把一次 episode 变成可评分、可训练的 rollout： [05 · Rollout & Reward](/learn/05-rollout-reward)。
