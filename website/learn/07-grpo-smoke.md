---
title: 07 · Real GRPO Smoke
description: 通过真实 verl/vLLM pipeline 完成一次 Qwen3-1.7B 参数更新。
---

# 07 · 真正更新一次 Agent

这一节直接调用项目现有的 verl 训练入口，因此你运行的是同一套 GRPO pipeline，而不是教学版模拟器。

<div class="lesson-meta">
  <MetricPill label="runtime" value="1 GPU" tone="tool" />
  <MetricPill label="model" value="Qwen3-1.7B" tone="agent" />
  <MetricPill label="rollouts" value="8 × 4" />
  <MetricPill label="updates" value="1" tone="positive" />
</div>

## 先看这一 update 穿过什么

<ConceptFlow :items="[
  { label: '8 Prompts', detail: 'small train slice' },
  { label: '32 Trajectories', detail: 'n = 4 per prompt', tone: 'agent' },
  { label: 'Tool Loop', detail: 'real multi-turn rollout', tone: 'tool' },
  { label: 'Rewards', detail: 'task-only EM / F1' },
  { label: 'Advantages', detail: 'group-relative' },
  { label: 'Gradient', detail: 'non-zero grad_norm', tone: 'positive' },
  { label: 'Optimizer', detail: 'actor parameters updated', tone: 'agent' },
  { label: 'Checkpoint', detail: 'actor + optimizer state' }
]" />

## 1. 先 dry-run，确认所有边界

```bash
PYTHONPATH=src python scripts/train_grpo_smoke.py \
  --model /path/to/Qwen3-1.7B \
  --data-dir /path/to/efficienttool-rl-data \
  --run-dir /path/to/efficienttool-rl-runs/learn_smoke_01 \
  --verl-config-path /path/to/verl/verl/trainer/config \
  --run-name learn_smoke_01 \
  --dry-run
```

预期末尾：

```text
This is a real one-update verl GRPO run: 8 prompts × 4 rollouts.
Resolved command:
... scripts/run_ppo_m3.py --config-name learn_grpo_smoke ...
Dry run: Ray was not started and no output directory was created.
```

wrapper 会拒绝不存在的 model/data/config path，也会拒绝非空 run directory；脚本要求使用独立输出目录，避免覆盖之前的训练结果。

## 2. 资源确认后再启动

```bash
nvidia-smi
df -h
```

确认 GPU 进程归属、free memory 和输出盘后，使用全新目录移除 `--dry-run`：

```bash
PYTHONPATH=src python scripts/train_grpo_smoke.py \
  --model /path/to/Qwen3-1.7B \
  --data-dir /path/to/efficienttool-rl-data \
  --run-dir /path/to/efficienttool-rl-runs/learn_smoke_01 \
  --verl-config-path /path/to/verl/verl/trainer/config \
  --run-name learn_smoke_01
```

## 3. 什么才算 smoke 成功？

仅仅“Ray 启动了”不算。需要在 log 与 run directory 中找到：

1. native multi-turn rollout 产物；
2. task reward 与 group signal；
3. non-zero actor gradient / optimizer step；
4. `global_step = 1`；
5. actor 与 optimizer checkpoint；
6. update 后 validation。

一个参考运行得到：

<div class="lesson-meta">
  <MetricPill label="mean reward" value="0.15625" tone="agent" />
  <MetricPill label="actor/grad_norm" value="4.333" tone="positive" />
  <MetricPill label="global_step" value="1" />
  <MetricPill label="valid-answer" value="0.25" tone="tool" />
  <MetricPill label="validation EM/F1" value="0 / 0" tone="negative" />
</div>

这些数值来自一个参考运行；它展示了 rollout、reward、gradient 与 optimizer update 如何串成训练链路。

## Optimization signal ≠ benchmark improvement

`grad_norm = 4.333` 说明确实发生了参数更新；validation EM/F1 仍为 0，说明一次 update 没有建立任务提升。两句话可以同时为真。

<div class="compare-panel">
  <article class="is-new">
    <span class="section-kicker">PROVEN</span>
    <h3>系统真的更新了参数</h3>
    <p>真实 rollout、reward、non-zero gradient、optimizer state 与 checkpoint 均存在。</p>
  </article>
  <article>
    <span class="section-kicker">NOT PROVEN</span>
    <h3>模型性能得到提升</h3>
    <p>一次单步训练不足以支持 benchmark improvement；性能提升需要在独立验证集上的完整实验中比较。</p>
  </article>
</div>

## Wrapper 与真实 trainer 的边界

<<< ../../scripts/train_grpo_smoke.py{python}

<div class="code-map">
  <div><span>Safe wrapper</span><code>scripts/train_grpo_smoke.py</code></div>
  <div><span>Teaching config</span><code>configs/learn_grpo_smoke.yaml</code></div>
  <div><span>GRPO training entry</span><code>scripts/run_ppo_m3.py → verl</code></div>
</div>

## 常见失败

### Hydra 找不到 `ppo_trainer`

`--verl-config-path` 必须指向安装环境中含 trainer YAML 的目录，而不是项目根目录。

### vLLM / Ray 在 update 前失败

保存完整错误，确认 GPU ownership 与 memory，并使用新的输出目录。

### run 完成但 grad 为 0

依次检查 final-answer formatting、reward extraction、组内 reward variance 与 clipping。如果所有 reward 都为 0，训练链路虽然可能完成，但当前 batch 没有提供有效的相对学习信号。

## 动手检查

1. 先运行 `--dry-run`，逐个解释 resolved path。
2. 成功 smoke 后打开第一份 rollout JSONL，找到一个 final answer。
3. 对照 config 的 `n: 4`，检查每个 prompt 是否有四条 trajectory。
4. 找到 actor 与 optimizer checkpoint；解释二者为何都需要保存。

<LearningCheckpoint>

- 哪些证据证明 optimizer 真正更新了 actor？
- 为什么 non-zero gradient 仍不足以证明性能提升？
- wrapper 为什么拒绝非空 output directory？
- 为什么一次单步训练只能验证训练链路，而不能证明模型性能提升？

</LearningCheckpoint>

最后一课把“调用次数”拆成真正可分析的行为成本： [08 · Efficient Tools](/learn/08-efficient-tools)。
