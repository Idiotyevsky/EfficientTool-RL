---
title: 07 · Real GRPO Smoke
description: 通过真实 verl/vLLM pipeline 完成一次 Qwen3-1.7B 参数更新。
---

# 07 · 真正更新一次 Agent

这一课不实现第二个 trainer。`train_grpo_smoke.py` 只是安全 wrapper：解析路径、选择现有 Hydra config，然后调用项目真实的 verl 入口完成一次 update。

<div class="lesson-meta">
  <MetricPill label="runtime" value="1 GPU" tone="tool" />
  <MetricPill label="model" value="Qwen3-1.7B" tone="agent" />
  <MetricPill label="rollouts" value="8 × 4" />
  <MetricPill label="updates" value="1" tone="positive" />
</div>

## 先看这一 update 穿过什么

<ConceptFlow :items="[
  { label: '8 Prompts', detail: 'bounded train slice' },
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

wrapper 会拒绝不存在的 model/data/config path，也会拒绝非空 run directory。这些 guard 防止教学 smoke 覆盖研究产物。

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

这个仓库已经验证过一条 Qwen3-1.7B teaching smoke：

<div class="lesson-meta">
  <MetricPill label="mean reward" value="0.15625" tone="agent" />
  <MetricPill label="actor/grad_norm" value="4.333" tone="positive" />
  <MetricPill label="global_step" value="1" />
  <MetricPill label="valid-answer" value="0.25" tone="tool" />
  <MetricPill label="validation EM/F1" value="0 / 0" tone="negative" />
</div>

这些值来自已保存、已验证的 Learn Track smoke。它们证明 rollout → reward → gradient → optimizer update 的 plumbing 成立。

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
    <p>一次 teaching update 不足以支持 benchmark improvement；这需要 held-out formal experiment。</p>
  </article>
</div>

## Wrapper 与真实 trainer 的边界

<<< ../../scripts/train_grpo_smoke.py{python}

<div class="code-map">
  <div><span>Safe wrapper</span><code>scripts/train_grpo_smoke.py</code></div>
  <div><span>Teaching config</span><code>configs/learn_grpo_smoke.yaml</code></div>
  <div><span>Real trainer entry</span><code>scripts/run_ppo_m3.py → verl</code></div>
</div>

## 常见失败

### Hydra 找不到 `ppo_trainer`

`--verl-config-path` 必须指向安装环境中含 trainer YAML 的目录，而不是项目根目录。

### vLLM / Ray 在 update 前失败

保存完整错误，确认 GPU ownership 与 memory。换一个 fresh run directory；不要修改 active formal 8B config 来迁就教学 smoke。

### run 完成但 grad 为 0

依次检查 final-answer formatting、reward extraction、组内 reward variance 与 clipping。全零 reward 的 run 是 failed signal evidence，不是成功 smoke。

## 动手检查

1. 先运行 `--dry-run`，逐个解释 resolved path。
2. 成功 smoke 后打开第一份 rollout JSONL，找到一个 final answer。
3. 对照 config 的 `n: 4`，检查每个 prompt 是否有四条 trajectory。
4. 找到 actor 与 optimizer checkpoint；解释二者为何都需要保存。

<LearningCheckpoint>

- 哪些证据证明 optimizer 真正更新了 actor？
- 为什么 non-zero gradient 仍不支持 performance claim？
- wrapper 为什么拒绝非空 output directory？
- 这条 teaching smoke 与 formal Qwen3-8B run 的角色有何不同？

</LearningCheckpoint>

最后一课把“调用次数”拆成真正可分析的行为成本： [08 · Efficient Tools](/learn/08-efficient-tools)。
