---
title: 06 · GRPO
description: 从 Reward、组内 Advantage、policy ratio 与 clipping 走到 verl 配置。
---

# 06 · GRPO：Reward 怎样改变参数？

“算出四个 reward 的 z-score”还不是训练。真正链路必须继续经过 policy ratio、clipped objective、KL、gradient 与 optimizer step。本课先让组内信号变得可见，再把每个概念映射到项目配置。

<div class="lesson-meta">
  <MetricPill label="runtime" value="CPU" />
  <MetricPill label="interaction" value="reward sliders" tone="agent" />
  <MetricPill label="optimizer" value="next chapter" />
  <MetricPill label="prerequisite" value="Rollout · Reward" />
</div>

## 先动手改变一个 group

拖动任意 Reward。观察 mean、std 和四条 Advantage 如何一起变化；再把四个值调成相同，触发 zero-variance group。

<GRPOGroupDemo />

命令行版本复用真实 task reward：

```bash
PYTHONPATH=src python examples/04_grpo_concepts.py
```

它会打印四条 rollout、`0.5 EM + 0.5 F1` reward、组均值与标准化 Advantage，并明确说明没有调用 optimizer。

## 1. 为什么同一个 Prompt 要采样多次？

对同一问题采样 $G$ 条 trajectory：

$$
\mu_R = \frac{1}{G}\sum_{i=1}^{G}R_i,
\qquad
A_i = \frac{R_i - \mu_R}{\max(\sigma_R, \epsilon)}
$$

绝对 Reward 变成“相对同组其他尝试表现如何”。高于组均值的 trajectory 获得正 Advantage，低于均值获得负 Advantage。若所有 Reward 相同，$\sigma_R=0$，组内没有偏好方向。

## 2. Advantage 怎样影响 token probability？

Rollout 由旧 policy 生成。更新中的 candidate policy 与它比较：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

clipped objective 的核心形式：

$$
L_{policy}=\mathbb{E}_t\left[\min\left(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]
$$

- $A_t > 0$：提高这条 sampled trajectory 中 action token 的相对概率；
- $A_t < 0$：降低它们；
- ratio 表示新旧 policy 对同一采样 action 的概率变化；
- clipping 限制一次 update 的步幅；
- KL regularization 进一步约束 policy 不要偏离 reference 太远。

<ConceptFlow :items="[
  { label: 'Tokens', detail: 'sampled actions' },
  { label: 'Rollouts', detail: 'G per prompt' },
  { label: 'Reward', detail: 'EM / F1' },
  { label: 'Advantage', detail: 'group-relative', tone: 'positive' },
  { label: 'Ratio + Clip', detail: 'bounded objective' },
  { label: 'Gradient', detail: 'backprop', tone: 'agent' },
  { label: 'New θ', detail: 'optimizer step', tone: 'agent' }
]" />

## 3. Formula → Config → Code

这些公式会逐一对应到实际配置，而不是停留在抽象符号层面。

<div class="formula-config-grid">
  <article class="formula-config-card">
    <span>GROUP SIZE</span><h3>G = 4</h3>

```yaml
actor_rollout_ref:
  rollout:
    n: 4
```
  </article>
  <article class="formula-config-card">
    <span>RELATIVE ADVANTAGE</span><h3>GRPO estimator</h3>

```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true
```
  </article>
  <article class="formula-config-card">
    <span>KL REGULARIZATION</span><h3>Keep updates bounded</h3>

```yaml
actor:
  use_kl_loss: true
  kl_loss_coef: 0.001
```
  </article>
  <article class="formula-config-card">
    <span>TASK REWARD</span><h3>0.5 EM + 0.5 F1</h3>

```yaml
custom_reward_function:
  path: rewards/task_reward.py
  name: compute_score
```
  </article>
</div>

完整教学配置是 [`configs/learn_grpo_smoke.yaml`](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/configs/learn_grpo_smoke.yaml)。verl 的统一入口属于 PPO-family trainer；`algorithm.adv_estimator: grpo` 才决定这里使用 GRPO。

## 4. 为什么 Agent RL rollout 很贵？

每条 rollout 都可能包含多次模型 generation、tool execution、Observation tokenization，再为 actor/reference 计算 log probability。`8 prompts × 4 rollouts` 不是 8 次普通问答，而是 32 条可能多轮的 trajectory，随后还要反向传播。

## 常见失败

### 全组 Reward 都一样

记录 `zero_variance_group_ratio`。它可能表示模型输出过于同质、reward 太稀疏、parser 没取到 final answer，或任务对当前 policy 太难/太容易。

### 所有 Reward 都是 0

先检查 answer extraction、response clipping、native tool scaffolding 与 final tag；过早加入 shaping 可能掩盖真正的问题。

### loss 在变，参数却没变

检查 non-zero grad norm、optimizer step、checkpoint 和具体 parameter delta。打印一个 loss 不是 policy update 的充分证据。

## 动手改一下

1. 在交互组件里把四个 Reward 调成一样，解释为什么 Advantage 全 0。
2. 修改 `examples/04_grpo_concepts.py` 中一个答案，运行前先预测 Advantage 符号。
3. 在 `configs/learn_grpo_smoke.yaml` 找到 group size、KL coefficient 与 learning rate。

<LearningCheckpoint>

- 为什么 Reward z-score 本身不是 GRPO training？
- positive Advantage 最终影响什么概率？
- clipping 与 KL 分别限制什么？
- `rollout.n: 4` 在数学里对应哪个符号？

</LearningCheckpoint>

下一课把图中的 Gradient 与 optimizer step 真实跑一次： [07 · Real GRPO Smoke](/learn/07-grpo-smoke)。
