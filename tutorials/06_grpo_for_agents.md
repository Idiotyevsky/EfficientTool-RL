# Chapter 06 — GRPO for Agents: From Reward to Policy Update

## What you will learn

You will:

1. compute group-relative rewards without mistaking that for training;
2. connect the math to this repository’s verl configuration;
3. understand how a positive advantage eventually changes token probabilities.

## See the final effect first

Run the bounded numerical demonstration:

~~~bash
PYTHONPATH=src python examples/04_grpo_concepts.py
~~~

You should see four rollouts, their rewards, and relative advantages:

~~~text
Rollouts:
  rollout 1: reward=1.000, relative_advantage=+...
  rollout 2: reward=1.000, relative_advantage=+...
  rollout 3: reward=0.333, relative_advantage=-...
  rollout 4: reward=0.000, relative_advantage=-...

Group mean reward: ...
Group reward std:  ...
~~~

This example uses the real task reward, but it does not call an optimizer. It is the first half of the explanation, not GRPO training.

## 1. Why groups?

For one prompt, sample G trajectories:

~~~text
prompt
  ├── rollout 1 → R1
  ├── rollout 2 → R2
  ├── rollout 3 → R3
  └── rollout 4 → R4
~~~

The group mean and standard deviation turn absolute task scores into a relative signal:

\[
\mu_R = \frac{1}{G}\sum_i R_i
\]

\[
A_i = \frac{R_i-\mu_R}{\max(\sigma_R,\epsilon)}
\]

A rollout above its group mean has positive advantage. A rollout below it has negative advantage. If every reward is identical, the group has zero variance and no useful relative direction.

The current task reward is:

\[
R_i = 0.5\,EM_i + 0.5\,F1_i
\]

No search penalty is included in the active vanilla objective.

## 2. How does advantage change the policy?

During rollout, the old policy produced action tokens with probabilities \(\pi_{\theta_{\mathrm{old}}}\). After the actor is updated, the candidate policy has probabilities \(\pi_\theta\). For each sampled token:

\[
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\]

A clipped policy objective has the form:

\[
L_{\mathrm{policy}} =
\mathbb{E}_t\left[
\min\left(
r_t A_t,\,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
\]

The implementation minimizes the corresponding loss, so the intuition is:

- \(A_t > 0\): increase the probability of tokens from a relatively successful trajectory;
- \(A_t < 0\): decrease their probability;
- the ratio compares the new policy with the policy that generated the data;
- clipping prevents one update from changing probabilities too aggressively.

With KL regularization, the actor also pays for drifting too far from its reference policy:

\[
L = L_{\mathrm{policy}} - \beta\,D_{\mathrm{KL}}
\]

The sign and exact reduction conventions are handled by verl; the conceptual job of KL is to stabilize the update.

## 3. Formula to project configuration

The same ideas appear in the real configs:

| Idea | Configuration |
| --- | --- |
| Group size \(G=4\) | actor_rollout_ref.rollout.n: 4 |
| GRPO estimator | algorithm.adv_estimator: grpo |
| Normalize by group std | algorithm.norm_adv_by_std_in_grpo: true |
| KL actor loss | actor_rollout_ref.actor.use_kl_loss: true |
| KL coefficient | actor_rollout_ref.actor.kl_loss_coef: 0.001 |
| Learning rate | actor_rollout_ref.actor.optim.lr: 1e-6 |
| Task-only reward | custom_reward_function → rewards/task_reward.py |

The unified verl entry point is named run_ppo_m3.py because verl uses a PPO-family trainer. The config selects GRPO; the filename does not turn the run into PPO.

## 4. What happens in one update?

~~~text
sample grouped agent rollouts
  → execute search and record observations
  → compute task rewards
  → compute group-relative advantages
  → compute policy/reference log probabilities
  → evaluate clipped actor loss and KL term
  → backpropagate
  → optimizer updates actor parameters
  → save metrics, rollouts, and optionally a checkpoint
~~~

GRPO is critic-free in the sense used here: the group supplies the relative baseline instead of training a separate value critic for the update. That does not make rollouts cheap; every group member still requires generation and tool interaction.

## Common problems

### All rewards in a group are equal

This is a zero-variance group. It is valid data but gives no within-group preference. Log the ratio of such groups and inspect whether the model, reward, or task is too homogeneous.

### Rewards are all zero

Check final answer extraction, response clipping, and whether the reward function receives the native tool-response scaffolding it expects. Do not add shaping before the task-only path is understood.

### The loss changes but the model does not

Check actor gradients, optimizer steps, checkpoints, and a before/after held-out evaluation. A printed loss alone is not evidence of a policy update.

### The update is too large

Inspect KL statistics and response lengths before changing many knobs. Change one approved factor at a time.

## Exercises

1. Change one response in examples/04_grpo_concepts.py and predict the sign of its advantage.
2. Set all four responses to the same answer. What should happen to the standard deviation?
3. Find rollout.n and norm_adv_by_std_in_grpo in the active config.
4. Explain why computing z-scores without using them in an actor loss is not training.

## Checkpoint

You can continue when you can trace:

\[
\text{tokens}
\rightarrow \text{rollout}
\rightarrow \text{reward}
\rightarrow \text{advantage}
\rightarrow \text{ratio}
\rightarrow \text{loss}
\rightarrow \text{gradient}
\rightarrow \text{updated parameters}
\]

> **Research note.** The strict Qwen3-8B four-update run changed validation EM/F1 from 0.240/0.3635 to 0.320/0.4324. This is technical sanity evidence, not the accepted formal task-improvement result.

## Next

[Chapter 07 — Run a Real GRPO Smoke](07_grpo_smoke.md) launches one bounded update through the project’s actual verl path.
