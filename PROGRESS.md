# Current Milestone

Recipe comparison — Qwen3-8B Multi-turn Tool Agent RL

# Status

IN PROGRESS — infrastructure complete; two headline runs queued/running.

# Main Line

```text
Qwen3-8B Base
  -> vanilla GRPO (task-only reward)        [DONE - see experiments/baselines.md]
  -> GRPO + composite reward                [QUEUED - auto-starts after DAPO]
  -> DAPO (task-only reward)                [RUNNING - 3090-1, 4 GPUs, 62 steps]
  -> DAPO + composite reward                [optional, later]
```

All methods share one pipeline (Qwen3-8B, strict Hotpot-MT train 2000,
async vLLM multi-turn BM25 tool environment) and one held-out protocol
(Natural Bridge-Hard 200). Claim wording: compare *training recipes*
end-to-end; no component-level attribution is claimed.

# Extension (capped at ~20% of project effort): Agent Distillation

```text
Qwen3-8B RL Agent (teacher, from the main line)
  -> On-Policy Distillation -> Qwen3-1.7B (student)
```

- Teacher token-level supervision on student on-policy rollouts
  (`y ~ pi_student`, teacher provides `pi_teacher(.|s_t)`); no new reward
  design, no new benchmark, same Hotpot-MT environment and held-out
  evaluation (EM / F1 / completion / invalid action / multi-search).
- Comparison target: 1.7B Base vs 1.7B + OPD vs 8B Teacher. A 1.7B GRPO
  run is optional context (a 1.7B pilot and a 500/2000-prompt 1.7B GRPO
  run already exist from earlier milestones).
- Fallback: if 1.7B does not learn under OPD, switch the student to 4B
  rather than tuning endlessly.
- Gate: starts only after the main-line runs complete and are evaluated.

# Completed

- [x] Repository refactor: teaching site/tutorials removed; configs/
      scripts/src reorganized; unified evaluate.py / analyze_rollouts.py
      entries; 85/85 tests pass.
- [x] Vanilla GRPO baseline reproduced and documented
      (experiments/baselines.md): Natural Bridge-Hard 200,
      EM 32.5% -> 51.5%, F1 42.03% -> 62.53%, invalid action
      10.06% -> 0.17%, multi-search 31.5% -> 86.0%.
- [x] Composite reward (answer + saturating evidence coverage + format)
      implemented with configurable weights and offline distribution
      validation on stored trajectories
      (analysis/composite_reward_pilot/; corr(evidence, EM) 0.53-0.59,
      answer keeps ~76% of reward mass).
- [x] DAPO recipe integration on verl RayDAPOTrainer with three async
      multi-turn compatibility fixes (agent-loop keys in gen batches,
      manager-routed rollouts, true_reward_score default) plus a
      dynamic-sampling cap sized to this task's 60-70% zero-variance rate.
- [x] Historical cost-aware experiments and pre-refactor progress archived
      under docs/archive/.

# In Progress

- [ ] DAPO (task-only): Qwen3-8B, 4 GPUs (3090-1, even cards), 62 steps,
      screen session `etrl_dapo`, log
      `/home/jiangjr/etrl/runs/dapo_4gpu_main/run_screen.log`.
      Validation before training: reward 0.3164 (strict val-100 slice).
      Step 1 completed with actor/entropy 0.259, no traceback, and dynamic
      sampling recovering effective prompts across generation batches.

# Blockers

- None on the code path. Run output dirs are on 3090-1 local disk; archive
  checkpoints to ZFS after the run finishes (local disk is small).

# Latest Evidence

- 85/85 unit/integration tests pass (pytest -q).
- DAPO step-1 metrics printed (entropy 0.259, pg_clipfrac 0.0 at init),
  zero-variance filtering active (`num_prompt_in_batch` log), no
  `passages must be` environment errors after the tools_kwargs fix.
- Composite-reward offline pilot: see analysis/composite_reward_pilot/.

# Next Actions

1. DAPO run completes -> run unified held-out evaluation (Natural
   Bridge-Hard 200) via scripts/evaluate.py; fill experiments/results.md.
2. Auto-chain: GRPO + composite reward (`qwen8b_hotpot_mt_strict_composite`)
   starts when the DAPO run exits; same protocol and GPUs.
3. Keep narrative discipline: DAPO vs GRPO claims are recipe-level;
   unrun rows stay TBD; no fabricated numbers.

# Known Risks

- 3090-1 is shared: check GPU ownership and the /tmp/ray cluster registry
  (`ray_current_cluster`) before restarting anything; use the screen session.
- ZFS writes are slow (~60 MB/s); keep run dirs on 3090-1 local disk and
  archive checkpoints to ZFS only after a run finishes.
- The DAPO dynamic-sampling batch fill depends on the zero-variance rate;
  if effective-prompt recovery degrades in later steps, raise
  `algorithm.filter_groups.max_num_gen_batches` (currently 30) only after
  checking the observed per-step utilization logs.