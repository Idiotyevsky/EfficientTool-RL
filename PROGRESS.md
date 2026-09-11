# Current Milestone

Recipe comparison — Qwen3-8B Multi-turn Tool Agent RL

# Status

IN PROGRESS — composite GRPO and corrected assistant-only DAPO are running.

# Main Line

```text
Qwen3-8B Base
  -> vanilla GRPO (task-only reward)        [DONE - see experiments/baselines.md]
  -> GRPO + composite reward                [RUNNING - 3090-1, 4 GPUs, 62 steps]
  -> DAPO (task + assistant-only overlong) [RUNNING - 4090-1, 4 GPUs, 62 steps]
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
- [x] Fresh DAPO completed and was evaluated on Natural Bridge-Hard 200
      (EM 33.0%, F1 41.83%), but async `rm_scores` bypassed overlong shaping.
      Treat it as a DAPO recipe run without effective overlong reward.
- [x] Project-local assistant-only DAPO manager implemented and validated:
      91/91 tests pass; no third-party verl or policy-loss code changed.
- [x] Historical cost-aware experiments and pre-refactor progress archived
      under docs/archive/.

# In Progress

- [ ] GRPO + composite reward: Qwen3-8B, 4 GPUs, 62 steps in detached
      session `etrl_grpo_composite62`. Do not interrupt this run for the DAPO fix.
- [ ] Corrected Fresh DAPO: Qwen3-8B, 4 GPUs, 62 steps in detached
      session etrl_dapo_asstlen62_r7. Step 1 completed end to end. The
      4090 host uses PyTorch SDPA because its installed FlashAttention binary
      does not contain a valid SM89 kernel; research hyperparameters are unchanged.

- [x] Diagnosed the apparent "exit after checkpoint save": this was not a
      checkpoint-writer crash. DAPO variance filtering consumed 2-3 raw
      prompt batches for many optimizer updates; the one-epoch, 2,000-prompt
      dataloader was exhausted after update 44. The recipe then followed its
      normal post-loop path, saved `global_step_45`, and returned. Both DAPO
      configs now allow two data passes while `total_training_steps=62`
      remains the hard update cap.

# Blockers

- None on the code path. Run output dirs are on 3090-1 local disk; archive
  checkpoints to ZFS after the run finishes (local disk is small).

# Latest Evidence

- 91/91 unit/integration tests pass (pytest -q).
- Corrected Fresh DAPO step 1 completed on 128 trajectories: mean assistant
  length 464.04 versus total trajectory length 672.45, mean overlong penalty
  -0.1248, 45/128 nonzero penalties, and mean search count 1.219. Tool
  observations contributed a mean 208.41 tokens but were excluded from the
  overlong length.
- DAPO reached update 44 before data exhaustion and wrote a complete
  `global_step_45` checkpoint: four model shards, four optimizer shards,
  four extra-state shards, Hugging Face metadata, and dataloader state.
- Resume evidence on 2026-09-10: `Load from checkpoint folder`,
  `Setting global step to 45`, and per-rank model/optimizer/RNG/scheduler
  load records are present; GPU 0/2/4/6 are active under the detached screen.
- Dynamic sampling and invalid-action accounting remain active during the
  resumed rollouts; malformed model actions are logged rather than crashing.
- Composite-reward offline pilot: see analysis/composite_reward_pilot/.

# Next Actions

1. Let the active GRPO + composite reward run finish and evaluate it on
   Natural Bridge-Hard 200.
2. Let the corrected Fresh DAPO run finish, then evaluate its final checkpoint
   on Natural Bridge-Hard 200. Step-1 artifacts already verify assistant-only
   length accounting and overlong-penalty logging.
3. Keep narrative discipline: the earlier Fresh DAPO result did not exercise
   effective overlong shaping; DAPO vs GRPO claims remain recipe-level;
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