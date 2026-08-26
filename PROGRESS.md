# Current Milestone

M4 — Main Vanilla GRPO

# Status

IN PROGRESS

# Completed

- [x] Completed and accepted M0 and M1.
- [x] Normalized and fingerprinted HotpotQA distractor validation data.
- [x] Implemented deterministic, bounded BM25 search.
- [x] Implemented EM, token F1, behavior metrics, and trajectory analysis.
- [x] Froze the ReAct prompt after isolated prompt-development examples.
- [x] Evaluated 60 held-out examples and inspected 20 successes/20 failures.
- [x] Reproduced a 20-example run byte-for-byte.
- [x] Selected verl's native multi-turn ToolAgentLoop integration path.
- [x] Materialized and fingerprinted a 500-example training subset.
- [x] Downloaded the official HotpotQA distractor train split (90,447
  records) and verified its SHA-256; materialized reproducible 2,000- and
  5,000-example parquet subsets outside the Git checkout.
- [x] Implemented and tested the task-only `0.5 EM + 0.5 F1` reward.
- [x] Implemented and tested the verl search-tool adapter and parquet records.
- [x] Verified a bounded 8×4 rollout has distinct trajectories in every group.
- [x] Completed a full M3 plumbing run through validation, update, dumps, and checkpoint.
- [x] Fixed native verl reward parsing for tool-response and reasoning scaffolding.
- [x] M3 technical learning-signal gate passed on the reward-fixed smoke.
- [x] Verified a one-update, three-GPU vanilla GRPO smoke with rollout,
  validation, actor update, and world-size-3 checkpoint output.

# In Progress

- Run the formal 2,000-example vanilla GRPO experiment on two confirmed-free
  GPUs, then evaluate the final checkpoint on the fixed held-out slice.

# Blockers

- No infrastructure blocker. Task-performance evidence is still insufficient
  for claims; M4 must establish the vanilla GRPO baseline before cost shaping.

# Latest Evidence

- Deterministic/unit tests: 45/45 passed (one upstream warning).
- Held-out 60: EM 0.400, F1 0.506, completion 100%.
- Average search calls: 1.000; average turns: 2.017.
- Average supporting-title recall: 0.775.
- Repeated held-out20 trajectory and metric files are byte-identical.
- Pre-update 8×4 rollout: every group had four distinct trajectories; 3/8
  groups had non-zero reward variance and 5/8 were zero-variance.
- M3 256-token run: 32/32 training rewards zero, no final answer tags, and
  response clip ratio 0.5; retained under the ZFS run directory as failed
  zero-signal evidence.
- M3 reward-fixed run `m3_sanity_a6000_6_g2_reward_fix`: 32 rollouts,
  mean reward 0.03125, one non-zero reward group out of eight, advantage std
  0.8706, grad norm 2.6584, and verified actor parameter deltas. Validation
  task score remained 0; valid-answer rate changed 0 → 0.25.
- Staged 500-example M4 run completed 15/15 updates and 1,920 rollouts; mean
  task reward rose from 0.0563 at step 1 to 0.4139 at step 15. Post-hoc
  malformed-call rate was 1.67%, valid search calls averaged 1.199, and
  zero-variance groups were 70.2%. Its step-8 checkpoint gave held-out100
  EM/F1 0.360/0.450 versus base 0.340/0.436; this is intermediate evidence,
  not the final M4 claim.
- The three-GPU smoke completed in 6:04 with 30 prompts, world-size-3 model
  and optimizer shards, and no runtime error. The formal run is now
  `qwen1.7b_grpo_hotpotqa_2000_2gpu_seed42` on A6000-6 GPUs 0–1.
- Official train artifact: 90,447 normalized records, SHA-256
  `89b6635152ea8f3038bdc9c7bac6708ceb718ec82b0a246fdc97ebab62a09ec2`;
  2,000-row parquet SHA-256
  `cb26c45e74c6fc80868c722ffecf9e3c92b8bed5effb38e2b65a876aa4b87b6f`.
- M2 gate: PASS; M3 technical learning-signal gate: PASS. M4 task-performance
  gate remains open.

# Known Risks

- GPU availability is dynamic; recheck ownership and memory before every run.
- The project environment's `verl` is editable from a clean OPD checkout;
  local runner hooks must remain isolated from that checkout.
- Checkpoint writes are large (~20GB for actor plus optimizer) and belong on
  durable storage outside the Git checkout.
- Keep the source checkout separate from large model, data, and run artifacts.

# Next Actions

1. Monitor the formal 2,000-example vanilla GRPO run and retain its logs.
2. Summarize rollout behavior and evaluate held-out EM/F1, search calls,
   turns, length, and malformed actions.
3. Decide whether the M4 behavior supports cost-aware M5 or a narrower cost
   metric such as redundant queries/tokens.
