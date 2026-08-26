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
- [x] Implemented and tested the task-only `0.5 EM + 0.5 F1` reward.
- [x] Implemented and tested the verl search-tool adapter and parquet records.
- [x] Verified a bounded 8×4 rollout has distinct trajectories in every group.
- [x] Completed a full M3 plumbing run through validation, update, dumps, and checkpoint.
- [x] Fixed native verl reward parsing for tool-response and reasoning scaffolding.
- [x] M3 technical learning-signal gate passed on the reward-fixed smoke.

# In Progress

- Run the first staged vanilla GRPO experiment on the 500-example subset and
  collect post-hoc malformed-action diagnostics.

# Blockers

- No infrastructure blocker. Task-performance evidence is still insufficient
  for claims; M4 must establish the vanilla GRPO baseline before cost shaping.

# Latest Evidence

- Deterministic unit tests: 44/44 passed.
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
- M4 step 3/16 is running on A6000-6 GPU2. Its first 128 rollouts have
  malformed tool-call rate 1.30% (2/154), malformed episode rate 0.78%, and
  zero unknown-tool calls; the native parser warning is now measured from raw
  rollout text rather than inferred from log-line counts.
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

1. Finish the staged 500-prompt vanilla GRPO run with stored diagnostics.
2. Evaluate held-out EM/F1, search calls, turns, length, and malformed actions.
3. Scale only if held-out behavior remains interpretable.
