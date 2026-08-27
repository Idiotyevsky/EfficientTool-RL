# Current Milestone

M4 — Main Vanilla GRPO

# Status

IN PROGRESS — canonical evaluator reconciliation passed; canonical-loop
retraining is active before the final M4 claim is accepted.

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
- [x] Completed the formal 2,000-example vanilla GRPO run on two GPUs: 62/62
  updates, 7,936 rollout rows, final checkpoint and merged model saved outside
  the Git checkout.
- [x] Evaluated the final model and base model on the same fixed native verl
  held-out slice (validation indices 100–199), 100 examples each.
- [x] Unified the prompt, search schema, action parser, and first-action
  boundary across the local evaluator and native verl through a project-local
  canonical ToolAgentLoop adapter.
- [x] Re-ran matched native validation with the canonical loop for both the
  base model and the existing 2k checkpoint.

# In Progress

- The formal 2k vanilla GRPO experiment is running with the canonical loop so
  training-time and evaluation-time trajectory semantics are identical.

# Blockers

- No infrastructure blocker. The existing 2k checkpoint was trained before
  the canonical loop was installed, so its corrected evaluation is diagnostic
  evidence rather than the final end-to-end M4 claim.

# Latest Evidence

- Deterministic/unit tests: 47/47 passed (one upstream warning).
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
- Formal 2k run completed cleanly: 62/62 updates, final task-only reward
  0.5042 at step 62, overall rollout reward 0.3786, EM 0.3497, F1 0.4075,
  valid-answer rate 0.7088, and zero-variance group ratio 0.6840.
- Formal rollout behavior: valid search calls averaged 1.1604; assistant
  turns averaged 2.1035; generated model tokens averaged 338.8; malformed
  tool-call rate was 2.05%; duplicate-query episode rate was 2.72%.
- Independent local evaluator on fixed heldout100 (same evaluator and
  generation settings): base EM/F1 0.340/0.4356, final GRPO 0.380/0.4755;
  search calls 1.000→0.990, generated tokens 44.42→49.83, and invalid-action
  rate 0.50%→1.00%. This is promising but not yet the accepted M4 claim.
- Native verl evaluator on the same heldout100: base EM/F1 0.010/0.010,
  valid-answer rate 0.09; final GRPO EM/F1 0.040/0.0517, valid-answer rate
  0.12. This is retained as legacy-protocol diagnostic evidence.
- Canonical native evaluator on the same validation indices: base EM/F1
  0.350/0.4204 with valid-answer rate 0.85; existing final checkpoint
  0.400/0.4930 with valid-answer rate 0.83. Post-hoc malformed-call rate was
  0 for both, and search-call averages were 1.00 and 1.00.
- Canonical local evaluator on the same 100 examples: base EM/F1
  0.340/0.4220; existing final checkpoint 0.380/0.4775. These close results
  validate the cross-evaluator protocol alignment, but the checkpoint still
  needs canonical-loop retraining for a final M4 claim.
- Canonical-loop formal retraining launched on A6000-6 GPUs 0–1 with the
  approved 2,000-example, 62-update configuration. At the latest checkpoint
  it had reached 7/62 updates with rollout files through `6.jsonl` and no
  OOM or traceback; final metrics are intentionally pending.
- Search-count analysis shows a long tail (rare episodes with 10–14 searches)
  and lower accuracy as search count increases; this is evidence for studying
  efficiency, not yet justification for launching cost shaping.
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

1. Let the canonical-loop 2k vanilla GRPO run finish and preserve its logs.
2. Evaluate its held-out checkpoint against the canonical base protocol and
   inspect at least 20 successes and 20 failures.
3. Only after the M4 gate is accepted, decide whether M5 should penalize
   search calls, redundant queries, or token/trajectory cost.
