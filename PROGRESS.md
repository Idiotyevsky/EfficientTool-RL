# Current Milestone

M4 — Main Vanilla GRPO

# Status

IN PROGRESS — the original canonical M4 run is complete and archived.
The bounded Qwen3-8B strict Hotpot-MT vanilla GRPO sanity gate has passed;
the full strict experiment is the next active run.

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
- [x] Completed the bounded Qwen3-8B strict Hotpot-MT vanilla GRPO sanity
  run: four updates, actor update, validation before/after, and checkpoint.

# In Progress

- The canonical-loop 2k vanilla GRPO experiment is complete: 62/62 updates,
  final native validation EM/F1 0.390/0.5110, and final checkpoint saved.
- Search statistics now distinguish attempted, valid, executed, useful, and
  wasted calls; the historical canonical evaluator has been replayed under
  the new executed-call definition.
- Hotpot-MT scaffolding is ready: official `type`/`level` metadata is retained,
  bridge/level filtering is available, and the native/local environment can
  enforce top-k and executed-search budgets consistently.
- Strict Hotpot-MT data is materialized: train 2,000 rows
  (SHA-256 `481774f211516ac0dde7f7287914b84e7a77a256e76478cb8ec5f4f4598ad820`)
  and validation 100 rows
  (SHA-256 `91044f84aaccb5bd5bdfa6ec2970575e5d8bd1636dd88bd37b5bdeb40b1da8be`).
- Qwen3-1.7B/4B/8B strict pilots reached multi-search rates 3.0%/15.0%/31.5%;
  the 8B pilot is the first strong enough candidate for strict vanilla GRPO.

# Blockers

- No infrastructure blocker. Original-environment M4 is complete; strict
  Hotpot-MT vanilla GRPO remains the active research experiment.

# Latest Evidence

- Deterministic/unit tests: 52/52 passed (two upstream warnings).
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
  validate the cross-evaluator protocol alignment. The canonical-loop
  retraining subsequently completed and is the accepted implementation for
  the original-environment M4 comparison.
- Canonical-loop formal retraining completed on A6000-6 GPUs 0–1 with the
  approved 2,000-example, 62-update configuration and no OOM or traceback.
- Its final native validation reached EM/F1 0.390/0.5110, valid-answer rate
  0.94, and task reward 0.4505; global_step_62 was saved and merged outside
  the Git checkout.
- Replaying the canonical held-out outputs with executed-response accounting
  gives base/final average executed searches of 0.92/0.91, versus 1.00/1.00
  raw valid-call tags; useful-search rates are not inferred for unrun
  experiments and remain tied to stored trajectory evidence.
- Strict Hotpot-MT pilot, Qwen3-1.7B, 200 bridge-hard candidates, question-
  level top-1 incomplete, search top-k=1: EM/F1 0.070/0.1499,
  average executed searches 1.035, `P(search>=2)=0.030`, and second-search
  usefulness 0.8333. The second hop is highly valuable but the open policy
  still answers after one search.
- Qwen3-4B on the same 200 strict candidates: EM/F1 0.145/0.2411,
  average executed searches 1.185, `P(search>=2)=0.15`, second-search
  usefulness 0.50, and `P(search>=3)=0.035`. Scaling improves exploration
  but remains one-search dominated.
- Qwen3-8B on the same fixed strict candidates: EM/F1 0.215/0.3344,
  average executed searches 1.345, `P(search>=2)=0.315`,
  `P(search>=3)=0.030`, second-search usefulness 0.6190, and
  successful multi-turn episode rate 0.15. Among examples with exactly two
  searches, EM was 0.5263 versus 0.0949 after one search; this supports
  genuine information-completion behavior without forcing a fixed number of
  calls.
- Strict Qwen3-8B vanilla GRPO sanity used 128 training prompts, group size
  four, four updates, and the strict 100-example validation slice. Across
  512 rollouts, mean reward was 0.3039, reward std 0.4383, non-trivial reward
  groups 36.7%, and zero-variance groups 63.3%; actor grad norms were
  non-zero at every step (4.04 → 2.54). Executed searches averaged 1.502,
  with 43.9% multi-search episodes and 65.8% useful second searches.
- The same validation protocol changed from step 0 to step 4 as follows:
  EM 0.240 → 0.320, F1 0.3635 → 0.4324, valid-answer rate 0.91 → 0.96;
  executed searches remained 1.62 → 1.60. This is technical sanity evidence,
  not a full task-improvement claim. The saved FSDP checkpoint is under the
  corresponding ZFS run directory outside Git.
- Strict train/validation parquet artifacts are materialized and fingerprinted
  outside Git; their SHA-256 values are recorded above. No strict full-run or
  cost-reward result is being claimed yet.
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

1. Launch the full strict Qwen3-8B vanilla GRPO run on the fixed 2,000/100
   train/validation artifacts with the corrected native executed-search budget.
2. Compare the final strict checkpoint with the fixed ReAct pilot using EM,
   F1, executed/useful/wasted searches, turns, tokens, and search-count
   distributions.
3. Accept or reject the strict M4 gate from stored held-out evidence; do not
   infer success from training reward alone.
4. Only after strict vanilla GRPO is accepted, implement and sweep the
   success-gated waste-aware cost reward.
