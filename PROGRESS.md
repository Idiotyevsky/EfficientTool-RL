# Current Milestone

M5 — Cost-Aware GRPO preparation

# Status

BOUNDED COST-AWARE SANITY COMPLETE — the isolated success-gated reward has
completed an 8×4 smoke and a four-update Qwen3-8B sanity run over 32 total
training prompts (128 rollout trajectories overall).
The training and metadata paths work; the sample is still too small to claim
that cost-aware training improves tool efficiency.

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
- [x] Completed the Qwen3-8B vanilla GRPO baseline comparison on the fixed
  Natural Bridge-Hard secondary evaluation (200 examples); results recorded below.
- [x] Completed M5.0 offline counterfactual reward analysis on 400 stored
  Base/Step 62 Natural Bridge-Hard trajectories; output is reproducible and
  does not modify the vanilla reward or training configuration.
- [x] Implemented the isolated success-gated cost-aware reward and native verl
  adapter; the vanilla task-only reward remains unchanged.
- [x] Completed an 8×4 Qwen3-8B cost-aware smoke on four A6000 GPUs with a
  real rollout, non-zero advantage, actor update, and checkpoint.
- [x] Completed a bounded four-update Qwen3-8B cost-aware sanity run with
  rollout, validation, non-zero cost penalties, and a step-4 checkpoint;
  it used 32 total training prompts × 4 rollouts (128 trajectories), not 32
  prompts per optimizer step. No efficiency claim is made from this run.

# In Progress

- M5.1 fixed-policy cost-signal validation is complete: a Step-62 Qwen3-8B
  checkpoint produced 2,000 inference-only trajectories (500 groups) under
  the strict Hotpot-MT protocol. The next step is bounded cost-aware GRPO
  behavioral validation; no efficiency claim is made yet.
- The completed vanilla baseline remains the comparison point for upcoming
  cost-aware runs.
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

- No infrastructure blocker. A larger behavioral sanity run is needed before
  starting the lambda sweep or making a cost-efficiency claim.

# Latest Evidence

- Deterministic/unit tests: 68/68 passed (two upstream warnings).
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
- Prepared a fixed Natural Bridge-Hard secondary validation artifact outside
  Git: 200 bridge/hard validation rows without the strict answer-absence filter,
  using the same top-k=1, 384-token, three-executed-search limits. Its SHA-256
  is `1835707b46734751610d42a6f5ebba8bb3098789f841fede1c88a63b3cbf5fdc`.
- Natural Bridge-Hard vanilla GRPO, 200 examples, Base → Step 62:
  EM 0.325 → 0.515; F1 0.4203 → 0.6253; completion 0.935 → 0.975.
  Invalid action 0.1006 → 0.0017; executed search 1.335 → 1.960.
  Multi-search 0.315 → 0.860; useful search 0.965 → 1.445.
  Wasted search 0.370 → 0.515; tool efficiency 0.7228 → 0.7372.
  This is a vanilla baseline result; cost-aware training is now in bounded sanity validation.
- The completed Qwen3-8B vanilla GRPO checkpoint was evaluated on the fixed
  Natural Bridge-Hard secondary set; the exact comparison is summarized below.
- Strict train/validation parquet artifacts are materialized and fingerprinted
  outside Git; their SHA-256 values remain recorded above. The cost-aware
  implementation and bounded sanity artifacts are also outside Git; the step-4
  checkpoint is in the corresponding M5 run directory.
- Search-count analysis shows a long tail (rare episodes with 10–14 searches)
  and lower accuracy as search count increases; this is evidence for studying
  efficiency; the observed long tail informs cost-aware objective design.
- Official train artifact: 90,447 normalized records, SHA-256
  `89b6635152ea8f3038bdc9c7bac6708ceb718ec82b0a246fdc97ebab62a09ec2`;
  2,000-row parquet SHA-256
  `cb26c45e74c6fc80868c722ffecf9e3c92b8bed5effb38e2b65a876aa4b87b6f`.
- M5.0 offline design used lambdas 0, 0.025, 0.05, 0.10, 0.20, and 0.30;
  no correct-versus-wrong ranking inversion appeared through 0.20, while
  0.30 produced a small number of edge-case inversions. First smoke values
  are 0.025, 0.05, and 0.10; details are in docs/m5_cost_reward_offline.md.
- Cost-aware 8×4 smoke at λ=0.05 completed with 32 training rollouts,
  metadata available for all rows, non-zero advantage, grad norm 3.5012, and
  a real actor update. Its sampled wasted calls were not attached to positive
  task reward, so the mean cost penalty was 0; this verified success gating,
  not cost-sensitive behavior.
- Cost-aware 32-prompt × 4-update sanity at λ=0.05 completed 128 rollouts.
  Aggregate task reward/total reward was 0.3904/0.3898; executed/useful/wasted
  searches were 1.6641/1.3359/0.3281; cost metadata was available for 128/128
  rows; two rows received a non-zero penalty (maximum 0.05). Actor grad norms
  were 3.5362, 6.2044, 2.7231, and 1.6461. The 8-example validation slice
  changed EM/F1 from 0.250/0.3631 to 0.375/0.4688, while executed/useful/wasted
  searches stayed 1.50/1.25/0.25 and validation cost penalty stayed 0. This
  is bounded training evidence, not a final cost-aware result.

- Cost-signal audit on the 62 stored vanilla rollout batches (7,936
  trajectories and 1,984 groups) at λ=0.05: 1,008 rows (12.70%) had a
  non-zero penalty; 396 groups (19.96%) were cost-active; 137 groups
  (6.91%) changed normalized advantage; 327 equal-task pairs received a
  lower-waste tie-break; no task-order ranking flips occurred.
- The audit is cross-step stored-rollout evidence rather than a fixed-policy
  rollout-only evaluation; it supports a larger behavioral sanity run but
  does not establish a learned efficiency improvement.
- Fixed-policy rollout-only audit on the strict 2,000-row train artifact at
  lambda=0.05: EM/F1 were 0.6255/0.6939; completion was 98.4%; executed/useful/
  wasted searches were 2.0185/1.6200/0.3985; multi-search rate was 92.25%
  and 3+ search rate was 9.60%. Cost was non-zero for 207/2,000 rows
  (10.35%), active in 68/500 groups (13.60%), and changed normalized
  advantage in 14/500 groups (2.80%); no strict task-order ranking flips
  occurred. Artifact and config are stored outside Git at
  /home/zfs01/jiangjr/efficienttool-rl-audit-20260828/.

# Known Risks

- GPU availability is dynamic; recheck ownership and memory before every run.
- The project environment's `verl` is editable from a clean OPD checkout;
  local runner hooks must remain isolated from that checkout.
- Checkpoint writes are large (the four-GPU step-4 checkpoint was about 86 GiB)
  and belong on durable storage outside the Git checkout.
- Keep the source checkout separate from large model, data, and run artifacts.

# Next Actions

1. Preserve the vanilla and cost-aware checkpoints, rollout artifacts, and
   validation outputs outside the Git checkout.
2. Run a bounded cost-aware GRPO behavioral sanity experiment at lambda=0.05 and
   compare it with the fixed-policy audit.
3. Inspect whether the cost term changes group-relative signals without
   causing under-search or task-quality collapse.
4. Select a small lambda sweep only after the behavioral sanity run; report
   negative or null cost-aware effects explicitly.
