# Debug Log

## 2026-08-26 — CUDA memory-stat device argument

**Symptom:** The first smoke run failed before model loading with
`RuntimeError: Invalid device argument` in `torch.cuda.reset_peak_memory_stats`.

**Cause:** The selected PyTorch build rejected the explicit `torch.device`
object for the memory-stat call.

**Fix:** Resolve the CUDA index, set it explicitly, and pass the integer index
to CUDA memory APIs.

**Validation:** The corrected script passed syntax/CLI checks and later
completed inference in both the base and project environments.

**Impact:** No model was loaded and no experiment result was invalidated.

## 2026-08-26 — Shared verl environment FlashAttention ABI mismatch

**Symptom:** Qwen model import failed with an undefined C++ symbol from
`flash_attn_2_cuda`.

**Cause:** The pre-existing FlashAttention build was ABI-incompatible with the
environment's PyTorch build.

**Fix:** The shared environment was not modified. A validated OPD environment
with PyTorch 2.8.0+cu128 and FlashAttention 2.8.1 was cloned to the dedicated
project path.

**Validation:** All key imports passed in the new environment and Qwen3-1.7B
generated `M0_SMOKE_OK`.

**Impact:** The old shared `verl` environment is rejected for this project;
no valid experiment was invalidated.

## 2026-08-26 — Qwen M1 action-format mismatch

**Symptom:** Initial real-model episodes produced an unwrapped token or raw
JSON instead of one of the required XML-delimited actions.

**Cause:** The generic chat template did not include Qwen's native tool schema.
The model therefore had no template-level tool-call contract.

**Fix:** Pass the OpenAI-style `search` schema through the tokenizer's `tools=`
argument and retain strict parser validation plus corrective observations.

**Validation:** Qwen3-1.7B emitted one valid `<tool_call>` action, consumed the
structured observation, then emitted `<answer>M1_AGENT_OK</answer>`. The episode
ended with one tool call and zero invalid actions; 18 unit tests passed.

**Impact:** The two failed smoke trajectories are invalid as M1 evidence but
did not affect training or benchmark results.

## 2026-08-26 — Hugging Face Xet download stall

**Symptom:** HotpotQA download produced no progress for multiple bounded
observation windows and blocked inside `huggingface_hub.file_download.xet_get`.

**Cause:** The Xet transfer backend stalled on this network path.

**Fix:** Interrupt only the owned download process and retry with
`HF_HUB_DISABLE_XET=1`, using standard HTTP and the same dedicated ZFS cache.

**Validation:** Both train (90,447 rows) and validation (7,405 rows) cache
splits generated successfully. Validation was normalized and fingerprinted.

**Impact:** No experiment ran on partial data and no result was invalidated.

## 2026-08-26 — M3 startup compatibility fixes

**Symptoms:** The first vLLM launch lacked `ninja`; the next launch used a
Ray temporary path too long for a Unix socket; FlashInfer then failed because
the host CUDA 11.8 toolkit lacks `cuda/functional` and rejects GCC 13.

**Fix:** Added the Anaconda `ninja` path, shortened `RAY_TMPDIR`, and set
`VLLM_USE_FLASHINFER_SAMPLER=0`. The native PyTorch sampler is an approved
vLLM fallback; no upstream checkout was changed.

**Validation:** vLLM initialized on A6000 GPU2 and the local search tool ran.

## 2026-08-26 — M3 reward and dump integration

**Symptoms:** The experimental reward loop passed optional router/tokenizer
keywords not accepted by the callback, and verl's generation dump attempted
to JSON-encode Tensor-valued `true_reward_score` metadata.

**Fix:** The callback now ignores framework extension keywords. A local Ray
task runner converts Tensor/NumPy metadata before calling the unchanged verl
dump method; the shared OPD checkout remains untouched.

**Validation:** The full suite passed 40/40. A bounded run completed validation,
training, rollout dumping, and checkpoint saving.

## 2026-08-26 — M3 zero-signal smoke result

**Observed:** The 256-token bounded run produced 32/32 training rewards of
zero and no final answer tags; response clipping was 0.5. The run reached
`global_step_1` but had zero policy-gradient signal, so M3 did not pass.

**Decision:** Increase only `data.max_response_length` to 1024, matching the
approved starting scale, while keeping model, data, reward, temperature, and
seed fixed. The prior run is retained as failed evidence.

## 2026-08-26 — Native verl reward-format mismatch

**Symptom:** Native rollout dumps contained semantically correct outputs such
as `<answer>Badr Hari</answer>`, but the custom reward returned zero.

**Cause:** `solution_str` includes `<tool_response>` blocks, role markers, and
Qwen reasoning blocks. The original parser removed only `<tool_call>` blocks
and then required the entire trajectory to equal the answer block.

**Fix:** Strip tool responses, completed `<think>` blocks, and structural role
lines before applying the unchanged strict single-terminal-answer rule. A
replay of the retained patch run recovered 3 positive rewards from 32 rows.

**Validation:** The full suite passes 42/42. The native-dump analyzer and a
regression test cover this exact trajectory shape; gold answers remain outside
tool observations.

## 2026-08-26 — M3 reward-fixed smoke

The bounded run `m3_sanity_a6000_6_g2_reward_fix` used Qwen3-1.7B, eight
training prompts, group size four, seed 42, task-only reward, and a 1024-token
response cap. It produced mean reward 0.03125 (one of 32 rollouts scored 1),
non-zero advantage std 0.8706, `grad_norm=2.6584`, and a verified parameter
delta in saved actor tensors. Per-prompt grouping was reconstructed from the
native `input` field: one of eight groups had non-zero reward variance and all
groups had trajectory diversity.

Validation task score stayed at zero; valid-answer rate changed from 0 to
0.25 and five of eight serialized outputs changed. M3 therefore passes the
technical learning-signal sanity gate, but no task-improvement claim is
allowed. Vanilla GRPO must be evaluated at larger scale before cost shaping.

## 2026-08-26 — M4 malformed-action instrumentation

**Observed:** The native Hermes parser emits `Failed to decode tool call` for
malformed JSON and drops the call before the trajectory metadata is returned.
This is safe at the process level, but it is not a complete structured
`invalid_action` metric and can terminate a turn without an explicit marker.

**Action:** Kept the approved M4 run unchanged and added a project-side
post-hoc classifier over the serialized `solution_str`. It mirrors the native
parser boundary for malformed, valid, and unknown tool calls; it does not
alter rewards or rollouts.

**Validation:** On M4 training step 1 (128 rollouts, 32 groups), malformed
tool calls were 2/154 = 1.30% of raw calls and appeared in 1/128 = 0.78% of
episodes; unknown tool calls were zero. The analyzer is covered by the full
44-test suite. These values are diagnostics, not a task-performance claim.

## 2026-08-27 — Native/local evaluator protocol reconciliation

**Symptom:** On the same held-out indices, the legacy native evaluator scored
the base/final models at EM/F1 0.010/0.010 and 0.040/0.0517, while the local
evaluator scored them at 0.340/0.4356 and 0.380/0.4755.

**Cause:** The async native `ToolAgentLoop` delegated parsing to verl's Hermes
parser, which extracts tool calls but does not enforce the project's exactly
one-action protocol or stop generation at the first complete answer. Async
validation also does not consume `val_kwargs.max_tokens` as a per-turn limit;
the server derives a larger context remainder. Serialized outputs therefore
contained repeated/mixed answer blocks and malformed calls. Prompt wording and
tool-schema descriptions were duplicated and had also drifted.

**Fix:** Centralized the prompt and search schema in `protocol.py`, made the
local runner use the same first-action boundary, and added a thin
`CanonicalToolAgentLoop` subclass selected through `agent_loop_config_path`.
The adapter reuses verl tool execution and response masks, but validates each
generated turn with `parse_action`, rejects mixed/multiple actions, and trims a
valid generation at its first complete action. The upstream OPD/verl checkout
was not modified.

**Validation:** 47/47 tests pass. On the fixed 100-example slice, canonical
native base/final achieved EM/F1 0.350/0.4204 and 0.400/0.4930, with valid-answer
rates 0.85/0.83 and post-hoc malformed-call rate 0. The canonical local
evaluator achieved 0.340/0.4220 and 0.380/0.4775. The existing checkpoint was
trained before this adapter was installed; a canonical-loop retraining run is
required before accepting the final M4 claim.

## 2026-08-27 — Canonical validation resource conflict

**Symptom:** A duplicate final validation attempt failed during vLLM startup
because GPU 1 had only 2.3 GiB free.

**Cause:** An earlier project-owned validation session was still initializing
on that GPU; the duplicate attempt was launched before its delayed startup was
visible in the log.

**Action:** Kept the already-running original validation, recorded the
duplicate as failed infrastructure evidence, and did not terminate any
unrelated process. The original completed successfully; no metric from the
failed duplicate was used.

## 2026-08-27 — Executed-search accounting and Hotpot-MT scaffolding

The analyzer previously treated valid-looking `<tool_call>` tags as executed
searches. It now distinguishes literal attempts, parser-valid calls, and
successful native `<tool_response>` executions; local trajectories record the
same counters directly. When supporting titles are supplied only to offline
analysis, each executed search is marked useful if it adds a previously unseen
supporting title, otherwise wasted.

The loader now retains HotpotQA `type` and `level`. Native and local tool
configuration can enforce the same top-k cap, observation bound, and
per-trajectory executed-search budget. A `hotpot_multi_turn.yaml` profile and
filtered parquet-preparation options provide the next `bridge`-focused,
`top_k=1` pilot environment. The 52-test suite passes; no cost reward has
been added yet.

## 2026-08-27 — Canonical-loop formal retraining completed

The corrected end-to-end M4 run was launched with the shared prompt/schema,
the project-local `CanonicalToolAgentLoop`, Qwen3-1.7B, 2,000 training rows,
group size four, seed 42, and 62 updates on A6000-6 GPUs 0–1. At the first
progress check it advanced through 7/62 updates, produced sequential rollout
files and showed no OOM or traceback. It subsequently completed all 62
updates without OOM or traceback. Final native validation
reached EM/F1 0.390/0.5110 with valid-answer rate 0.94, and global_step_62
was merged outside the Git checkout.

## 2026-08-27 — Native executed-search budget lifecycle

The first budget implementation stored the counter inside a search-tool
instance. Native verl creates and releases that instance for each
tool call, so the counter would reset between calls and the configured
three-search limit was not actually trajectory-scoped.

The canonical loop now records successful search executions in per-trajectory
context and checks the per-record `max_executed_search_calls` before allowing
another search action. A focused regression test verifies that record-level
configuration takes precedence over the tool default.

Validation: the full suite passes 52/52. The fix is project-local and does not
modify the editable upstream verl checkout.

## 2026-08-27 — Strict Hotpot-MT ReAct pilots

The strict filter retained bridge-hard examples whose question-level BM25
top-1 result contains exactly one supporting title and whose answer is absent
from that passage and the question. With top-k=1 and a 384-token observation
bound, Qwen3-1.7B completed 200 examples at EM/F1 0.070/0.1499, average
executed searches 1.035, and multi-search rate 3.0%; the six second searches
were useful in five cases.

Qwen3-4B on the same fixed 200 examples reached EM/F1 0.145/0.2411, average
executed searches 1.185, multi-search rate 15.0%, and second-search usefulness
50.0%. The higher model size increases exploration, but both policies remain
one-search dominated. The 1.7B/4B results are diagnostic evidence only; no
strict GRPO run or cost reward has been launched.

## 2026-08-27 — Strict artifacts and Qwen3-8B pilot

The strict train/validation artifacts were materialized from the official
normalized HotpotQA split with bridge medium/hard filtering, question-level
top-1 incompleteness, top-k=1, a 384-token observation bound, and a
three-executed-search budget. The train artifact contains 2,000 rows
(`481774f211516ac0dde7f7287914b84e7a77a256e76478cb8ec5f4f4598ad820`);
the 100-row validation artifact contains
`91044f84aaccb5bd5bdfa6ec2970575e5d8bd1636dd88bd37b5bdeb40b1da8be`.

Qwen3-8B on the fixed 200-example strict ReAct pilot reached EM/F1
0.215/0.3344, average executed searches 1.345, multi-search rate 31.5%,
and second-search usefulness 0.6190. Exactly-two-search episodes reached
EM 0.5263 versus 0.0949 after one search. This is sufficient evidence to
enter a bounded vanilla GRPO sanity run; it is not a cost-aware result.

## 2026-08-27 — Strict GRPO startup retries

The first launch exited during Hydra composition because the remote
`VERL_CONFIG_PATH` was misspelled. The second launch reached Ray but failed
before training because its temporary directory made the Ray Unix socket
longer than the 107-byte platform limit. Both failures were retained in
separate ZFS run directories and produced no checkpoint.

The third launch initially failed when vLLM reported no available KV-cache
blocks at `gpu_memory_utilization=0.30`; the hybrid actor/rollout footprint
was already about 14.7 GiB per card. The strict config was raised to `0.50`,
and the retry used physical GPUs 0, 1, 3, and 4 because GPU2 and GPU7 were
owned by unrelated services. vLLM then initialized successfully and entered
validation/training without OOM.

## 2026-08-27 — Strict Qwen3-8B vanilla GRPO sanity gate

The bounded strict run used Qwen3-8B, 128 training prompts, group size four,
four task-only GRPO updates, the bridge medium/hard Hotpot-MT artifacts,
top-k=1, a 384-token observation bound, and a three-executed-search budget on
four A6000 GPUs. It completed rollout, actor update, validation at steps 0/2/4,
and FSDP checkpoint saving without OOM or traceback.

Across 512 training rollouts, mean reward was 0.3039 with reward standard
deviation 0.4383; 36.7% of groups had non-trivial reward variance and 63.3%
were zero-variance. Actor gradient norms were non-zero at all four updates
(4.04, 3.36, 2.81, 2.54), and the actor/optimizer checkpoint contains four
model shards and four optimizer shards. Validation changed from step 0 to
step 4: EM 0.240 to 0.320, F1 0.3635 to 0.4324, and valid-answer rate 0.91 to
0.96. Executed search calls were stable at 1.62 to 1.60.

Rollout behavior also confirmed the strict environment is genuinely
multi-turn: executed searches averaged 1.502, 43.9% of episodes used at
least two searches, and 65.8% of second searches added new supporting
evidence. The run passed the technical sanity gate. It does not establish a
full task-improvement claim; the 2,000-example strict vanilla run remains
required before any cost reward is introduced.

## 2026-08-27 — Strict 2,000-example vanilla GRPO launch

The formal Qwen3-8B strict vanilla run was launched from the corrected upstream
verl configuration path on A6000-1 physical GPUs 0, 1, 4, and 6. The first
launch attempt exited during Hydra composition because `VERL_CONFIG_PATH`
pointed at the project config directory instead of the upstream trainer config
directory; it produced no training output and is retained as startup-failure
evidence. A unique retry directory was used rather than overwriting it.

The active retry is
