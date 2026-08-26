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
