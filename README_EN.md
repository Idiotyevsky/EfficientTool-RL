# EfficientTool-RL

**Multi-turn Tool Agent Reinforcement Learning with Qwen3, verl and vLLM.**

Train a Qwen3-8B agent that calls a BM25 search tool across multiple turns,
evaluate it under a fixed held-out protocol, and compare mature RL
post-training recipes — **vanilla GRPO vs DAPO**, with a task-only or a
process-aware composite reward.

`Qwen3-8B` · `Multi-turn Tool Calling` · `GRPO / DAPO` · `verl` · `vLLM` · `FSDP`

[中文说明](README.md)

---

## Highlights

- **Qwen3-8B agent policy** trained with verl's native multi-turn tool-agent
  rollout (async vLLM server, per-trajectory tool instances).
- **Deterministic BM25 search environment**: per-trajectory indexes over each
  sample's HotpotQA distractor passages; gold answers and supporting titles
  are evaluation-only metadata, never in prompts.
- **GRPO baseline** and a **DAPO recipe** (dynamic sampling, token-level loss
  aggregation, asymmetric clipping, soft overlong handling) on the same
  pipeline — reusing verl's implementations, not reimplementations.
- **Process-aware composite reward**: answer + saturating evidence-coverage +
  format checks, validated offline before training.
- **Unified held-out evaluation** across methods: EM / F1 / completion /
  invalid-action / executed-useful-wasted searches / multi-search rate.
- **Reproducible**: seeded configs, data fingerprints (SHA-256), per-step
  rollout/validation dumps, resolved Hydra config saved with every run.

## Architecture

```text
                ┌──────────────────────── multi-turn episode ───────────────────────┐
                │                                                                   │
question ──▶ Qwen3-8B policy ──▶ <tool_call>{search} ──▶ BM25 tool (per-trajectory index)
   ▲              ▲                                              │
   │              │                                    <tool_response> observation
   │              │                                              │
   │              └────────── observation re-enters context ──────┘
   │                                     │
   │                        terminal <answer> answer span
   │                                     │
GRPO / DAPO update ◀── 0.5·EM + 0.5·F1 (+ evidence & format for composite)
```

- **Agent loop**: verl `ToolAgentLoop` with a project-local canonical adapter
  enforcing the same strict one-action protocol as the local evaluator,
  terminating on invalid actions and enforcing the executed-search budget.
- **Trainer**: verl `RayPPOTrainer` (GRPO) / `RayDAPOTrainer` (DAPO recipe),
  FSDP full sharding + param/optimizer offload, colocated async vLLM rollout.
- **Reward**: file-based custom reward functions loaded by verl
  (`src/efficienttool_rl/verl/reward_adapters/`), weights via
  `custom_reward_function.reward_kwargs`.

## Results

Qwen3-8B, Natural Bridge-Hard held-out set (200 examples), identical
evaluation protocol for every method. Details:
[experiments/baselines.md](experiments/baselines.md),
[experiments/results.md](experiments/results.md).

| Method | Reward | EM | F1 | Completion | Invalid action | Executed searches | Multi-search |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | — | 32.5% | 42.03% | 93.5% | 10.06% | 1.335 | 31.5% |
| Vanilla GRPO (step 62) | task-only | **51.5%** | **62.53%** | 97.5% | 0.17% | 1.960 | 86.0% |
| GRPO + composite reward | 0.8/0.15/0.05 | TBD | TBD | TBD | TBD | TBD | TBD |
| DAPO (task-only) | task-only | TBD | TBD | TBD | TBD | TBD | TBD |

Vanilla task-only GRPO improves task quality and encourages more multi-step
retrieval; both useful (0.965 → 1.445) and wasted (0.370 → 0.515) searches
increase — this is **not** a cost-aware result.

Training-signal motivation for DAPO on this task: on the 2,000-prompt
vanilla GRPO run, **68.4% of rollout groups had zero reward variance**.

## Training Algorithms

| Component | Vanilla GRPO | DAPO |
|---|---|---|
| Advantage | GRPO, group-mean normalized | same |
| Dynamic sampling | off | on — `filter_groups` drops zero-variance groups and regenerates (`max_num_gen_batches=10`) |
| Loss aggregation | `token-mean` | `token-mean` (`seq-mean-token-mean` available by config) |
| Clipping | symmetric 0.2 | asymmetric: low 0.2 / high 0.28 |
| KL | `use_kl_loss=true`, coef 0.001 | off (`use_kl_loss=false`), per DAPO recipe |
| Overlong handling | truncation at `max_response_length` | + soft overlong buffer (len 256, factor 1.0) via the DAPO reward manager; validation uses the pure task reward |

Composite reward (optional for both): `R = 0.8·answer + 0.15·evidence +
0.05·format` — document-level gold-evidence coverage of successful searches
(saturating, duplicate-insensitive) plus three binary protocol checks.
Offline validation: [analysis/composite_reward_pilot/README.md](analysis/composite_reward_pilot/README.md).

## Quick Start

```bash
git clone https://github.com/Idiotyevsky/EfficientTool-RL.git
cd EfficientTool-RL
pip install -e ".[test,data,hf]"          # lightweight package + tests
PYTHONPATH=src python examples/01_tool_calling.py   # CPU: parse → act → observe
pytest -q                                  # 81 unit/integration tests
```

The RL stack is hardware-dependent; install a matching verl/vLLM/CUDA pair.
Validated with verl 0.7.0.dev0, vLLM 0.11.0, PyTorch 2.8.0+cu128,
Transformers 4.57.1, Python 3.12.

## Training

```bash
export ETRL_MODEL=/path/to/Qwen3-8B
export ETRL_DATA_DIR=/path/to/efficienttool-rl-data
export ETRL_RUN_DIR=/path/to/efficienttool-rl-runs
export ETRL_ROOT=$PWD
export VERL_CONFIG_PATH=/path/to/verl/verl/trainer/config
export VLLM_USE_FLASHINFER_SAMPLER=0

python scripts/train_grpo.py --config-name qwen8b_hotpot_mt_strict            # vanilla GRPO
python scripts/train_grpo.py --config-name qwen8b_hotpot_mt_strict_composite  # + composite reward
python scripts/train_dapo.py --config-name qwen8b_hotpot_mt_strict            # DAPO
python scripts/train_grpo.py --config-name qwen1.7b_smoke                     # bounded smoke
```

Every run stores the resolved config, per-step rollout dumps, validation
dumps, and console metrics under `$ETRL_RUN_DIR/<experiment_name>/`.

## Evaluation

```bash
python scripts/evaluate.py --data <heldout.parquet> --model <ckpt> \
    --output results/base_nbh200 --backend vllm --tensor-parallel-size 2

python scripts/analyze_rollouts.py --rollouts "$ETRL_RUN_DIR/<run>/rollouts" --output report.json

python scripts/validate_composite_reward.py --trajectories rows.jsonl \
    --examples "$ETRL_DATA_DIR/verl_hotpotqa_mt_strict_train_2000.parquet" --output-dir audit_out
```

## Repository Structure

See the Chinese README for the annotated tree; the layout is identical:
`configs/{grpo,dapo,tool}/`, `src/efficienttool_rl/{agent,rewards,evaluation,
policies,training,verl,...}`, `scripts/`, `examples/`, `experiments/`,
`tests/`, `analysis/`, `docs/`, `PROGRESS.md`.

## Reproducibility

Hydra-resolved configs saved per run; SHA-256 dataset fingerprints; pinned
seeds; per-step rollout/validation dumps for offline metric recomputation;
results tables only contain numbers backed by stored artifacts (`TBD` marks
unrun comparisons).

## License

No project-level license yet; upstream dependencies remain under their own
licenses. Do not assume redistribution rights until a `LICENSE` file exists.
