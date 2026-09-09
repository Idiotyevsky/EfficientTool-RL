# EfficientTool-RL

**Multi-turn Tool Agent Reinforcement Learning with Qwen3, verl and vLLM.**

Train a Qwen3-8B agent that calls a BM25 search tool across multiple turns,
evaluate it under a fixed held-out protocol, and compare mature RL
post-training recipes — **vanilla GRPO vs DAPO**, with a task-only or a
process-aware composite reward.

`Qwen3-8B` · `Multi-turn Tool Calling` · `GRPO / DAPO` · `verl` · `vLLM` · `FSDP`

---

## Highlights

- **Qwen3-8B agent policy** trained with verl's native multi-turn tool-agent
  rollout (async vLLM server, per-trajectory tool instances).
- **Deterministic BM25 search environment**: every trajectory gets its own
  index over that sample's HotpotQA distractor passages; gold answers and
  supporting titles are evaluation-only metadata, never in prompts.
- **GRPO baseline** and a **DAPO recipe** (dynamic sampling, token-level loss
  aggregation, asymmetric clipping, soft overlong handling) on the same
  pipeline — reusing verl's implementations, not reimplementations.
- **Process-aware composite reward**: answer + saturating evidence-coverage +
  format checks, with offline distribution validation before training.
- **Unified held-out evaluation** across methods: EM / F1 / completion /
  invalid-action / executed-useful-wasted searches / multi-search rate.
- **Reproducible**: seeded configs, data fingerprints (SHA-256), rollout and
  validation dumps per step, resolved Hydra config saved with every run.

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
  (`src/efficienttool_rl/verl/canonical_agent_loop.py`) that enforces the
  same strict one-action protocol as the local evaluator, terminates on
  invalid actions, and enforces the executed-search budget.
- **Trainer**: verl `RayPPOTrainer` (GRPO) / `RayDAPOTrainer` (DAPO recipe),
  FSDP full sharding + param/optimizer offload, colocated async vLLM
  rollout, Ray single-controller scheduling.
- **Reward**: file-based custom reward function loaded by verl
  (`src/efficienttool_rl/verl/reward_adapters/`), weights supplied through
  `custom_reward_function.reward_kwargs` in Hydra configs.

## Results

Qwen3-8B, Natural Bridge-Hard held-out set (200 examples), same fixed
evaluation protocol for every method. Full numbers and diagnostics:
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
vanilla GRPO run, **68.4% of rollout groups had zero reward variance** — all
four rollouts shared one reward, contributing no advantage signal.

## Training Algorithms

| Component | Vanilla GRPO | DAPO |
|---|---|---|
| Advantage | GRPO, group-mean normalized (`norm_adv_by_std_in_grpo=true`) | same |
| Dynamic sampling | off — zero-variance groups consume batch budget | on — `filter_groups` drops zero-variance groups and regenerates (`max_num_gen_batches=10`) |
| Loss aggregation | `token-mean` | `token-mean` (sample-level `seq-mean-token-mean` available by config) |
| Clipping | symmetric 0.2 | asymmetric: low 0.2 / high 0.28 |
| KL | `use_kl_loss=true`, coef 0.001 (low_var_kl) | off (`use_kl_loss=false`), per DAPO recipe |
| Overlong handling | truncation at `max_response_length` | + soft overlong buffer (len 256, factor 1.0) via the DAPO reward manager; validation always uses the pure task reward |

Composite reward (optional for both): `R = 0.8·answer + 0.15·evidence +
0.05·format` — document-level gold-evidence coverage of successful searches
(saturating, duplicate-insensitive) plus three binary protocol checks.
Offline validation and hacking checks:
[analysis/composite_reward_pilot/README.md](analysis/composite_reward_pilot/README.md).

## Quick Start

```bash
git clone https://github.com/Idiotyevsky/EfficientTool-RL.git
cd EfficientTool-RL
pip install -e ".[test,data,hf]"          # lightweight package + tests
PYTHONPATH=src python examples/01_tool_calling.py   # CPU: parse → act → observe
pytest -q                                  # 81 unit/integration tests
```

The RL stack is hardware-dependent; install a matching verl/vLLM/CUDA pair
(see `requirements.txt` and `docs/environment_report.md`). The project was
validated with verl 0.7.0.dev0, vLLM 0.11.0, PyTorch 2.8.0+cu128,
Transformers 4.57.1, Python 3.12.

## Data Preparation

```bash
python scripts/prepare_hotpotqa.py --output-dir "$ETRL_DATA_DIR"     # normalized HotpotQA JSONL
python scripts/prepare_verl_hotpotqa.py ...                          # verl parquet records
```

Outputs are fingerprinted (SHA-256) and recorded in `PROGRESS.md`. Strict
Hotpot-MT train/val parquet artifacts ship with manifests.

## Training

```bash
# Environment (adjust paths; run dirs must be on large storage)
export ETRL_MODEL=/path/to/Qwen3-8B
export ETRL_DATA_DIR=/path/to/efficienttool-rl-data
export ETRL_RUN_DIR=/path/to/efficienttool-rl-runs
export ETRL_ROOT=$PWD
export VERL_CONFIG_PATH=/path/to/verl/verl/trainer/config
export VLLM_USE_FLASHINFER_SAMPLER=0

# Vanilla GRPO (Qwen3-8B, strict Hotpot-MT, 4 GPUs)
python scripts/train_grpo.py --config-name qwen8b_hotpot_mt_strict

# GRPO + composite reward
python scripts/train_grpo.py --config-name qwen8b_hotpot_mt_strict_composite

# DAPO (dynamic sampling + clip-higher + overlong buffer)
python scripts/train_dapo.py --config-name qwen8b_hotpot_mt_strict

# Bounded smoke first (override anything via Hydra):
python scripts/train_grpo.py --config-name qwen1.7b_smoke
```

Every run stores the resolved config, per-step rollout dumps
(`rollouts/*.jsonl`), validation dumps, and console metrics under
`$ETRL_RUN_DIR/<experiment_name>/`.

## Evaluation

```bash
# Fixed-policy held-out evaluation (Transformers or vLLM backend)
python scripts/evaluate.py --data "$ETRL_DATA_DIR/verl_hotpotqa_mt_natural_bridge_hard_val_200.parquet" \
    --model /path/to/checkpoint --output results/base_nbh200 \
    --backend vllm --tensor-parallel-size 2 --rollouts-per-prompt 1

# Rollout / training diagnostics from stored dumps (zero-variance ratio,
# reward distribution, length stats, search behavior)
python scripts/analyze_rollouts.py --rollouts "$ETRL_RUN_DIR/<run>/rollouts" --output report.json

# Composite-reward component audit on stored trajectories
python scripts/validate_composite_reward.py --trajectories rows.jsonl \
    --examples "$ETRL_DATA_DIR/verl_hotpotqa_mt_strict_train_2000.parquet" \
    --output-dir audit_out
```

## Repository Structure

```text
EfficientTool-RL/
├── configs/
│   ├── grpo/            # GRPO runs: 1.7B smoke/500, 8B strict, composite variants
│   ├── dapo/            # DAPO recipe configs (task-only, composite)
│   └── tool/            # agent-loop registration + search tool schemas
├── src/efficienttool_rl/
│   ├── agent.py         # local multi-turn AgentRunner + trajectory schema
│   ├── protocol.py      # strict one-action protocol (parse, canonicalize)
│   ├── data/            # HotpotQA loaders + verl parquet reader
│   ├── rewards/         # task-only, cost-aware, composite rewards + parsing
│   ├── evaluation/      # metrics, search usage, verl dump analysis
│   ├── policies/        # Transformers / vLLM inference policies
│   ├── training/        # verl record conversion
│   └── verl/            # canonical agent loop, search tool, compat patches,
│                        #   reward adapters loaded by verl
├── scripts/             # train_grpo / train_dapo / evaluate / analyze_rollouts
├── examples/            # runnable minimal examples (CPU-friendly)
├── experiments/         # baselines.md + results.md (stored evidence only)
├── tests/               # 81 unit/integration tests
├── analysis/            # failure analysis + composite-reward pilot reports
├── docs/                # environment report, debug log, milestone plans
└── PROGRESS.md          # milestone status + latest evidence
```

## Reproducibility

- Every config is Hydra-managed; runs save the fully resolved config.
- Datasets are fingerprinted (SHA-256 in manifests and `PROGRESS.md`).
- Seeds are pinned (data shuffle, rollout sampling, per-request vLLM seeds).
- Rollout and validation dumps per step allow offline recomputation of every
  reported metric (`scripts/analyze_rollouts.py`, `scripts/evaluate.py`).
- Results tables in `experiments/` only contain numbers backed by stored
  artifacts; unrun comparisons stay `TBD`.

## License

No project-level license has been added yet; upstream dependencies (Qwen,
verl, vLLM, Transformers, HotpotQA) remain under their own licenses. Do not
assume redistribution rights until a `LICENSE` file is added.
