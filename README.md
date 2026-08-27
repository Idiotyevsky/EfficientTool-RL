# EfficientTool-RL

**Train LLM agents to search less, search better, and answer correctly with GRPO.**

EfficientTool-RL is a reproducible research project for cost-aware,
multi-turn tool-agent reinforcement learning. The MVP uses a Qwen3 model, a
deterministic local BM25 search tool, HotpotQA distractor passages, and
verl-style grouped rollouts. Vanilla task-only GRPO is validated before any
search-cost shaping is introduced.

## Status

M0–M3 are accepted: environment, agent protocol, ReAct baseline, and the
GRPO learning-signal sanity gate. The canonical 2,000-example vanilla GRPO
run is complete and archived. Strict Hotpot-MT pilots show genuine
multi-search behavior with Qwen3-8B: 31.5% of episodes execute at least two
searches, and exactly-two-search episodes reach 52.6% EM versus 9.5% after one
search. The strict vanilla GRPO sanity gate is next; final claims remain
`TBD`.

## Architecture

```text
HotpotQA passages → local BM25 search → multi-turn Agent loop
                                      ↓
                         grouped rollouts → GRPO reward/update
                                      ↓
                 EM/F1 + tool calls + tokens + trajectory analysis
```

The repository keeps tools, environment data, rewards, training, evaluation,
and analysis separate. Gold answers are available only to offline reward and
evaluation code, never to the search tool. The next experiment uses a
bridge-focused Hotpot-MT profile with one result per search and an explicit
three-search execution budget. Its strict candidate filter makes the first
question-level hop incomplete without exposing support metadata to the agent.
The Qwen3-8B pilot already produces useful second-hop behavior, so the next
step is vanilla GRPO under this fixed information structure; no search penalty
is used yet.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Prepare normalized data with `scripts/prepare_hotpotqa.py`, then materialize
verl records with `scripts/prepare_verl_hotpotqa.py`. Run the inference-only
baseline with:

```bash
python scripts/evaluate_react.py \
  --data data/hotpotqa_distractor_validation.jsonl \
  --model /path/to/Qwen3-1.7B \
  --output-dir outputs/react_hotpotqa
```

## GRPO experiment

Set `ETRL_ROOT`, `ETRL_MODEL`, `ETRL_DATA_DIR`, `ETRL_RUN_DIR`, and
`VERL_CONFIG_PATH` as shown in `.env.example`, install a compatible verl/vLLM
stack, then run:

```bash
python scripts/run_ppo_m3.py --config-name qwen1.7b_grpo
```

Every run must use a unique output directory and retain its resolved config,
logs, rollouts, and metrics. See [AGENTS.md](AGENTS.md),
[PROGRESS.md](PROGRESS.md), and [docs/m4_plan.md](docs/m4_plan.md) for gates,
reproducibility rules, and the approved experiment sequence.

For the Hotpot-MT pilot, prepare filtered records with `--question-type
bridge --levels medium hard --max-top-k 1 --max-observation-tokens 384
--max-executed-search-calls 3`, then run the local evaluator with matching
`--top-k 1 --max-top-k 1 --max-search-calls 3`.
