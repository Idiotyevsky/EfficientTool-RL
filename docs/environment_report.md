# ToolAgentLab Reference Environment

This report records the validated software stack for local inference and the
verl/vLLM training path. Exact compatibility depends on GPU architecture; use a
matched CUDA, PyTorch, vLLM, and FlashAttention combination.

| Component | Validated version |
|---|---:|
| Python | 3.12.13 |
| PyTorch | 2.8.0 + CUDA 12.8 |
| Transformers | 4.57.1 |
| vLLM | 0.11.0 |
| verl | 0.7.0.dev0 |
| FlashAttention | 2.8.1 |
| datasets | 5.0.1 |

The validated verl checkout used upstream commit
`4532fd35ccfdde82adc918b265e4c964534e83d1`. SGLang is not required by the
current pipeline.

## Model and Data

- Qwen3-1.7B is used for inference and bounded training smoke tests.
- Qwen3-8B is used for the reported GRPO/DAPO experiments.
- Normalized HotpotQA and derived parquet files stay outside Git because they
  are generated data.
- Checkpoints and rollout artifacts also stay outside Git; reports record
  stable names and fingerprints.

## Reproduction

```bash
export ETRL_ROOT=/path/to/toolagentlab
export ETRL_MODEL=/path/to/Qwen3-1.7B
export ETRL_DATA_DIR=/path/to/prepared/parquet
export ETRL_RUN_DIR=/path/to/run-output
export VERL_CONFIG_PATH=/path/to/verl/verl/trainer/config

cd "$ETRL_ROOT"
pytest -q
python scripts/smoke_qwen_inference.py \
  --model "$ETRL_MODEL" --device cuda:0 --max-new-tokens 32 --seed 42
```

For long runs, use a persistent session or scheduler and inspect GPU process
ownership, free memory, disk capacity, and the target output directory before
launch. On GPUs whose installed FlashAttention build lacks a matching kernel,
use a verified PyTorch SDPA override rather than changing research
hyperparameters.
