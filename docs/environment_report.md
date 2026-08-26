# M0 Environment Report

**Recorded:** 2026-08-26 (Asia/Shanghai)
**Milestone:** M0 — Environment Reconnaissance
**Sol gate decision:** PASS

## Reference Environment

The project was validated on a Linux CUDA server with an NVIDIA RTX A6000
class GPU and sufficient host memory for a 1.7B model plus vLLM rollout. GPU
selection is workload-dependent; always inspect ownership before a long run.

| Component | Validated version |
|---|---:|
| Python | 3.12.13 |
| PyTorch | 2.8.0 + CUDA 12.8 |
| Transformers | 4.57.1 |
| vLLM | 0.11.0 |
| verl | 0.7.0.dev0 |
| FlashAttention | 2.8.1 |
| datasets | 5.0.1 |

SGLang was not installed during M0; vLLM is the initial rollout engine. The
verl integration was tested against upstream commit
`4532fd35ccfdde82adc918b265e4c964534e83d1`. Use a compatible verl/vLLM/CUDA
combination rather than assuming these exact versions work on every GPU.

## Model and Data

The smoke test used Qwen3-1.7B loaded from a local Hugging Face-format
checkpoint. The normalized HotpotQA distractor data and derived parquet files
are intentionally kept outside Git because they are large generated inputs.
Their SHA-256 fingerprints are recorded in the milestone and run reports.

## Reproduction

```bash
export PROJECT_ROOT=/path/to/EfficientTool-RL
export MODEL_PATH=/path/to/Qwen3-1.7B
export DATA_DIR=/path/to/efficienttool-rl-data
export RUN_DIR=/path/to/efficienttool-rl-runs
export VERL_CONFIG_PATH=/path/to/verl/verl/trainer/config

cd "$PROJECT_ROOT"
python -m pytest -q
python scripts/smoke_qwen_inference.py \
  --model "$MODEL_PATH" --device cuda:0 --max-new-tokens 32 --seed 42
```

For vLLM runs, set `VLLM_USE_FLASHINFER_SAMPLER=0` when the installed CUDA
toolkit cannot compile FlashInfer. Use a persistent shell such as `screen` or
the host scheduler for long jobs, and write logs/checkpoints under `RUN_DIR`.

## M0 Gate Evidence

Qwen inference completed with the expected smoke output, core Torch/
Transformers/vLLM/verl imports passed, and no shared process was terminated or
modified. M0 is accepted; the next gate is the minimal multi-turn Agent loop.
