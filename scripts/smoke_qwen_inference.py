#!/usr/bin/env python3
"""Run one bounded, local-only Qwen inference for the M0 environment gate."""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib import metadata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local model directory.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: M0_SMOKE_OK",
        help="Prompt used for the bounded generation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def render_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    if not tokenizer.chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cuda_index: int | None = None
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(cuda_index)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    load_seconds = time.perf_counter() - started

    rendered_prompt = render_prompt(tokenizer, args.prompt)
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generation_seconds = time.perf_counter() - generation_started
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if generated_ids.numel() == 0:
        raise RuntimeError("Generation completed without producing a token")

    result = {
        "status": "PASS",
        "model": str(model_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(cuda_index) if cuda_index is not None else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "seed": args.seed,
        "prompt": args.prompt,
        "generated_text": generated_text,
        "generated_tokens": int(generated_ids.numel()),
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "peak_gpu_memory_gib": (
            round(torch.cuda.max_memory_allocated(cuda_index) / (1024**3), 3)
            if cuda_index is not None
            else None
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
