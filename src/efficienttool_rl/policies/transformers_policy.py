"""Local Hugging Face policy with Qwen-compatible tool templating."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..protocol import SEARCH_TOOL_SCHEMA


class TransformersToolPolicy:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        max_new_tokens: int = 256,
        seed: int = 42,
        temperature: float | None = None,
        top_p: float = 0.95,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda":
            cuda_index = self.device.index if self.device.index is not None else 0
            torch.cuda.set_device(cuda_index)
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        self.max_new_tokens = max_new_tokens
        if temperature is not None and temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.tools = [SEARCH_TOOL_SCHEMA]

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        materialized = [dict(message) for message in messages]
        template_args = {
            "tools": self.tools,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt = self.tokenizer.apply_chat_template(
                materialized,
                enable_thinking=False,
                **template_args,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(materialized, **template_args)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.temperature is not None,
        }
        if self.temperature is not None:
            generation_kwargs.update({"temperature": self.temperature, "top_p": self.top_p})
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
