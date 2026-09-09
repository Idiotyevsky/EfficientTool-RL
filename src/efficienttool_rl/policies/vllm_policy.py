"""vLLM-backed policy adapter for fixed-policy held-out evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..protocol import SEARCH_TOOL_SCHEMA


class VLLMToolPolicy:
    """Small vLLM-backed Policy adapter for the existing AgentRunner."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        tensor_parallel_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_batched_tokens: int,
        max_num_seqs: int,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        self._sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
        )
        self._SamplingParams = SamplingParams
        self._temperature = temperature
        self._top_p = top_p
        self._max_new_tokens = max_new_tokens
        self._request_seed = seed
        self._request_index = 0
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.llm = LLM(
            model=str(model_path),
            tokenizer=str(model_path),
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            enforce_eager=True,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
        )

    def _render(self, messages: Sequence[Mapping[str, str]]) -> str:
        materialized = [dict(message) for message in messages]
        kwargs: dict[str, Any] = {
            "tools": [SEARCH_TOOL_SCHEMA],
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return self.tokenizer.apply_chat_template(
                materialized,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(materialized, **kwargs)

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        prompt = self._render(messages)
        # Different seeds preserve rollout diversity within each GRPO-style
        # group while keeping the complete run reproducible.
        params = self._SamplingParams(
            n=1,
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_new_tokens,
            seed=self._request_seed + self._request_index,
        )
        self._request_index += 1
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


__all__ = ["VLLMToolPolicy"]
