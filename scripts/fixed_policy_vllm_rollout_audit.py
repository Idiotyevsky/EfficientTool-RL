#!/usr/bin/env python3
"""Run inference-only fixed-policy trajectories with vLLM tensor parallelism.

The policy is loaded once across the requested GPUs.  Each trajectory still
uses the production ``AgentRunner`` and strict local BM25 tool, so this script
does not start an optimizer, trainer, or checkpoint writer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from efficienttool_rl.agent import AgentConfig, AgentRunner
from efficienttool_rl.evaluation.metrics import answer_metrics
from efficienttool_rl.protocol import SEARCH_TOOL_SCHEMA, SYSTEM_PROMPT
from fixed_policy_rollout_audit import (
    _local_search_usage,
    load_examples,
)
from efficienttool_rl.tools import BM25Search


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-search-calls", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-top-k", type=int, default=1)
    parser.add_argument("--max-observation-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.start_index < 0 or args.limit < 1:
        raise ValueError("start-index must be non-negative and limit must be positive")
    if args.rollouts_per_prompt < 1 or args.tensor_parallel_size < 1:
        raise ValueError("rollouts-per-prompt and tensor-parallel-size must be positive")
    if args.max_turns < 1 or args.max_search_calls < 0:
        raise ValueError("invalid turn or search budget")
    if args.top_k < 1 or args.max_top_k < args.top_k:
        raise ValueError("max-top-k must be at least top-k and both must be positive")
    if args.max_observation_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("token limits must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be positive and top-p must be in (0, 1]")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")

    examples = load_examples(args.data, args.split)
    selected = examples[args.start_index : args.start_index + args.limit]
    if len(selected) != args.limit:
        raise ValueError("requested range exceeds the dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing[str(row["trajectory_id"])] = row
    expected_ids = {
        f"{example.example_id}__rollout_{rollout_index}"
        for example in selected
        for rollout_index in range(args.rollouts_per_prompt)
    }
    if expected_ids.issubset(existing):
        print(json.dumps({"status": "already_complete", "rows": len(expected_ids)}))
        return

    policy = VLLMToolPolicy(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )

    completed = len(existing)
    with args.output.open("a", encoding="utf-8") as handle:
        for example_offset, example in enumerate(selected):
            search = BM25Search(
                example.passages,
                max_observation_tokens=args.max_observation_tokens,
            )

            def bounded_search(arguments: dict[str, object]) -> list[dict[str, object]]:
                supplied = dict(arguments)
                top_k = supplied.get("top_k", args.top_k)
                if isinstance(top_k, int) and not isinstance(top_k, bool):
                    supplied["top_k"] = min(top_k, args.max_top_k)
                return search.tool(supplied)

            runner = AgentRunner(
                policy,
                tools={"search": bounded_search},
                config=AgentConfig(
                    max_turns=args.max_turns,
                    max_tool_calls=args.max_search_calls,
                ),
                system_prompt=SYSTEM_PROMPT,
            )
            for rollout_index in range(args.rollouts_per_prompt):
                trajectory_id = f"{example.example_id}__rollout_{rollout_index}"
                if trajectory_id in existing:
                    continue
                episode = runner.run(example.question, episode_id=trajectory_id)
                scores = answer_metrics(episode.final_answer or "", example.answer)
                usage = _local_search_usage(episode, example.supporting_titles)
                record: dict[str, Any] = {
                    "trajectory_id": trajectory_id,
                    "input": example.question,
                    "episode_id": episode.episode_id,
                    "reference_answer": example.answer,
                    "task_reward": 0.5 * scores["exact_match"] + 0.5 * scores["f1"],
                    "exact_match": scores["exact_match"],
                    "f1": scores["f1"],
                    "valid_answer": float(bool(episode.final_answer)),
                    "wasted_search_calls": usage["wasted_search_calls"],
                    "useful_search_calls": usage["useful_search_calls"],
                    "executed_search_calls": usage["executed_search_calls"],
                    "attempted_tool_calls": episode.attempted_tool_calls,
                    "valid_tool_calls": episode.valid_tool_calls,
                    "invalid_actions": episode.invalid_actions,
                    "generated_tokens": sum(
                        policy.count_tokens(step.model_output) for step in episode.steps
                    ),
                    "turns": len(episode.steps),
                    "termination_reason": episode.termination_reason,
                    "trajectory": episode.to_dict(),
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                existing[trajectory_id] = record
                completed += 1
                print(
                    json.dumps(
                        {
                            "global_index": args.start_index + example_offset,
                            "example_id": example.example_id,
                            "rollout": rollout_index,
                            "completed": completed,
                            "em": scores["exact_match"],
                            "f1": scores["f1"],
                            **usage,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if args.config_output:
        _write_json(
            args.config_output,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data": str(args.data.resolve()),
                "data_sha256": _sha256(args.data),
                "model": str(args.model.resolve()),
                "start_index": args.start_index,
                "limit": args.limit,
                "rollouts_per_prompt": args.rollouts_per_prompt,
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_turns": args.max_turns,
                "max_search_calls": args.max_search_calls,
                "top_k": args.top_k,
                "max_top_k": args.max_top_k,
                "max_observation_tokens": args.max_observation_tokens,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "inference_only": True,
            },
        )


if __name__ == "__main__":
    main()
