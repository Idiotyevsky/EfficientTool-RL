#!/usr/bin/env python3
"""Run one real-Qwen, multi-turn M1 episode against a deterministic fake search."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from efficienttool_rl.agent import AgentConfig, AgentRunner


class TransformersPolicy:
    def __init__(
        self,
        model_path: Path,
        device: str,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        self.device = torch.device(device)
        cuda_index = self.device.index if self.device.index is not None else 0
        if self.device.type == "cuda":
            torch.cuda.set_device(cuda_index)
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search deterministic local evidence.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        materialized = [dict(message) for message in messages]
        try:
            prompt = self.tokenizer.apply_chat_template(
                materialized,
                tools=self.tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                materialized,
                tools=self.tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fake_search(arguments: dict[str, object]) -> dict[str, object]:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("search requires a non-empty string query")
    return {
        "query": query,
        "results": [
            {
                "title": "M1 protocol evidence",
                "passage": "The required verification token is M1_AGENT_OK.",
                "required_next_action": "<answer>M1_AGENT_OK</answer>",
            }
        ],
    }


def main() -> None:
    args = parse_args()
    policy = TransformersPolicy(
        model_path=args.model.expanduser().resolve(),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    system_prompt = """You are a strict tool agent.
On the first turn, call the provided search tool exactly once.
Do not answer before receiving the tool observation.
After the observation, copy the `required_next_action` value exactly.
Final answers must use <answer>final text</answer>.
Emit exactly one action block per turn and no additional prose."""
    runner = AgentRunner(
        policy=policy,
        tools={"search": fake_search},
        config=AgentConfig(max_turns=3, max_tool_calls=1),
        system_prompt=system_prompt,
    )
    result = runner.run(
        "Use search to find the verification token, then answer with that token.",
        episode_id="m1-real-qwen-smoke",
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if not (
        result.termination_reason == "final_answer"
        and result.tool_calls == 1
        and result.final_answer == "M1_AGENT_OK"
    ):
        raise SystemExit("M1 real-Qwen smoke test failed acceptance criteria")


if __name__ == "__main__":
    main()
