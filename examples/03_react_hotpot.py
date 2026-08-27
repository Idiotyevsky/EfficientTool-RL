"""Run one or more bounded local ReAct episodes on normalized HotpotQA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from efficienttool_rl.agent import AgentConfig, AgentRunner
from efficienttool_rl.data import load_hotpotqa
from efficienttool_rl.evaluation import answer_metrics
from efficienttool_rl.policies import TransformersToolPolicy
from efficienttool_rl.protocol import SYSTEM_PROMPT
from efficienttool_rl.tools import BM25Search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-search-calls", type=int, default=3)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Per-search result count and hard cap (use 1 for strict Hotpot-MT).",
    )
    parser.add_argument("--max-observation-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable temperature sampling; default generation is greedy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_index < 0 or args.limit < 1:
        raise ValueError("start-index must be non-negative and limit must be positive")
    if args.max_turns < 1 or args.max_search_calls < 0:
        raise ValueError("max-turns must be positive and max-search-calls non-negative")
    if args.top_k < 1 or args.max_observation_tokens < 1:
        raise ValueError("top-k and max-observation-tokens must be positive")

    examples = load_hotpotqa(args.data, split="validation")
    selected = examples[args.start_index : args.start_index + args.limit]
    if len(selected) != args.limit:
        raise ValueError("requested range exceeds the dataset")

    policy = TransformersToolPolicy(
        args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        temperature=0.8 if args.sample else None,
    )

    for offset, example in enumerate(selected):
        search = BM25Search(
            example.passages,
            max_observation_tokens=args.max_observation_tokens,
        )

        def bounded_search(arguments: dict[str, object]) -> list[dict[str, object]]:
            supplied = dict(arguments)
            requested = supplied.get("top_k", args.top_k)
            if not isinstance(requested, int) or isinstance(requested, bool):
                raise ValueError("search.top_k must be an integer")
            supplied["top_k"] = min(requested, args.top_k)
            return search.tool(supplied)

        runner = AgentRunner(
            policy=policy,
            tools={"search": bounded_search},
            config=AgentConfig(
                max_turns=args.max_turns,
                max_tool_calls=args.max_search_calls,
            ),
            system_prompt=SYSTEM_PROMPT,
        )
        episode = runner.run(example.question, episode_id=example.example_id)
        scores = answer_metrics(episode.final_answer or "", example.answer)

        print(
            json.dumps(
                {
                    "index": args.start_index + offset,
                    "id": example.example_id,
                    "question_type": example.question_type,
                    "level": example.level,
                    "question": example.question,
                    "answer": episode.final_answer,
                    "reference": example.answer,
                    "exact_match": scores["exact_match"],
                    "f1": scores["f1"],
                    "turns": len(episode.steps),
                    "attempted_tool_calls": episode.attempted_tool_calls,
                    "valid_tool_calls": episode.valid_tool_calls,
                    "executed_search_calls": episode.executed_search_calls,
                    "termination_reason": episode.termination_reason,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
