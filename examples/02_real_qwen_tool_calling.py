"""Run the first real Qwen tool-calling episode on a tiny local corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from efficienttool_rl.agent import AgentConfig, AgentRunner
from efficienttool_rl.data import Passage
from efficienttool_rl.policies import TransformersToolPolicy
from efficienttool_rl.protocol import SYSTEM_PROMPT
from efficienttool_rl.tools import BM25Search


def parse_args() -> argparse.Namespace:
    default_model = os.environ.get("ETRL_MODEL")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(default_model) if default_model else None,
        required=default_model is None,
        help="Local Qwen-compatible Hugging Face checkpoint.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-search-calls", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-observation-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use temperature sampling instead of greedy decoding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1 or args.max_turns < 1:
        raise ValueError("max-new-tokens and max-turns must be positive")
    if args.max_search_calls < 0 or args.top_k < 1:
        raise ValueError("max-search-calls must be non-negative and top-k positive")

    passages = [
        Passage(
            title="Ada Lovelace",
            text="Ada Lovelace wrote notes on Charles Babbage's Analytical Engine.",
        ),
        Passage(
            title="Charles Babbage",
            text="Charles Babbage designed the Analytical Engine, an early general-purpose computer.",
        ),
        Passage(
            title="Alan Turing",
            text="Alan Turing made foundational contributions to theoretical computer science.",
        ),
    ]
    search = BM25Search(
        passages,
        max_observation_tokens=args.max_observation_tokens,
    )

    def bounded_search(arguments: dict[str, object]) -> list[dict[str, object]]:
        supplied = dict(arguments)
        requested = supplied.get("top_k", args.top_k)
        if not isinstance(requested, int) or isinstance(requested, bool):
            raise ValueError("search.top_k must be an integer")
        supplied["top_k"] = min(requested, args.top_k)
        return search.tool(supplied)

    policy = TransformersToolPolicy(
        args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        temperature=0.8 if args.sample else None,
    )
    runner = AgentRunner(
        policy=policy,
        tools={"search": bounded_search},
        config=AgentConfig(
            max_turns=args.max_turns,
            max_tool_calls=args.max_search_calls,
        ),
        system_prompt=SYSTEM_PROMPT,
    )

    question = "Use search before answering: what machine did Ada Lovelace write notes about?"
    print("Real model episode (the policy below is not scripted).")
    print(f"Model: {args.model}")
    print(f"Question: {question}")

    episode = runner.run(question, episode_id="real-qwen-tool-call")
    for step in episode.steps:
        print(f"\nTurn {step.turn + 1} model output:")
        print(step.model_output)
        print("Parsed action:")
        print(json.dumps(step.action, indent=2, ensure_ascii=False))
        if step.observation is not None:
            print("Observation:")
            print(json.dumps(step.observation, indent=2, ensure_ascii=False))

    print("\nEpisode summary:")
    print(
        json.dumps(
            {
                "final_answer": episode.final_answer,
                "termination_reason": episode.termination_reason,
                "attempted_tool_calls": episode.attempted_tool_calls,
                "valid_tool_calls": episode.valid_tool_calls,
                "executed_search_calls": episode.executed_search_calls,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
