#!/usr/bin/env python3
"""Run a bounded inference-only ReAct baseline on HotpotQA distractor data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from efficienttool_rl.agent import AgentConfig, AgentRunner, JsonlTrajectoryWriter
from efficienttool_rl.data import load_hotpotqa
from efficienttool_rl.evaluation import answer_metrics, summarize_episodes
from efficienttool_rl.policies import TransformersToolPolicy
from efficienttool_rl.protocol import SYSTEM_PROMPT
from efficienttool_rl.tools import BM25Search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--question-type",
        choices=("all", "bridge", "comparison"),
        default="all",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=("easy", "medium", "hard"),
        help="Optional difficulty filter applied before start-index/limit.",
    )
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-search-calls", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--max-top-k",
        type=int,
        help="Hard environment cap on results per search; defaults to --top-k.",
    )
    parser.add_argument("--max-observation-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.start_index < 0:
        raise ValueError("limit must be positive and start-index non-negative")
    if args.max_search_calls < 0 or args.top_k < 1:
        raise ValueError("max-search-calls must be non-negative and top-k must be positive")
    max_top_k = args.top_k if args.max_top_k is None else args.max_top_k
    if max_top_k < 1 or args.top_k > max_top_k:
        raise ValueError("max-top-k must be positive and at least top-k")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_hotpotqa(args.data, split="validation")
    if args.question_type != "all":
        examples = [example for example in examples if example.question_type == args.question_type]
    if args.levels:
        allowed_levels = set(args.levels)
        examples = [example for example in examples if example.level in allowed_levels]
    selected = examples[args.start_index : args.start_index + args.limit]
    if len(selected) != args.limit:
        raise ValueError("requested range exceeds the dataset")
    policy = TransformersToolPolicy(
        args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    trajectory_path = args.output_dir / "trajectories.jsonl"
    failure_path = args.output_dir / "failures.jsonl"
    writer = JsonlTrajectoryWriter(trajectory_path)
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for offset, example in enumerate(selected):
        search = BM25Search(
            example.passages,
            max_observation_tokens=args.max_observation_tokens,
        )

        def bounded_search(arguments: dict[str, object]) -> list[dict[str, object]]:
            supplied = dict(arguments)
            supplied["top_k"] = min(int(supplied.get("top_k", args.top_k)), max_top_k)
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
        episode = runner.run(example.question, episode_id=example.example_id)
        writer.append(episode)
        scores = answer_metrics(episode.final_answer or "", example.answer)
        generated_tokens = sum(policy.count_tokens(step.model_output) for step in episode.steps)
        record: dict[str, object] = {
            **episode.to_dict(),
            "reference_answer": example.answer,
            "exact_match": scores["exact_match"],
            "f1": scores["f1"],
            "generated_tokens": generated_tokens,
        }
        records.append(record)
        if not scores["exact_match"]:
            failures.append(record)
        print(
            json.dumps(
                {
                    "index": args.start_index + offset,
                    "id": example.example_id,
                    "em": scores["exact_match"],
                    "f1": scores["f1"],
                    "tool_calls": episode.tool_calls,
                    "termination": episode.termination_reason,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    behavior = summarize_episodes(records)
    metrics = {
        **behavior,
        "exact_match": sum(float(item["exact_match"]) for item in records) / len(records),
        "f1": sum(float(item["f1"]) for item in records) / len(records),
        "avg_generated_tokens": (
            sum(int(item["generated_tokens"]) for item in records) / len(records)
        ),
    }
    with failure_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(args.output_dir / "metrics.json", metrics)
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data.resolve()),
        "data_sha256": file_sha256(args.data),
        "model": str(args.model.resolve()),
        "device": args.device,
        "start_index": args.start_index,
        "limit": args.limit,
        "question_type": args.question_type,
        "levels": args.levels,
        "max_turns": args.max_turns,
        "max_search_calls": args.max_search_calls,
            "top_k": args.top_k,
            "max_top_k": max_top_k,
        "max_observation_tokens": args.max_observation_tokens,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    write_json(args.output_dir / "run_config.yaml", config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
