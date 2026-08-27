#!/usr/bin/env python3
"""Materialize normalized HotpotQA records in verl-compatible parquet format."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset

from efficienttool_rl.data import is_two_hop_candidate, load_hotpotqa
from efficienttool_rl.training import to_verl_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
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
    parser.add_argument(
        "--require-two-hop",
        action="store_true",
        help="Keep bridge rows whose question-level top-1 hop is incomplete.",
    )
    parser.add_argument("--max-observation-tokens", type=int, default=512)
    parser.add_argument("--max-top-k", type=int, default=3)
    parser.add_argument("--max-executed-search-calls", type=int, default=3)
    parser.add_argument("--data-source", default="hotpotqa_distractor")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.start_index < 0 or args.limit < 1:
        raise ValueError("start-index must be non-negative and limit must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")

    if (
        args.max_observation_tokens < 1
        or args.max_top_k < 1
        or args.max_executed_search_calls < 0
    ):
        raise ValueError(
            "observation/top-k limits must be positive and search budget non-negative"
        )

    examples = load_hotpotqa(args.input, split=args.split)
    if args.question_type != "all":
        examples = [example for example in examples if example.question_type == args.question_type]
    if args.levels:
        allowed_levels = set(args.levels)
        examples = [example for example in examples if example.level in allowed_levels]
    if args.require_two_hop:
        examples = [
            example
            for example in examples
            if is_two_hop_candidate(
                example,
                max_observation_tokens=args.max_observation_tokens,
            )
        ]
    selected = examples[args.start_index : args.start_index + args.limit]
    if len(selected) != args.limit:
        raise ValueError("requested range exceeds input dataset")
    records = [
        to_verl_record(
            example,
            index=args.start_index + offset,
            max_observation_tokens=args.max_observation_tokens,
            max_top_k=args.max_top_k,
            max_executed_search_calls=args.max_executed_search_calls,
            data_source=args.data_source,
        )
        for offset, example in enumerate(selected)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    Dataset.from_list(records).to_parquet(str(temporary))
    temporary.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "split": args.split,
        "start_index": args.start_index,
        "rows": len(records),
        "question_type": args.question_type,
        "levels": args.levels,
        "require_two_hop": args.require_two_hop,
        "max_observation_tokens": args.max_observation_tokens,
        "max_top_k": args.max_top_k,
        "max_executed_search_calls": args.max_executed_search_calls,
        "data_source": args.data_source,
        "output": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "sha256": digest,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
