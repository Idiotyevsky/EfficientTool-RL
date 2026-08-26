#!/usr/bin/env python3
"""Run one deterministic BM25 retrieval against a normalized HotpotQA record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from efficienttool_rl.data import load_hotpotqa
from efficienttool_rl.tools import BM25Search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-observation-tokens", type=int, default=256)
    args = parser.parse_args()

    examples = load_hotpotqa(args.data, split="validation")
    example = examples[args.index]
    search = BM25Search(
        example.passages,
        max_observation_tokens=args.max_observation_tokens,
    )
    results = search.search(example.question, top_k=args.top_k)
    returned_titles = {result.title for result in results}
    payload = {
        "example_id": example.example_id,
        "question": example.question,
        "query": example.question,
        "results": [result.to_dict() for result in results],
        "supporting_title_recall": (
            len(returned_titles.intersection(example.supporting_titles))
            / max(len(example.supporting_titles), 1)
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
