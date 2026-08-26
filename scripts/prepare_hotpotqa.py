#!/usr/bin/env python3
"""Materialize a normalized, fingerprinted HotpotQA split as JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(row: dict[str, object]) -> dict[str, object]:
    context = row["context"]
    support = row["supporting_facts"]
    assert isinstance(context, dict) and isinstance(support, dict)
    return {
        "_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "supporting_facts": [
            [title, sentence_id]
            for title, sentence_id in zip(support["title"], support["sent_id"], strict=True)
        ],
        "context": [
            [title, sentences]
            for title, sentences in zip(context["title"], context["sentences"], strict=True)
        ],
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.limit}" if args.limit is not None else ""
    output = args.output_dir / f"hotpotqa_distractor_{args.split}{suffix}.jsonl"
    manifest = output.with_suffix(".manifest.json")
    if (output.exists() or manifest.exists()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    temporary = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in dataset:
            line = (json.dumps(normalize(row), ensure_ascii=False, sort_keys=True) + "\n").encode()
            handle.write(line)
            digest.update(line)
    temporary.replace(output)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": args.split,
        "rows": len(dataset),
        "dataset_fingerprint": dataset._fingerprint,
        "output": output.name,
        "bytes": output.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
