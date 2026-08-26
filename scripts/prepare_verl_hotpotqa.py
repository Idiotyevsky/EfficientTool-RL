#!/usr/bin/env python3
"""Materialize normalized HotpotQA records in verl-compatible parquet format."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset

from efficienttool_rl.data import load_hotpotqa
from efficienttool_rl.training import to_verl_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.start_index < 0 or args.limit < 1:
        raise ValueError("start-index must be non-negative and limit must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")

    examples = load_hotpotqa(args.input, split=args.split)
    selected = examples[args.start_index : args.start_index + args.limit]
    if len(selected) != args.limit:
        raise ValueError("requested range exceeds input dataset")
    records = [
        to_verl_record(example, index=args.start_index + offset)
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
