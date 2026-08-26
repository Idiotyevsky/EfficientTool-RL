#!/usr/bin/env python3
"""Analyze stored agent trajectories without rerunning inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from efficienttool_rl.data import load_hotpotqa
from efficienttool_rl.evaluation.trajectory_analysis import analyze_trajectories, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, categorized = analyze_trajectories(
        read_jsonl(args.trajectories),
        load_hotpotqa(args.data, split="validation"),
    )
    (args.output_dir / "behavior_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "categorized_trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for item in categorized:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
