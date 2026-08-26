#!/usr/bin/env python3
"""Analyze a stored native verl rollout dump without rerunning inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from efficienttool_rl.evaluation.verl_analysis import analyze_verl_rollouts, read_verl_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_verl_rollouts(read_verl_jsonl(str(args.rollouts)))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
