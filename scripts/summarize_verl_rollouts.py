#!/usr/bin/env python3
"""Aggregate native verl rollout files for reproducible M4/M5 analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from efficienttool_rl.evaluation.verl_analysis import (
    analyze_verl_behavior,
    analyze_verl_rollouts,
    read_verl_jsonl,
)


def _step_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _core_report(report: dict[str, Any]) -> dict[str, Any]:
    """Drop verbose per-group details while retaining gate diagnostics."""
    keep = {
        "episodes",
        "groups",
        "group_size_histogram",
        "mean_reward",
        "reward_std",
        "em_mean",
        "f1_mean",
        "valid_answer_rate",
        "literal_tool_call_count_mean",
        "valid_tool_call_count_mean",
        "malformed_tool_call_count_mean",
        "malformed_tool_call_episode_rate",
        "malformed_tool_call_rate",
        "unknown_tool_call_count_mean",
        "mean_group_reward_variance",
        "zero_variance_group_ratio",
        "nontrivial_reward_group_ratio",
        "all_groups_have_trajectory_diversity",
        "answer_tag_rate",
    }
    return {key: report[key] for key in keep if key in report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional local HF tokenizer for generated-token counts.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files = sorted(args.rollout_dir.glob("*.jsonl"), key=_step_number)
    if not files:
        raise SystemExit(f"no JSONL rollout files found in {args.rollout_dir}")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    all_rows: list[dict[str, Any]] = []
    by_step: list[dict[str, Any]] = []
    for path in files:
        rows = read_verl_jsonl(str(path))
        all_rows.extend(rows)
        by_step.append(
            {
                "step": _step_number(path),
                "file": str(path),
                "task": _core_report(analyze_verl_rollouts(rows)),
                "behavior": analyze_verl_behavior(rows, tokenizer=tokenizer),
            }
        )

    report = {
        "rollout_dir": str(args.rollout_dir),
        "rollout_files": [str(path) for path in files],
        "tokenizer": str(args.tokenizer) if args.tokenizer else None,
        "by_step": by_step,
        "overall": {
            "task": _core_report(analyze_verl_rollouts(all_rows)),
            "behavior": analyze_verl_behavior(all_rows, tokenizer=tokenizer),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
