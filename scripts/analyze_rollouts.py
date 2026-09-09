#!/usr/bin/env python3
"""Analyze stored native verl rollout dumps (single file or run directory).

Single-file mode replays task reward and per-prompt GRPO group statistics,
including zero-variance group diagnostics. Directory mode additionally
aggregates per-step task and behavior reports across all rollout dumps.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from efficienttool_rl.data import load_verl_examples
from efficienttool_rl.evaluation.verl_analysis import (
    analyze_verl_behavior,
    analyze_verl_rollouts,
    read_verl_jsonl,
)

_CORE_REPORT_KEYS = {
    "episodes",
    "groups",
    "group_size_histogram",
    "mean_reward",
    "reward_std",
    "em_mean",
    "f1_mean",
    "valid_answer_rate",
    "attempted_tool_call_count_mean",
    "literal_tool_call_count_mean",
    "valid_tool_call_count_mean",
    "valid_search_call_count_mean",
    "executed_tool_call_count_mean",
    "executed_search_call_count_mean",
    "malformed_tool_call_count_mean",
    "malformed_tool_call_episode_rate",
    "malformed_tool_call_rate",
    "unknown_tool_call_count_mean",
    "mean_group_reward_variance",
    "zero_variance_group_ratio",
    "nontrivial_reward_group_ratio",
    "all_groups_have_trajectory_diversity",
    "answer_tag_rate",
    "avg_search_calls",
    "multi_search_rate",
    "three_plus_search_rate",
    "tool_efficiency",
    "useful_search_call_count",
    "wasted_search_call_count",
    "second_search_useful_rate",
}


def _core_report(report: dict[str, Any]) -> dict[str, Any]:
    """Drop verbose per-group details while retaining gate diagnostics."""
    return {key: report[key] for key in _CORE_REPORT_KEYS if key in report}


def _step_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _analyze_file(path: Path, tokenizer, supporting_titles_by_question) -> dict[str, Any]:
    rows = read_verl_jsonl(str(path))
    return {
        "task": analyze_verl_rollouts(rows),
        "behavior": analyze_verl_behavior(
            rows,
            tokenizer=tokenizer,
            supporting_titles_by_question=supporting_titles_by_question,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollouts",
        type=Path,
        required=True,
        help="A rollout dump JSONL file, or a run directory containing *.jsonl dumps.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional local HF tokenizer for generated-token counts.",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        help="Optional HotpotQA JSON/JSONL or verl parquet for useful-search metadata.",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    supporting_titles_by_question = None
    if args.examples:
        supporting_titles_by_question = {
            example.question: example.supporting_titles
            for example in load_verl_examples(args.examples, args.split)
        }

    if args.rollouts.is_dir():
        files = sorted(args.rollouts.glob("*.jsonl"), key=_step_number)
        if not files:
            raise SystemExit(f"no JSONL rollout files found in {args.rollouts}")
        all_rows: list[dict[str, Any]] = []
        by_step: list[dict[str, Any]] = []
        for path in files:
            rows = read_verl_jsonl(str(path))
            all_rows.extend(rows)
            analysis = _analyze_file(path, tokenizer, supporting_titles_by_question)
            by_step.append(
                {
                    "step": _step_number(path),
                    "file": str(path),
                    "task": _core_report(analysis["task"]),
                    "behavior": analysis["behavior"],
                }
            )
        report = {
            "rollout_dir": str(args.rollouts),
            "rollout_files": [str(path) for path in files],
            "tokenizer": str(args.tokenizer) if args.tokenizer else None,
            "by_step": by_step,
            "overall": {
                "task": _core_report(analyze_verl_rollouts(all_rows)),
                "behavior": analyze_verl_behavior(
                    all_rows,
                    tokenizer=tokenizer,
                    supporting_titles_by_question=supporting_titles_by_question,
                ),
            },
        }
    else:
        report = _analyze_file(args.rollouts, tokenizer, supporting_titles_by_question)

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
