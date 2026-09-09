#!/usr/bin/env python3
"""Offline reward-distribution validation for the composite reward.

Replays ``episode_composite_reward`` over stored fixed-policy trajectory
rows (the schema written by ``scripts/evaluate.py``) and reports component
distributions, evidence-vs-answer correlation, and search-behavior checks.
This is the pre-training gate for choosing composite reward weights; it never
modifies the training reward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from efficienttool_rl.data import load_verl_examples
from efficienttool_rl.rewards.composite import (
    DEFAULT_ANSWER_WEIGHT,
    DEFAULT_EVIDENCE_WEIGHT,
    DEFAULT_FORMAT_WEIGHT,
    episode_composite_reward,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="Fixed-policy trajectory JSONL (evaluate.py schema); repeatable.",
        action="extend",
        nargs="+",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        required=True,
        help="HotpotQA JSON/JSONL or verl parquet supplying supporting titles.",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--answer-weight", type=float, default=DEFAULT_ANSWER_WEIGHT)
    parser.add_argument("--evidence-weight", type=float, default=DEFAULT_EVIDENCE_WEIGHT)
    parser.add_argument("--format-weight", type=float, default=DEFAULT_FORMAT_WEIGHT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.answer_weight, args.evidence_weight, args.format_weight) < 0:
        raise ValueError("reward weights must be non-negative")
    if args.answer_weight + args.evidence_weight + args.format_weight <= 0:
        raise ValueError("at least one reward weight must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise ValueError("no trajectory rows found")
    return rows


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    arr_x, arr_y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if arr_x.std() == 0 or arr_y.std() == 0:
        return float("nan")
    return float(np.corrcoef(arr_x, arr_y)[0, 1])


def _component_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _row_fields(row: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    """Return (episode, question, reference) for either stored row schema."""
    if isinstance(row.get("trajectory"), dict):
        return row["trajectory"], str(row.get("input", "")), str(row.get("reference_answer", ""))
    if isinstance(row.get("steps"), list):
        return row, str(row.get("prompt", row.get("input", ""))), ""
    raise ValueError("rows must contain a 'trajectory' episode or inline episode fields")


def main() -> None:
    args = _parse_args()
    titles_by_question: dict[str, list[str]] = {}
    reference_by_question: dict[str, str] = {}
    for example in load_verl_examples(args.examples, args.split):
        titles_by_question[example.question] = list(example.supporting_titles)
        reference_by_question[example.question] = example.answer

    rows = _read_rows(list(args.trajectories))
    scored: list[dict[str, float]] = []
    missing_metadata = 0
    for row in rows:
        episode, question, reference = _row_fields(row)
        if not reference:
            reference = reference_by_question.get(question, "")
        supporting = titles_by_question.get(question)
        if supporting is None:
            missing_metadata += 1
            supporting = []
        scored.append(
            episode_composite_reward(
                episode,
                reference,
                supporting_titles=supporting,
                answer_weight=args.answer_weight,
                evidence_weight=args.evidence_weight,
                format_weight=args.format_weight,
            )
        )

    em = [row["em"] for row in scored]
    report: dict[str, Any] = {
        "rows": len(scored),
        "weights": {
            "answer_weight": args.answer_weight,
            "evidence_weight": args.evidence_weight,
            "format_weight": args.format_weight,
        },
        "components": {
            key: _component_stats([row[key] for row in scored])
            for key in (
                "score",
                "answer_reward",
                "evidence_reward",
                "format_reward",
                "evidence_coverage",
                "format_valid_answer",
                "format_no_malformed_calls",
                "format_ended_with_answer",
            )
        },
        "evidence_metadata_missing_rows": missing_metadata,
        "correlations": {
            "evidence_coverage_vs_em": _pearson(
                [row["evidence_coverage"] for row in scored], em
            ),
            "evidence_coverage_vs_total": _pearson(
                [row["evidence_coverage"] for row in scored],
                [row["score"] for row in scored],
            ),
            "searches_vs_em": _pearson(
                [float(row["executed_search_calls"]) for row in scored], em
            ),
        },
        "answer_dominance": {
            "aggregate_answer_share_of_total": float(
                np.sum([row["answer_reward"] for row in scored])
                * args.answer_weight
                / np.sum([row["score"] for row in scored])
                if np.sum([row["score"] for row in scored]) > 0
                else float("nan")
            ),
            "aggregate_evidence_share_of_total": float(
                np.sum([row["evidence_reward"] for row in scored])
                * args.evidence_weight
                / np.sum([row["score"] for row in scored])
                if np.sum([row["score"] for row in scored]) > 0
                else float("nan")
            ),
            "evidence_reward_on_zero_em_rows": _component_stats(
                [row["evidence_reward"] for row in scored if row["em"] == 0]
            ),
        },
        "search_behavior": {
            "executed_search_calls": _component_stats(
                [float(row["executed_search_calls"]) for row in scored]
            ),
            "wasted_search_calls": _component_stats(
                [float(row["wasted_search_calls"]) for row in scored]
            ),
            "em_by_search_count": {
                str(count): _component_stats(
                    [row["em"] for row in scored if row["executed_search_calls"] == count]
                )["mean"]
                if any(row["executed_search_calls"] == count for row in scored)
                else None
                for count in sorted({row["executed_search_calls"] for row in scored})
            },
        },
    }

    output = args.output_dir / "composite_reward_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
