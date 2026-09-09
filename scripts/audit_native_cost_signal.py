#!/usr/bin/env python3
"""Audit cost-aware GRPO signals in stored native verl rollout batches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from efficienttool_rl.rewards.cost_aware import search_usage
from efficienttool_rl.rewards.task import task_reward

from scripts.audit_cost_signal import audit_batch, read_jsonl, render_report


def load_supporting_titles(parquet_path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load only question/support annotations from a verl parquet artifact."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional data extra
        raise RuntimeError(
            "pyarrow is required; install the project's data dependencies"
        ) from exc

    table = pq.read_table(str(parquet_path), columns=["extra_info"])
    mapping: dict[str, tuple[str, ...]] = {}
    for row in table.to_pylist():
        info = row.get("extra_info")
        if not isinstance(info, dict):
            raise ValueError("parquet extra_info must be a mapping")
        question = info.get("question")
        titles = info.get("supporting_titles")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("each extra_info row requires a question")
        if not isinstance(titles, (list, tuple)) or not all(
            isinstance(title, str) for title in titles
        ):
            raise ValueError("each extra_info row requires supporting_titles")
        value = tuple(titles)
        previous = mapping.get(question)
        if previous is not None and previous != value:
            raise ValueError(f"conflicting supporting titles for {question!r}")
        mapping[question] = value
    return mapping


def question_from_input(input_text: str) -> str:
    """Extract the user question from a serialized native verl prompt."""
    if not isinstance(input_text, str):
        raise ValueError("native input must be a string")
    marker = "\nuser\n"
    if marker in input_text:
        question = input_text.split(marker, 1)[1]
        if "\nassistant" in question:
            question = question.split("\nassistant", 1)[0]
        if question.strip():
            return question.strip()
    raise ValueError("could not extract question from native input")


def native_rows_to_audit_rows(
    rows: list[dict[str, Any]],
    supporting_titles_by_question: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    """Replay task and executed/useful/wasted counts for native rows."""
    converted: list[dict[str, Any]] = []
    for row in rows:
        output = row.get("output")
        reference = row.get("gts")
        input_text = row.get("input")
        if not isinstance(output, str) or not isinstance(reference, str):
            raise ValueError("each native row requires string output and gts")
        question = question_from_input(input_text)
        titles = supporting_titles_by_question.get(question)
        if titles is None:
            raise ValueError(f"no supporting-title metadata for {question!r}")
        task = task_reward(output, reference, alpha=0.5)
        usage = search_usage(output, {"supporting_titles": titles})
        converted.append(
            {
                "input": input_text,
                "task_reward": float(task["score"]),
                "wasted_search_calls": usage["wasted_search_calls"],
                "executed_search_calls": usage["executed_search_calls"],
                "useful_search_calls": usage["useful_search_calls"],
            }
        )
    return converted


def _aggregate(batches: list[dict[str, Any]], lambda_cost: float) -> dict[str, Any]:
    if not batches:
        raise ValueError("at least one batch is required")
    rows = sum(item["rows"] for item in batches)
    groups = sum(item["groups"] for item in batches)
    active = sum(item["cost_active_group_count"] for item in batches)
    changed = sum(item["advantage_changed_group_count"] for item in batches)
    penalties = sum(item["nonzero_penalty_count"] for item in batches)
    positive = sum(item["positive_task_reward_count"] for item in batches)
    positive_waste = sum(item["positive_task_waste_count"] for item in batches)
    ranking_flips = sum(item["ranking_flip_count"] for item in batches)
    ranking_pairs = sum(item["ranking_pair_count"] for item in batches)
    return {
        "rows": rows,
        "groups": groups,
        "lambda_cost": lambda_cost,
        "nonzero_penalty_count": penalties,
        "nonzero_penalty_rate": penalties / rows,
        "cost_active_group_count": active,
        "cost_active_group_rate": active / groups,
        "zero_cost_group_rate": (groups - active) / groups,
        "positive_task_reward_count": positive,
        "positive_task_waste_count": positive_waste,
        "positive_task_waste_rate": positive_waste / max(positive, 1),
        "mean_abs_advantage_delta": sum(
            item["mean_abs_advantage_delta"] * item["groups"] for item in batches
        ) / groups,
        "max_abs_advantage_delta": max(
            item["max_abs_advantage_delta"] for item in batches
        ),
        "advantage_changed_group_count": changed,
        "advantage_changed_group_rate": changed / groups,
        "ranking_flip_count": ranking_flips,
        "ranking_pair_count": ranking_pairs,
        "ranking_flip_rate": ranking_flips / max(ranking_pairs, 1),
    }


def audit_native_files(
    rollout_files: list[str | Path],
    metadata_parquet: str | Path,
    *,
    lambda_cost: float,
) -> dict[str, Any]:
    titles = load_supporting_titles(metadata_parquet)
    batches = []
    for path in rollout_files:
        native_rows = read_jsonl(path)
        rows = native_rows_to_audit_rows(native_rows, titles)
        batches.append(
            audit_batch(
                rows,
                lambda_cost=lambda_cost,
                source=Path(path).name,
            )
        )
    return {
        "lambda_cost": lambda_cost,
        "metadata_parquet": str(metadata_parquet),
        "files": [str(path) for path in rollout_files],
        "batches": batches,
        "aggregate": _aggregate(batches, lambda_cost),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-files", type=Path, nargs="+", required=True)
    parser.add_argument("--metadata-parquet", type=Path, required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.lambda_cost) or args.lambda_cost < 0:
        raise SystemExit("--lambda-cost must be finite and non-negative")
    report = audit_native_files(
        args.rollout_files,
        args.metadata_parquet,
        lambda_cost=args.lambda_cost,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_report(report), encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
