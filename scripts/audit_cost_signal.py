#!/usr/bin/env python3
"""Audit whether a cost-aware reward changes GRPO group-level signals."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

EPSILON = 1e-9


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def _pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _advantages(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = statistics.mean(values)
    scale = max(_pstdev(values), EPSILON)
    return [(value - mean) / scale for value in values]


def _task_reward(row: dict[str, Any]) -> float:
    value = row.get("task_reward")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("each row requires numeric task_reward")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("task_reward must be finite and in [0, 1]")
    return value


def _wasted_searches(row: dict[str, Any]) -> int:
    value = row.get("wasted_search_calls")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("each row requires integer wasted_search_calls")
    if value < 0:
        raise ValueError("wasted_search_calls must be non-negative")
    return value


def _group_id(row: dict[str, Any], group_key: str) -> str:
    value = row.get(group_key)
    if value is None:
        raise ValueError(f"each row requires group field {group_key!r}")
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _group_report(rows: list[dict[str, Any]], lambda_cost: float) -> dict[str, Any]:
    task = [_task_reward(row) for row in rows]
    waste = [_wasted_searches(row) for row in rows]
    penalties = [lambda_cost * score * count for score, count in zip(task, waste)]
    total = [score - penalty for score, penalty in zip(task, penalties)]
    task_adv = _advantages(task)
    cost_adv = _advantages(total)

    ordered_pairs = 0
    ranking_flips = 0
    equal_task_pairs = 0
    lower_waste_tiebreaks = 0
    for index in range(len(rows)):
        for other in range(index + 1, len(rows)):
            task_delta = task[index] - task[other]
            cost_delta = total[index] - total[other]
            if abs(task_delta) > EPSILON:
                ordered_pairs += 1
                if task_delta * cost_delta < -EPSILON:
                    ranking_flips += 1
            elif waste[index] != waste[other] and abs(cost_delta) > EPSILON:
                equal_task_pairs += 1
                if (waste[index] < waste[other] and cost_delta > EPSILON) or (
                    waste[other] < waste[index] and cost_delta < -EPSILON
                ):
                    lower_waste_tiebreaks += 1

    return {
        "size": len(rows),
        "task_rewards": task,
        "cost_rewards": total,
        "cost_penalties": penalties,
        "wasted_search_calls": waste,
        "cost_active": any(penalty > EPSILON for penalty in penalties),
        "advantage_changed": any(
            abs(left - right) > EPSILON for left, right in zip(task_adv, cost_adv)
        ),
        "mean_abs_advantage_delta": _mean(
            abs(left - right) for left, right in zip(task_adv, cost_adv)
        ),
        "max_abs_advantage_delta": max(
            (abs(left - right) for left, right in zip(task_adv, cost_adv)),
            default=0.0,
        ),
        "ordered_pairs": ordered_pairs,
        "ranking_flips": ranking_flips,
        "equal_task_pairs_with_waste_difference": equal_task_pairs,
        "lower_waste_tiebreaks": lower_waste_tiebreaks,
    }


def audit_batch(
    rows: list[dict[str, Any]],
    *,
    lambda_cost: float,
    group_key: str = "input",
    source: str = "batch",
) -> dict[str, Any]:
    """Audit one optimizer-step rollout batch, grouped by its prompt."""
    if not rows:
        raise ValueError("rollout batch must not be empty")
    if not math.isfinite(lambda_cost) or lambda_cost < 0.0:
        raise ValueError("lambda_cost must be finite and non-negative")

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_id(row, group_key), []).append(row)
    group_reports = [
        _group_report(group_rows, lambda_cost) for group_rows in groups.values()
    ]
    task_rewards = [_task_reward(row) for row in rows]
    waste_counts = [_wasted_searches(row) for row in rows]
    penalties = [
        lambda_cost * score * waste
        for score, waste in zip(task_rewards, waste_counts)
    ]

    positive_task_count = sum(score > EPSILON for score in task_rewards)
    positive_task_waste_count = sum(
        score > EPSILON and waste > 0
        for score, waste in zip(task_rewards, waste_counts)
    )
    ordered_pairs = sum(item["ordered_pairs"] for item in group_reports)
    ranking_flips = sum(item["ranking_flips"] for item in group_reports)
    equal_task_pairs = sum(
        item["equal_task_pairs_with_waste_difference"] for item in group_reports
    )
    lower_waste_tiebreaks = sum(item["lower_waste_tiebreaks"] for item in group_reports)
    advantage_deltas = [item["mean_abs_advantage_delta"] for item in group_reports]
    return {
        "source": source,
        "rows": len(rows),
        "groups": len(group_reports),
        "group_size_histogram": dict(
            sorted(Counter(len(group_rows) for group_rows in groups.values()).items())
        ),
        "lambda_cost": lambda_cost,
        "mean_task_reward": _mean(task_rewards),
        "mean_cost_penalty": _mean(penalties),
        "mean_total_reward": _mean(
            score - penalty for score, penalty in zip(task_rewards, penalties)
        ),
        "nonzero_penalty_count": sum(penalty > EPSILON for penalty in penalties),
        "nonzero_penalty_rate": sum(penalty > EPSILON for penalty in penalties)
        / len(rows),
        "cost_active_group_count": sum(item["cost_active"] for item in group_reports),
        "cost_active_group_rate": sum(item["cost_active"] for item in group_reports)
        / len(group_reports),
        "zero_cost_group_count": sum(not item["cost_active"] for item in group_reports),
        "zero_cost_group_rate": sum(not item["cost_active"] for item in group_reports)
        / len(group_reports),
        "positive_task_reward_count": positive_task_count,
        "positive_task_waste_count": positive_task_waste_count,
        "positive_task_waste_rate": positive_task_waste_count
        / max(positive_task_count, 1),
        "mean_wasted_search_calls": _mean(waste_counts),
        "advantage_changed_group_count": sum(
            item["advantage_changed"] for item in group_reports
        ),
        "advantage_changed_group_rate": sum(
            item["advantage_changed"] for item in group_reports
        )
        / len(group_reports),
        "mean_abs_advantage_delta": _mean(advantage_deltas),
        "max_abs_advantage_delta": max(
            (item["max_abs_advantage_delta"] for item in group_reports),
            default=0.0,
        ),
        "ranking_flip_count": ranking_flips,
        "ranking_pair_count": ordered_pairs,
        "ranking_flip_rate": ranking_flips / max(ordered_pairs, 1),
        "equal_task_pairs_with_waste_difference": equal_task_pairs,
        "lower_waste_tiebreak_count": lower_waste_tiebreaks,
        "groups_detail": group_reports,
    }


def audit_files(
    paths: list[str | Path],
    *,
    lambda_cost: float,
    group_key: str = "input",
) -> dict[str, Any]:
    """Audit each rollout file independently, then provide an aggregate."""
    batches = [
        audit_batch(
            read_jsonl(path),
            lambda_cost=lambda_cost,
            group_key=group_key,
            source=Path(path).name,
        )
        for path in paths
    ]
    if not batches:
        raise ValueError("at least one rollout file is required")

    rows = sum(item["rows"] for item in batches)
    groups = sum(item["groups"] for item in batches)
    active_groups = sum(item["cost_active_group_count"] for item in batches)
    changed_groups = sum(item["advantage_changed_group_count"] for item in batches)
    penalties = sum(item["nonzero_penalty_count"] for item in batches)
    positive = sum(item["positive_task_reward_count"] for item in batches)
    positive_waste = sum(item["positive_task_waste_count"] for item in batches)
    ordered_pairs = sum(item["ranking_pair_count"] for item in batches)
    ranking_flips = sum(item["ranking_flip_count"] for item in batches)
    return {
        "lambda_cost": lambda_cost,
        "files": [str(path) for path in paths],
        "batches": batches,
        "aggregate": {
            "rows": rows,
            "groups": groups,
            "nonzero_penalty_count": penalties,
            "nonzero_penalty_rate": penalties / rows,
            "cost_active_group_count": active_groups,
            "cost_active_group_rate": active_groups / groups,
            "zero_cost_group_rate": (groups - active_groups) / groups,
            "positive_task_reward_count": positive,
            "positive_task_waste_count": positive_waste,
            "positive_task_waste_rate": positive_waste / max(positive, 1),
            "mean_abs_advantage_delta": _mean(
                item["mean_abs_advantage_delta"] for item in batches
            ),
            "max_abs_advantage_delta": max(
                item["max_abs_advantage_delta"] for item in batches
            ),
            "advantage_changed_group_count": changed_groups,
            "advantage_changed_group_rate": changed_groups / groups,
            "ranking_flip_count": ranking_flips,
            "ranking_pair_count": ordered_pairs,
            "ranking_flip_rate": ranking_flips / max(ordered_pairs, 1),
        },
    }


def render_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# M5.1 Cost-Signal Density Audit",
        "",
        "This is an offline audit of stored rollout batches. It does not train a policy or claim an efficiency improvement.",
        "",
        f"- Lambda: `{report['lambda_cost']}`",
        f"- Rows: `{aggregate['rows']}`",
        f"- GRPO groups: `{aggregate['groups']}`",
        "",
        "## Aggregate signal density",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Non-zero penalty rows | {aggregate['nonzero_penalty_count']} ({aggregate['nonzero_penalty_rate']:.2%}) |",
        f"| Cost-active groups | {aggregate['cost_active_group_count']} ({aggregate['cost_active_group_rate']:.2%}) |",
        f"| Zero-cost groups | {aggregate['zero_cost_group_rate']:.2%} |",
        f"| Positive-task rows with waste | {aggregate['positive_task_waste_count']} / {aggregate['positive_task_reward_count']} ({aggregate['positive_task_waste_rate']:.2%}) |",
        f"| Groups with changed advantage | {aggregate['advantage_changed_group_count']} ({aggregate['advantage_changed_group_rate']:.2%}) |",
        f"| Mean absolute advantage delta | {aggregate['mean_abs_advantage_delta']:.6f} |",
        f"| Max absolute advantage delta | {aggregate['max_abs_advantage_delta']:.6f} |",
        f"| Ranking flips | {aggregate['ranking_flip_count']} / {aggregate['ranking_pair_count']} ({aggregate['ranking_flip_rate']:.2%}) |",
        "",
        "## Per-batch summary",
        "",
        "| Batch | Rows | Groups | Penalty rows | Active groups | Mean |Δ advantage| | Ranking flips |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for batch in report["batches"]:
        lines.append(
            f"| {batch['source']} | {batch['rows']} | {batch['groups']} | "
            f"{batch['nonzero_penalty_count']} ({batch['nonzero_penalty_rate']:.1%}) | "
            f"{batch['cost_active_group_count']} ({batch['cost_active_group_rate']:.1%}) | "
            f"{batch['mean_abs_advantage_delta']:.6f} | "
            f"{batch['ranking_flip_count']} / {batch['ranking_pair_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `cost_active_group_rate` is the primary density measure for GRPO.",
            "- A non-zero penalty row does not necessarily change a group's relative ordering.",
            "- Ranking flips measure reversals among pairs with different task rewards; equal-task pairs are reported separately as lower-waste tie-breaks in JSON.",
            "- These statistics diagnose whether a larger behavioral run is justified; they do not establish that a trained policy will become more efficient.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-files", type=Path, nargs="+", required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--group-key", default="input")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_files(
        args.rollout_files,
        lambda_cost=args.lambda_cost,
        group_key=args.group_key,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(report), encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
