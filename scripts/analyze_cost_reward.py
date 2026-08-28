#!/usr/bin/env python3
"""Offline counterfactual analysis for the first cost-aware reward design."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from efficienttool_rl.data import load_hotpotqa
from efficienttool_rl.evaluation.trajectory_analysis import analyze_trajectories, read_jsonl
from efficienttool_rl.rewards import task_reward

DEFAULT_LAMBDAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.30)
EPSILON = 1e-9


@dataclass(frozen=True)
class TrajectoryScore:
    source: str
    episode_id: str
    task_reward: float
    em: float
    f1: float
    valid_answer: float
    executed_search_calls: int
    useful_search_calls: int
    wasted_search_calls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def counterfactual_reward(
    task_score: float,
    wasted_search_calls: int,
    lambda_cost: float,
) -> dict[str, float]:
    """Compute R_CA = R_task - lambda * R_task * N_waste."""
    if not math.isfinite(task_score) or not 0.0 <= task_score <= 1.0:
        raise ValueError("task_score must be finite and in [0, 1]")
    if not math.isfinite(lambda_cost) or lambda_cost < 0.0:
        raise ValueError("lambda_cost must be finite and non-negative")
    if not isinstance(wasted_search_calls, int) or isinstance(
        wasted_search_calls, bool
    ):
        raise ValueError("wasted_search_calls must be an integer")
    if wasted_search_calls < 0:
        raise ValueError("wasted_search_calls must be non-negative")
    penalty = lambda_cost * task_score * wasted_search_calls
    return {
        "task_reward": task_score,
        "cost_penalty": penalty,
        "total_reward": task_score - penalty,
    }


def _mean(values: Iterable[float | int]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def load_trajectory_scores(
    path: str | Path,
    examples: list[Any],
    source: str,
) -> tuple[list[TrajectoryScore], dict[str, Any]]:
    """Use the existing analyzer and canonical task-only reward."""
    trajectories = read_jsonl(path)
    if not trajectories:
        raise ValueError(f"{path}: no trajectories found")
    if any("steps" not in item for item in trajectories):
        raise ValueError(f"{path}: every trajectory must include structured steps")
    _, categorized = analyze_trajectories(trajectories, examples)
    if len(categorized) != len(trajectories):
        raise ValueError(f"{path}: analyzer row count does not match input")

    scores: list[TrajectoryScore] = []
    seen_ids: set[str] = set()
    for item in categorized:
        episode_id = item["episode_id"]
        if episode_id in seen_ids:
            raise ValueError(f"{path}: duplicate trajectory id {episode_id!r}")
        seen_ids.add(episode_id)
        answer = item.get("final_answer")
        response = (
            ""
            if not isinstance(answer, str) or not answer.strip()
            else "<answer>" + answer.strip() + "</answer>"
        )
        task = task_reward(response, item["reference_answer"])
        executed = int(item["executed_search_calls"])
        useful = int(item["useful_search_calls"])
        wasted = int(item["wasted_search_calls"])
        if executed < 0 or useful < 0 or wasted < 0:
            raise ValueError(f"{path}: negative tool-use count for {episode_id!r}")
        if useful > executed or wasted != max(executed - useful, 0):
            raise ValueError(f"{path}: inconsistent tool-use count for {episode_id!r}")
        scores.append(
            TrajectoryScore(
                source=source,
                episode_id=episode_id,
                task_reward=float(task["score"]),
                em=float(task["em"]),
                f1=float(task["f1"]),
                valid_answer=float(task["valid_answer"]),
                executed_search_calls=executed,
                useful_search_calls=useful,
                wasted_search_calls=wasted,
            )
        )
    behavior = {
        "episodes": len(scores),
        "mean_task_reward": _mean(x.task_reward for x in scores),
        "mean_em": _mean(x.em for x in scores),
        "mean_f1": _mean(x.f1 for x in scores),
        "mean_executed_search_calls": _mean(
            x.executed_search_calls for x in scores
        ),
        "mean_useful_search_calls": _mean(x.useful_search_calls for x in scores),
        "mean_wasted_search_calls": _mean(x.wasted_search_calls for x in scores),
        "multi_search_rate": _rate(x.executed_search_calls >= 2 for x in scores),
        "zero_search_rate": _rate(x.executed_search_calls == 0 for x in scores),
    }
    return scores, behavior


def _reward(item: TrajectoryScore, lambda_cost: float) -> float:
    return counterfactual_reward(
        item.task_reward, item.wasted_search_calls, lambda_cost
    )["total_reward"]


def pairwise_rank_stats(
    records: list[TrajectoryScore],
    lambda_cost: float,
) -> dict[str, float | int]:
    """Measure ranking reversals and lower-waste preferences."""
    if not math.isfinite(lambda_cost) or lambda_cost < 0.0:
        raise ValueError("lambda_cost must be finite and non-negative")
    task_pairs = task_inversions = 0
    equal_waste_pairs = lower_waste_preferences = 0
    correct_wrong_pairs = correct_wrong_inversions = 0
    correct_two_pairs = correct_two_inversions = 0
    for left, right in itertools.combinations(records, 2):
        left_value, right_value = _reward(left, lambda_cost), _reward(right, lambda_cost)
        delta = left.task_reward - right.task_reward
        if abs(delta) > EPSILON:
            task_pairs += 1
            high, low = (left, right) if delta > 0 else (right, left)
            if _reward(high, lambda_cost) + EPSILON < _reward(low, lambda_cost):
                task_inversions += 1
        elif left.wasted_search_calls != right.wasted_search_calls:
            equal_waste_pairs += 1
            low_value, high_value = (
                (left_value, right_value)
                if left.wasted_search_calls < right.wasted_search_calls
                else (right_value, left_value)
            )
            if low_value > high_value + EPSILON:
                lower_waste_preferences += 1

        left_correct, right_correct = left.em > 0.5, right.em > 0.5
        if left_correct != right_correct:
            correct_wrong_pairs += 1
            correct, wrong = (left, right) if left_correct else (right, left)
            if _reward(correct, lambda_cost) + EPSILON < _reward(wrong, lambda_cost):
                correct_wrong_inversions += 1

        left_target = left_correct and left.executed_search_calls >= 2
        right_target = right_correct and right.executed_search_calls >= 2
        left_wrong_zero = not left_correct and left.executed_search_calls == 0
        right_wrong_zero = not right_correct and right.executed_search_calls == 0
        if left_target and right_wrong_zero:
            correct_two_pairs += 1
            if left_value + EPSILON < right_value:
                correct_two_inversions += 1
        elif right_target and left_wrong_zero:
            correct_two_pairs += 1
            if right_value + EPSILON < left_value:
                correct_two_inversions += 1
    return {
        "records": len(records),
        "lambda_cost": lambda_cost,
        "task_order_pairs": task_pairs,
        "task_order_inversions": task_inversions,
        "task_order_inversion_rate": task_inversions / max(task_pairs, 1),
        "equal_task_pairs_with_waste_difference": equal_waste_pairs,
        "lower_waste_preference_pairs": lower_waste_preferences,
        "lower_waste_preference_rate": lower_waste_preferences / max(equal_waste_pairs, 1),
        "correct_vs_wrong_pairs": correct_wrong_pairs,
        "correct_vs_wrong_inversions": correct_wrong_inversions,
        "correct_vs_wrong_inversion_rate": correct_wrong_inversions / max(correct_wrong_pairs, 1),
        "correct_two_search_vs_wrong_zero_pairs": correct_two_pairs,
        "correct_two_search_vs_wrong_zero_inversions": correct_two_inversions,
        "correct_two_search_vs_wrong_zero_inversion_rate": correct_two_inversions / max(correct_two_pairs, 1),
    }


def _find_pair(
    records: list[TrajectoryScore],
    left_predicate: Callable[[TrajectoryScore], bool],
    right_predicate: Callable[[TrajectoryScore], bool],
    compatible: Callable[[TrajectoryScore, TrajectoryScore], bool] | None = None,
) -> tuple[TrajectoryScore, TrajectoryScore] | None:
    left_items = sorted(filter(left_predicate, records), key=lambda x: x.episode_id)
    right_items = sorted(filter(right_predicate, records), key=lambda x: x.episode_id)
    for left in left_items:
        for right in right_items:
            if left.episode_id != right.episode_id and (
                compatible is None or compatible(left, right)
            ):
                return left, right
    return None


def _pair_result(
    name: str,
    pair: tuple[TrajectoryScore, TrajectoryScore] | None,
    lambdas: list[float],
    relation: str,
    predicate: Callable[[float, float, float], bool],
) -> dict[str, Any]:
    if pair is None:
        return {
            "name": name,
            "status": "UNAVAILABLE",
            "relation": relation,
            "reason": "No matching pair exists in the stored trajectories.",
        }
    left, right = pair
    comparisons = []
    for lambda_cost in lambdas:
        left_value, right_value = _reward(left, lambda_cost), _reward(right, lambda_cost)
        comparisons.append(
            {
                "lambda_cost": lambda_cost,
                "left_reward": left_value,
                "right_reward": right_value,
                "passed": predicate(left_value, right_value, lambda_cost),
            }
        )
    return {
        "name": name,
        "status": "PASS" if all(x["passed"] for x in comparisons) else "FAIL",
        "relation": relation,
        "left": left.to_dict(),
        "right": right.to_dict(),
        "comparisons": comparisons,
    }


def build_pair_checks(
    records_by_source: dict[str, list[TrajectoryScore]],
    lambdas: list[float],
) -> dict[str, list[dict[str, Any]]]:
    """Check anti-under-search and waste-ordering cases from stored rows."""
    checks: dict[str, list[dict[str, Any]]] = {}
    for source, records in records_by_source.items():
        same_task = lambda left, right: abs(left.task_reward - right.task_reward) <= EPSILON
        checks[source] = [
            _pair_result(
                "correct_useful_vs_correct_wasted",
                _find_pair(
                    records,
                    lambda x: x.em > 0.5 and x.useful_search_calls >= 2 and x.wasted_search_calls == 0,
                    lambda x: x.em > 0.5 and x.useful_search_calls >= 2 and x.wasted_search_calls >= 1,
                    same_task,
                ),
                lambdas,
                "same task reward: zero waste should outrank added waste",
                lambda left, right, lambda_cost: (
                    abs(left - right) <= EPSILON
                    if lambda_cost == 0.0
                    else left > right + EPSILON
                ),
            ),
            _pair_result(
                "correct_two_search_vs_wrong_zero",
                _find_pair(
                    records,
                    lambda x: x.em > 0.5 and x.executed_search_calls >= 2,
                    lambda x: x.em <= 0.5 and x.executed_search_calls == 0,
                ),
                lambdas,
                "correct multi-search should outrank wrong zero-search",
                lambda left, right, _lambda_cost: left > right + EPSILON,
            ),
            _pair_result(
                "same_task_quality_no_waste",
                _find_pair(
                    records,
                    lambda x: x.task_reward > EPSILON and x.wasted_search_calls == 0 and x.useful_search_calls == 1,
                    lambda x: x.task_reward > EPSILON and x.wasted_search_calls == 0 and x.useful_search_calls >= 2,
                    same_task,
                ),
                lambdas,
                "equal task quality and zero waste should remain tied",
                lambda left, right, _lambda_cost: abs(left - right) <= EPSILON,
            ),
            _pair_result(
                "partial_with_waste_vs_wrong_zero",
                _find_pair(
                    records,
                    lambda x: 0.0 < x.task_reward < 1.0 and x.wasted_search_calls >= 1,
                    lambda x: x.em <= 0.5 and x.executed_search_calls == 0,
                ),
                lambdas,
                "partial task signal should outrank wrong zero-search",
                lambda left, right, _lambda_cost: left > right + EPSILON,
            ),
        ]
    return checks


def build_reward_landscape(lambdas: list[float]) -> dict[str, Any]:
    """Build synthetic formula cases, explicitly not benchmark evidence."""
    cases = [
        ("correct_2_useful_0_waste", 1.0, 0),
        ("correct_2_useful_1_waste", 1.0, 1),
        ("partial_task_reward_0.50_1_waste", 0.5, 1),
        ("wrong_0_search", 0.0, 0),
    ]
    return {
        "illustrative_only": True,
        "description": "Synthetic formula cases, not benchmark trajectories.",
        "rows": [
            {
                "case": name,
                "task_reward": task_score,
                "wasted_search_calls": wasted,
                "values": [
                    {
                        "lambda_cost": lambda_cost,
                        **counterfactual_reward(task_score, wasted, lambda_cost),
                    }
                    for lambda_cost in lambdas
                ],
            }
            for name, task_score, wasted in cases
        ],
    }


def summarize_source(
    records: list[TrajectoryScore],
    lambdas: list[float],
    behavior: dict[str, Any],
) -> dict[str, Any]:
    by_lambda = {}
    for lambda_cost in lambdas:
        totals = [_reward(x, lambda_cost) for x in records]
        penalties = [
            counterfactual_reward(
                x.task_reward, x.wasted_search_calls, lambda_cost
            )["cost_penalty"]
            for x in records
        ]
        by_lambda[str(lambda_cost)] = {
            "lambda_cost": lambda_cost,
            "mean_task_reward": _mean(x.task_reward for x in records),
            "mean_cost_penalty": _mean(penalties),
            "mean_total_reward": _mean(totals),
            "total_reward_std": _std(totals),
            "zero_total_reward_rate": _rate(abs(x) <= EPSILON for x in totals),
            "pairwise": pairwise_rank_stats(records, lambda_cost),
        }
    return {"behavior": behavior, "counterfactual_by_lambda": by_lambda}


def render_report(
    summary: dict[str, Any],
    landscape: dict[str, Any],
    pair_checks: dict[str, Any],
) -> str:
    lines = [
        "# M5.0 Offline Cost-Reward Design",
        "",
        "Offline counterfactual study only: no policy was trained and no performance improvement is claimed.",
        "",
        "## Objective",
        "",
        "R_CA = R_task - lambda * R_task * N_waste",
        "",
        "R_task = 0.5 * EM + 0.5 * F1",
        "",
        "Only wasted executed searches are penalized. Useful searches are not directly penalized, and zero task reward produces zero penalty.",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in summary["inputs"].items())
    lines.extend(
        [
            "",
            "Lambdas: " + ", ".join(str(x) for x in summary["lambdas"]),
            "",
            "## Stored behavior summary",
            "",
            "| Source | Episodes | Task reward | Executed | Useful | Wasted | Multi-search |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, item in summary["sources"].items():
        behavior = item["behavior"]
        lines.append(
            f"| {source} | {behavior['episodes']} | {behavior['mean_task_reward']:.4f} | "
            f"{behavior['mean_executed_search_calls']:.4f} | "
            f"{behavior['mean_useful_search_calls']:.4f} | "
            f"{behavior['mean_wasted_search_calls']:.4f} | "
            f"{behavior['multi_search_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Illustrative reward landscape",
            "",
            "Synthetic formula cases only; not benchmark evidence.",
            "",
            "| Case | Task reward | Waste | "
            + " | ".join(
                f"lambda={x['lambda_cost']:g}"
                for x in landscape["rows"][0]["values"]
            )
            + " |",
            "|---|---:|---:|" + "---:|" * len(landscape["rows"][0]["values"]),
        ]
    )
    for row in landscape["rows"]:
        lines.append(
            f"| {row['case']} | {row['task_reward']:.2f} | "
            f"{row['wasted_search_calls']} | "
            + " | ".join(f"{x['total_reward']:.4f}" for x in row["values"])
            + " |"
        )
    lines.extend(["", "## Counterfactual aggregate scores", ""])
    lines.extend(
        [
            "| Source | Lambda | Mean task | Mean penalty | Mean total | Total std |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for source, item in summary["sources"].items():
        for lambda_cost in summary["lambdas"]:
            row = item["counterfactual_by_lambda"][str(lambda_cost)]
            lines.append(
                f"| {source} | {lambda_cost:g} | {row['mean_task_reward']:.4f} | "
                f"{row['mean_cost_penalty']:.4f} | {row['mean_total_reward']:.4f} | "
                f"{row['total_reward_std']:.4f} |"
            )
    lines.extend(["", "## Pairwise ranking checks", ""])
    for source, checks in pair_checks.items():
        lines.append(f"### {source}")
        for check in checks:
            lines.append(f"- {check['name']}: {check['status']} — {check['relation']}.")
            if check["status"] == "UNAVAILABLE":
                lines.append(f"  {check['reason']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lambda 0 recovers task-only reward.",
            "- Useful calls are not charged by this objective.",
            "- Wrong zero-search rows remain at zero reward.",
            "- Pairwise results diagnose the reward design; they do not prove a trained policy will learn the intended behavior.",
            "",
            "## Next step",
            "",
            "If the checks are acceptable, implement the reward in an isolated module and run a bounded 8-by-4 smoke test.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    landscape: dict[str, Any],
    pair_checks: dict[str, Any],
    records: list[TrajectoryScore],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("summary.json", summary),
        ("reward_landscape.json", landscape),
        ("ranking_checks.json", pair_checks),
    ):
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    with (output_dir / "trajectory_scores.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            row = record.to_dict()
            row["counterfactual"] = [
                {
                    "lambda_cost": lambda_cost,
                    **counterfactual_reward(
                        record.task_reward,
                        record.wasted_search_calls,
                        lambda_cost,
                    ),
                }
                for lambda_cost in summary["lambdas"]
            ]
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "reward_landscape.csv").open("w", encoding="utf-8") as handle:
        headers = ["case", "task_reward", "wasted_search_calls"] + [
            f"total_lambda_{x:g}" for x in summary["lambdas"]
        ]
        handle.write(",".join(headers) + "\n")
        for row in landscape["rows"]:
            values = [
                row["case"],
                str(row["task_reward"]),
                str(row["wasted_search_calls"]),
            ] + [str(x["total_reward"]) for x in row["values"]]
            handle.write(",".join(values) + "\n")
    (output_dir / "report.md").write_text(
        render_report(summary, landscape, pair_checks),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--base-trajectories", type=Path, required=True)
    parser.add_argument("--step62-trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    args = parser.parse_args()
    if not args.lambdas or any(
        not math.isfinite(x) or x < 0.0 for x in args.lambdas
    ):
        raise SystemExit("--lambdas must be finite and non-negative")
    lambdas = list(dict.fromkeys(args.lambdas))
    examples = load_hotpotqa(args.examples, split="validation")
    records_by_source: dict[str, list[TrajectoryScore]] = {}
    behavior_by_source: dict[str, dict[str, Any]] = {}
    for source, path in (
        ("base", args.base_trajectories),
        ("step62", args.step62_trajectories),
    ):
        records, behavior = load_trajectory_scores(path, examples, source)
        records_by_source[source] = records
        behavior_by_source[source] = behavior
    all_records = [x for records in records_by_source.values() for x in records]
    summary = {
        "formula": "R_CA = R_task - lambda * R_task * N_waste",
        "task_reward_formula": "R_task = 0.5 * EM + 0.5 * F1",
        "lambdas": lambdas,
        "inputs": {
            "examples": args.examples.name,
            "base_trajectories": args.base_trajectories.name,
            "step62_trajectories": args.step62_trajectories.name,
        },
        "sources": {
            source: summarize_source(records, lambdas, behavior_by_source[source])
            for source, records in records_by_source.items()
        },
    }
    landscape = build_reward_landscape(lambdas)
    pair_checks = build_pair_checks(records_by_source, lambdas)
    write_outputs(args.output_dir, summary, landscape, pair_checks, all_records)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "trajectory_count": len(all_records),
                "sources": {
                    source: {
                        "episodes": behavior["episodes"],
                        "mean_task_reward": behavior["mean_task_reward"],
                        "mean_wasted_search_calls": behavior["mean_wasted_search_calls"],
                    }
                    for source, behavior in behavior_by_source.items()
                },
                "pair_check_status": {
                    source: [x["status"] for x in checks]
                    for source, checks in pair_checks.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
