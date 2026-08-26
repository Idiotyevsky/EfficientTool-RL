"""Metrics for native verl multi-turn generation dumps."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..rewards import extract_final_answer, task_reward

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", flags=re.DOTALL)


def classify_tool_calls(output: str, *, known_tool_names: frozenset[str] = frozenset({"search"})) -> dict[str, int]:
    """Classify raw Hermes tool-call blocks without changing the rollout.

    The native verl parser logs malformed JSON and drops the call, so the
    serialized response is the only stable post-hoc evidence.  ``malformed``
    covers parser-level failures (invalid JSON or missing ``name``/``arguments``);
    unknown names are tracked separately because their JSON is still valid.
    """
    if not isinstance(output, str):
        return {
            "tool_call_count": 0,
            "valid_tool_call_count": 0,
            "malformed_tool_call_count": 0,
            "unknown_tool_call_count": 0,
        }

    valid = 0
    malformed = 0
    unknown = 0
    for block in _TOOL_CALL_BLOCK.findall(output):
        try:
            decoded = json.loads(block)
            if not isinstance(decoded, dict) or "name" not in decoded or "arguments" not in decoded:
                raise ValueError("tool call must contain name and arguments")
            name = decoded["name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool call name must be a non-empty string")
        except (TypeError, ValueError, json.JSONDecodeError):
            malformed += 1
            continue
        valid += 1
        if name not in known_tool_names:
            unknown += 1

    return {
        "tool_call_count": len(_TOOL_CALL_BLOCK.findall(output)),
        "valid_tool_call_count": valid,
        "malformed_tool_call_count": malformed,
        "unknown_tool_call_count": unknown,
    }


def analyze_verl_rollouts(
    rows: Sequence[Mapping[str, Any]], *, group_key: str = "input"
) -> dict[str, Any]:
    """Replay task reward and summarize per-prompt GRPO groups.

    Native verl dumps are not guaranteed to be ordered by prompt, so groups
    are reconstructed from the initial prompt field instead of row position.
    The report intentionally uses the current project reward implementation.
    """
    if not rows:
        raise ValueError("at least one rollout row is required")

    scores: list[float] = []
    em_values: list[float] = []
    f1_values: list[float] = []
    valid_values: list[float] = []
    tool_call_counts: list[int] = []
    valid_tool_call_counts: list[int] = []
    malformed_tool_call_counts: list[int] = []
    unknown_tool_call_counts: list[int] = []
    groups: dict[str, list[tuple[Mapping[str, Any], float]]] = {}
    for row in rows:
        output = row.get("output", "")
        reference = row.get("gts")
        if not isinstance(output, str) or not isinstance(reference, str):
            raise ValueError("each row requires string output and gts fields")
        reward = task_reward(output, reference)
        score = float(reward["score"])
        scores.append(score)
        em_values.append(float(reward["em"]))
        f1_values.append(float(reward["f1"]))
        valid_values.append(float(reward["valid_answer"]))
        tool_stats = classify_tool_calls(output)
        tool_call_counts.append(tool_stats["tool_call_count"])
        valid_tool_call_counts.append(tool_stats["valid_tool_call_count"])
        malformed_tool_call_counts.append(tool_stats["malformed_tool_call_count"])
        unknown_tool_call_counts.append(tool_stats["unknown_tool_call_count"])
        group_value = row.get(group_key)
        if group_value is None:
            raise ValueError(f"each row requires group field {group_key!r}")
        group_id = json.dumps(group_value, ensure_ascii=False, sort_keys=True)
        groups.setdefault(group_id, []).append((row, score))

    group_reports: list[dict[str, Any]] = []
    variances: list[float] = []
    for group_index, members in enumerate(groups.values()):
        group_scores = [score for _, score in members]
        variance = statistics.pvariance(group_scores) if len(group_scores) > 1 else 0.0
        variances.append(variance)
        first_row = members[0][0]
        group_reports.append(
            {
                "group_index": group_index,
                "ground_truth": first_row.get("gts"),
                "size": len(members),
                "scores": group_scores,
                "reward_variance": variance,
                "distinct_reward_count": len(set(group_scores)),
                "distinct_trajectory_count": len(
                    {str(row.get("output", "")) for row, _ in members}
                ),
            }
        )

    group_sizes = [len(members) for members in groups.values()]
    return {
        "episodes": len(rows),
        "groups": len(groups),
        "group_size_histogram": dict(sorted(Counter(group_sizes).items())),
        "mean_reward": statistics.mean(scores),
        "reward_std": statistics.pstdev(scores),
        "em_mean": statistics.mean(em_values),
        "f1_mean": statistics.mean(f1_values),
        "valid_answer_rate": statistics.mean(valid_values),
        "literal_tool_call_count_mean": statistics.mean(tool_call_counts),
        "valid_tool_call_count_mean": statistics.mean(valid_tool_call_counts),
        "malformed_tool_call_count_mean": statistics.mean(malformed_tool_call_counts),
        "malformed_tool_call_episode_rate": sum(value > 0 for value in malformed_tool_call_counts)
        / len(rows),
        "malformed_tool_call_rate": sum(malformed_tool_call_counts) / max(sum(tool_call_counts), 1),
        "unknown_tool_call_count_mean": statistics.mean(unknown_tool_call_counts),
        "answer_tag_rate": statistics.mean(
            bool(extract_final_answer(str(row.get("output", "")))) for row in rows
        ),
        "mean_group_reward_variance": statistics.mean(variances),
        "zero_variance_group_ratio": sum(value == 0 for value in variances) / len(variances),
        "nontrivial_reward_group_ratio": sum(
            len(report["scores"]) > 1 and report["distinct_reward_count"] > 1
            for report in group_reports
        )
        / len(group_reports),
        "all_groups_have_trajectory_diversity": all(
            report["distinct_trajectory_count"] > 1 for report in group_reports
        ),
        "groups_detail": group_reports,
    }


def read_verl_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a native verl JSONL dump."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
