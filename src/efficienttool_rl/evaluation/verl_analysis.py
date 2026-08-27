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
_TOOL_CALL_OPENING = re.compile(r"<tool_call>")
_TOOL_RESPONSE_BLOCK = re.compile(r"<tool_response>(.*?)</tool_response>", flags=re.DOTALL)
_ROLE_LINE = re.compile(r"(?m)^(?:user|assistant|tool)\s*$")


def classify_tool_calls(
    output: str, *, known_tool_names: frozenset[str] = frozenset({"search"})
) -> dict[str, int]:
    """Classify raw calls and successful responses without changing rollout.

    A literal opening is an attempt; a valid payload is a valid call; and a
    successful ``<tool_response>`` is an execution. ``malformed`` covers
    parser-level failures, while unknown names remain separately visible.
    """
    if not isinstance(output, str):
        return {
            "attempted_tool_call_count": 0,
            "tool_call_count": 0,
            "valid_tool_call_count": 0,
            "valid_search_call_count": 0,
            "executed_tool_call_count": 0,
            "executed_search_call_count": 0,
            "malformed_tool_call_count": 0,
            "unknown_tool_call_count": 0,
        }

    attempted = len(_TOOL_CALL_OPENING.findall(output))
    valid = 0
    valid_search = 0
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
        valid_search += int(name == "search")
        if name not in known_tool_names:
            unknown += 1

    executed = 0
    executed_search = 0
    for payload in _tool_response_payloads(output):
        if payload.get("ok") is not True:
            continue
        executed += 1
        executed_search += int(
            payload.get("tool") == "search"
            or ("tool" not in payload and "query" in payload and "results" in payload)
        )

    return {
        "attempted_tool_call_count": attempted,
        # Compatibility alias: this now explicitly means literal attempts.
        "tool_call_count": attempted,
        "valid_tool_call_count": valid,
        "valid_search_call_count": valid_search,
        "executed_tool_call_count": executed,
        "executed_search_call_count": executed_search,
        "malformed_tool_call_count": malformed,
        "unknown_tool_call_count": unknown,
    }


def _tool_response_payloads(output: str) -> list[dict[str, Any]]:
    """Decode native tool responses; malformed responses are not executions."""
    payloads: list[dict[str, Any]] = []
    for block in _TOOL_RESPONSE_BLOCK.findall(output):
        try:
            payload = json.loads(block)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _executed_search_payloads(output: str) -> list[dict[str, Any]]:
    return [
        payload
        for payload in _tool_response_payloads(output)
        if payload.get("ok") is True
        and (
            payload.get("tool") == "search"
            or ("tool" not in payload and "query" in payload and "results" in payload)
        )
    ]


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
    attempted_tool_call_counts: list[int] = []
    valid_tool_call_counts: list[int] = []
    valid_search_call_counts: list[int] = []
    executed_tool_call_counts: list[int] = []
    executed_search_call_counts: list[int] = []
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
        attempted_tool_call_counts.append(tool_stats["attempted_tool_call_count"])
        valid_tool_call_counts.append(tool_stats["valid_tool_call_count"])
        valid_search_call_counts.append(tool_stats["valid_search_call_count"])
        executed_tool_call_counts.append(tool_stats["executed_tool_call_count"])
        executed_search_call_counts.append(tool_stats["executed_search_call_count"])
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
        "attempted_tool_call_count_mean": statistics.mean(attempted_tool_call_counts),
        # Compatibility alias; it is the number of literal call openings.
        "literal_tool_call_count_mean": statistics.mean(attempted_tool_call_counts),
        "valid_tool_call_count_mean": statistics.mean(valid_tool_call_counts),
        "valid_search_call_count_mean": statistics.mean(valid_search_call_counts),
        "executed_tool_call_count_mean": statistics.mean(executed_tool_call_counts),
        "executed_search_call_count_mean": statistics.mean(executed_search_call_counts),
        "malformed_tool_call_count_mean": statistics.mean(malformed_tool_call_counts),
        "malformed_tool_call_episode_rate": sum(value > 0 for value in malformed_tool_call_counts)
        / len(rows),
        "malformed_tool_call_rate": sum(malformed_tool_call_counts)
        / max(sum(attempted_tool_call_counts), 1),
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


def _decoded_tool_calls(output: str) -> list[dict[str, Any]]:
    """Return syntactically valid tool-call payloads from a raw rollout."""
    calls: list[dict[str, Any]] = []
    for block in _TOOL_CALL_BLOCK.findall(output):
        try:
            decoded = json.loads(block)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            isinstance(decoded, dict)
            and isinstance(decoded.get("name"), str)
            and decoded["name"].strip()
            and "arguments" in decoded
        ):
            calls.append(decoded)
    return calls


def _generated_text(output: str) -> str:
    """Remove native tool observations and role separators from model text."""
    without_observations = _TOOL_RESPONSE_BLOCK.sub("", output)
    return _ROLE_LINE.sub("", without_observations).strip()


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _numeric_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0, "p50": 0.0, "p90": 0.0, "max": 0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values),
    }


def _length_histogram(values: list[int]) -> dict[str, int]:
    boundaries = (
        (0, 127, "0-127"),
        (128, 255, "128-255"),
        (256, 511, "256-511"),
        (512, 1023, "512-1023"),
    )
    histogram = {label: 0 for _, _, label in boundaries}
    histogram["1024+"] = 0
    for value in values:
        for lower, upper, label in boundaries:
            if lower <= value <= upper:
                histogram[label] += 1
                break
        else:
            histogram["1024+"] += 1
    return histogram


def analyze_verl_behavior(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any | None = None,
    supporting_titles_by_question: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Summarize native rollout behavior using actual tool executions.

    A literal ``<tool_call>`` is an attempt.  A successful ``<tool_response>``
    is an execution.  This distinction is essential because the native parser
    can reject malformed calls or terminate before executing a valid-looking
    call.  Supporting titles are optional evaluation-only metadata; when
    supplied, useful and wasted searches are computed without exposing them to
    the model.
    """
    if not rows:
        raise ValueError("at least one rollout row is required")

    search_counts: list[int] = []
    attempted_counts: list[int] = []
    valid_counts: list[int] = []
    executed_tool_counts: list[int] = []
    assistant_turn_counts: list[int] = []
    verl_turn_counts: list[int] = []
    generated_char_counts: list[int] = []
    generated_token_counts: list[int] = []
    em_by_search: dict[int, list[float]] = {}
    f1_by_search: dict[int, list[float]] = {}
    valid_by_search: dict[int, list[float]] = {}
    query_counts: Counter[str] = Counter()
    duplicate_query_episodes = 0
    duplicate_query_calls = 0
    unknown_tool_calls = 0
    valid_search_calls = 0
    malformed_tool_calls = 0
    useful_search_calls = 0
    wasted_search_calls = 0
    support_metadata_rows = 0
    first_search_recalls: list[float] = []
    second_search_count = 0
    useful_second_searches = 0

    def support_for_row(row: Mapping[str, Any]) -> set[str] | None:
        if not supporting_titles_by_question:
            return None
        input_text = row.get("input")
        if not isinstance(input_text, str):
            return None
        marker = "\nuser\n"
        if marker in input_text:
            question = input_text.split(marker, 1)[1].split("\nassistant", 1)[0].strip()
            titles = supporting_titles_by_question.get(question)
            if titles is not None:
                return set(titles)
        for question, titles in supporting_titles_by_question.items():
            if question in input_text:
                return set(titles)
        return None

    for row in rows:
        output = row.get("output", "")
        reference = row.get("gts")
        if not isinstance(output, str) or not isinstance(reference, str):
            raise ValueError("each row requires string output and gts fields")
        tool_stats = classify_tool_calls(output)
        executed_search_payloads = _executed_search_payloads(output)
        search_count = len(executed_search_payloads)
        search_counts.append(search_count)
        attempted_counts.append(tool_stats["attempted_tool_call_count"])
        valid_counts.append(tool_stats["valid_tool_call_count"])
        executed_tool_counts.append(tool_stats["executed_tool_call_count"])
        valid_search_calls += tool_stats["valid_search_call_count"]
        malformed_tool_calls += tool_stats["malformed_tool_call_count"]
        unknown_tool_calls += tool_stats["unknown_tool_call_count"]

        tool_response_count = output.count("<tool_response>")
        assistant_turns = tool_response_count + (1 if output.strip() else 0)
        assistant_turn_counts.append(assistant_turns)
        verl_turn_counts.append(assistant_turns + tool_response_count + 1)
        generated = _generated_text(output)
        generated_char_counts.append(len(generated))
        if tokenizer is not None:
            generated_token_counts.append(
                len(tokenizer.encode(generated, add_special_tokens=False))
            )

        reward = task_reward(output, reference)
        em_by_search.setdefault(search_count, []).append(float(reward["em"]))
        f1_by_search.setdefault(search_count, []).append(float(reward["f1"]))
        valid_by_search.setdefault(search_count, []).append(float(reward["valid_answer"]))

        episode_queries: list[str] = []
        support = support_for_row(row)
        support_metadata_rows += int(support is not None)
        discovered_support: set[str] = set()
        row_useful = 0
        first_search_recall = 0.0
        for search_index, payload in enumerate(executed_search_payloads, start=1):
            query = payload.get("query")
            if isinstance(query, str) and query.strip():
                normalized = " ".join(query.casefold().split())
                query_counts[normalized] += 1
                episode_queries.append(normalized)
            if support is None:
                continue
            result = payload.get("results", [])
            titles = (
                {
                    item["title"]
                    for item in result
                    if isinstance(item, dict) and isinstance(item.get("title"), str)
                }
                if isinstance(result, list)
                else set()
            )
            newly_found = (titles & support) - discovered_support
            discovered_support.update(newly_found)
            is_useful = bool(newly_found)
            row_useful += int(is_useful)
            if search_index == 1:
                first_search_recall = len(discovered_support) / max(len(support), 1)
            elif search_index == 2:
                second_search_count += 1
                useful_second_searches += int(is_useful)
        if support is not None:
            useful_search_calls += row_useful
            wasted_search_calls += max(search_count - row_useful, 0)
            first_search_recalls.append(first_search_recall)

        repeated = len(episode_queries) - len(set(episode_queries))
        duplicate_query_calls += max(repeated, 0)
        duplicate_query_episodes += int(repeated > 0)

    def bucket_metrics(bucket: dict[int, list[float]]) -> dict[int, dict[str, float]]:
        return {
            count: {"mean": statistics.mean(values), "count": len(values)}
            for count, values in sorted(bucket.items())
        }

    support_available = bool(supporting_titles_by_question) and support_metadata_rows == len(rows)
    report: dict[str, Any] = {
        "episodes": len(rows),
        "avg_search_calls": statistics.mean(search_counts),
        "avg_executed_search_calls": statistics.mean(search_counts),
        "search_count_distribution": dict(sorted(Counter(search_counts).items())),
        "executed_search_count_distribution": dict(sorted(Counter(search_counts).items())),
        "multi_search_rate": sum(count >= 2 for count in search_counts) / len(rows),
        "three_plus_search_rate": sum(count >= 3 for count in search_counts) / len(rows),
        "avg_attempted_tool_calls": statistics.mean(attempted_counts),
        "avg_valid_tool_calls": statistics.mean(valid_counts),
        "avg_executed_tool_calls": statistics.mean(executed_tool_counts),
        "attempted_tool_call_count": sum(attempted_counts),
        "valid_tool_call_count": sum(valid_counts),
        "executed_tool_call_count": sum(executed_tool_counts),
        "executed_search_call_count": sum(search_counts),
        # Keep the old key as an explicit literal-attempt alias.
        "literal_tool_call_count": sum(attempted_counts),
        "valid_search_call_count": valid_search_calls,
        "malformed_tool_call_count": malformed_tool_calls,
        "unknown_tool_call_count": unknown_tool_calls,
        "malformed_tool_call_rate": malformed_tool_calls / max(sum(attempted_counts), 1),
        "avg_turns": statistics.mean(assistant_turn_counts),
        "turn_distribution": dict(sorted(Counter(assistant_turn_counts).items())),
        "avg_assistant_turns": statistics.mean(assistant_turn_counts),
        "assistant_turn_distribution": dict(sorted(Counter(assistant_turn_counts).items())),
        "avg_verl_num_turns_estimate": statistics.mean(verl_turn_counts),
        "verl_num_turn_distribution_estimate": dict(sorted(Counter(verl_turn_counts).items())),
        "generated_character_stats": _numeric_summary(generated_char_counts),
        "duplicate_query_rate": duplicate_query_calls / max(sum(search_counts), 1),
        "duplicate_query_episode_rate": duplicate_query_episodes / len(rows),
        "unique_query_count": len(query_counts),
        "no_search_answer_rate": sum(
            bool(extract_final_answer(str(row.get("output", "")))) and count == 0
            for row, count in zip(rows, search_counts)
        )
        / len(rows),
        "accuracy_by_search_count": {
            "em": bucket_metrics(em_by_search),
            "f1": bucket_metrics(f1_by_search),
            "valid_answer": bucket_metrics(valid_by_search),
        },
        "supporting_title_metadata_available": support_available,
        "second_search_count": second_search_count,
        "second_search_useful_rate": useful_second_searches / max(second_search_count, 1),
    }
    if support_available:
        report.update(
            {
                "useful_search_call_count": useful_search_calls,
                "wasted_search_call_count": wasted_search_calls,
                "avg_useful_search_calls": useful_search_calls / len(rows),
                "avg_wasted_search_calls": wasted_search_calls / len(rows),
                "tool_efficiency": useful_search_calls / max(sum(search_counts), 1),
                "first_search_support_recall": sum(first_search_recalls) / len(rows),
            }
        )
    else:
        report.update(
            {
                "useful_search_call_count": None,
                "wasted_search_call_count": None,
                "avg_useful_search_calls": None,
                "avg_wasted_search_calls": None,
                "tool_efficiency": None,
                "first_search_support_recall": None,
            }
        )
    if tokenizer is not None:
        report["generated_token_stats"] = _numeric_summary(generated_token_counts)
        report["generated_token_histogram"] = _length_histogram(generated_token_counts)
    return report
