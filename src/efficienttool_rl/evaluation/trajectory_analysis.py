"""Behavioral analysis for stored HotpotQA agent trajectories."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..data import HotpotExample
from ..protocol import count_tool_call_attempts
from .metrics import answer_metrics, normalize_answer


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze_trajectories(
    trajectories: list[dict[str, Any]],
    examples: list[HotpotExample],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {example.example_id: example for example in examples}
    executed_search_counts: Counter[int] = Counter()
    accuracy_by_search: dict[int, list[float]] = defaultdict(list)
    total_executed_queries = 0
    duplicate_executed_queries = 0
    early_answers = 0
    supporting_recalls: list[float] = []
    first_search_recalls: list[float] = []
    second_search_count = 0
    useful_second_searches = 0
    successful_multiturn_episodes = 0
    total_attempted = 0
    total_valid = 0
    total_executed = 0
    total_useful = 0
    total_wasted = 0
    categorized: list[dict[str, Any]] = []

    for trajectory in trajectories:
        example = by_id[trajectory["episode_id"]]
        steps = trajectory.get("steps", [])
        attempted = int(
            trajectory.get(
                "attempted_tool_calls",
                sum(count_tool_call_attempts(step.get("model_output", "")) for step in steps),
            )
        )
        valid = int(
            trajectory.get(
                "valid_tool_calls",
                sum(step.get("action", {}).get("kind") == "tool_call" for step in steps),
            )
        )
        executed = int(
            trajectory.get(
                "executed_search_calls",
                trajectory.get("executed_tool_calls", trajectory.get("tool_calls", 0)),
            )
        )
        total_attempted += attempted
        total_valid += valid
        total_executed += executed
        scores = answer_metrics(trajectory.get("final_answer") or "", example.answer)
        executed_search_counts[executed] += 1
        accuracy_by_search[executed].append(scores["exact_match"])
        queries: list[str] = []
        retrieved_titles: set[str] = set()
        useful_searches = 0
        first_search_recall = 0.0
        second_search_was_useful = False
        executed_search_index = 0
        support = set(example.supporting_titles)
        discovered_support: set[str] = set()
        for step in steps:
            action = step.get("action", {})
            if action.get("kind") == "tool_call" and action.get("name") == "search":
                if "tool_executed" in step:
                    was_executed = bool(step["tool_executed"])
                else:
                    observation = step.get("observation") or {}
                    error_code = (observation.get("error") or {}).get("code")
                    was_executed = bool(observation) and error_code not in {
                        "unknown_tool",
                        "tool_budget_exhausted",
                    }
                if not was_executed:
                    continue
                executed_search_index += 1
                query = action.get("arguments", {}).get("query")
                if isinstance(query, str):
                    queries.append(normalize_answer(query))
                result = (step.get("observation") or {}).get("result", [])
                newly_found: set[str] = set()
                if isinstance(result, list):
                    titles = {
                        item["title"]
                        for item in result
                        if isinstance(item, dict) and isinstance(item.get("title"), str)
                    }
                    retrieved_titles.update(titles)
                    newly_found = (titles & support) - discovered_support
                    discovered_support.update(newly_found)
                is_useful = bool(newly_found)
                useful_searches += int(is_useful)
                if executed_search_index == 1:
                    first_search_recall = len(discovered_support) / max(len(support), 1)
                elif executed_search_index == 2:
                    second_search_count += 1
                    useful_second_searches += int(is_useful)
                    second_search_was_useful = is_useful
        total_executed_queries += len(queries)
        duplicate_executed_queries += len(queries) - len(set(queries))
        early_answers += int(executed == 0)
        total_useful += useful_searches
        total_wasted += max(executed - useful_searches, 0)
        recall = len(retrieved_titles & support) / max(len(support), 1)
        supporting_recalls.append(recall)
        first_search_recalls.append(first_search_recall)
        successful_multiturn_episodes += int(executed >= 2 and scores["exact_match"] > 0)

        category = "success"
        if not scores["exact_match"]:
            if trajectory.get("invalid_actions", 0):
                category = "malformed_action"
            elif executed == 0:
                category = "premature_answer"
            elif len(queries) != len(set(queries)):
                category = "repeated_search"
            elif recall == 0:
                category = "poor_query_or_missing_evidence"
            elif recall < 1:
                category = "insufficient_search"
            else:
                category = "correct_evidence_wrong_answer"
        categorized.append(
            {
                "episode_id": example.example_id,
                "question": example.question,
                "reference_answer": example.answer,
                "final_answer": trajectory.get("final_answer"),
                "exact_match": scores["exact_match"],
                "f1": scores["f1"],
                "attempted_tool_calls": attempted,
                "valid_tool_calls": valid,
                "executed_search_calls": executed,
                "useful_search_calls": useful_searches,
                "wasted_search_calls": max(executed - useful_searches, 0),
                # Compatibility alias with the new executed-search meaning.
                "search_calls": executed,
                "supporting_title_recall": recall,
                "first_search_support_recall": first_search_recall,
                "second_search_was_useful": second_search_was_useful,
                "failure_category": category,
                "queries": queries,
                "retrieved_titles": sorted(retrieved_titles),
            }
        )

    count = len(trajectories)
    report = {
        "episodes": count,
        "attempted_tool_call_count": total_attempted,
        "valid_tool_call_count": total_valid,
        "executed_search_call_count": total_executed,
        "useful_search_call_count": total_useful,
        "wasted_search_call_count": total_wasted,
        "avg_attempted_tool_calls": total_attempted / max(count, 1),
        "avg_valid_tool_calls": total_valid / max(count, 1),
        "avg_executed_search_calls": total_executed / max(count, 1),
        "avg_useful_search_calls": total_useful / max(count, 1),
        "avg_wasted_search_calls": total_wasted / max(count, 1),
        "search_count_distribution": dict(sorted(executed_search_counts.items())),
        "executed_search_count_distribution": dict(sorted(executed_search_counts.items())),
        "multi_search_rate": sum(count >= 2 for count in executed_search_counts.elements())
        / max(count, 1),
        "three_plus_search_rate": sum(count >= 3 for count in executed_search_counts.elements())
        / max(count, 1),
        "tool_efficiency": total_useful / max(total_executed, 1),
        "duplicate_query_rate": duplicate_executed_queries / max(total_executed_queries, 1),
        "early_answer_rate": early_answers / max(count, 1),
        "avg_supporting_title_recall": sum(supporting_recalls) / max(count, 1),
        "first_search_support_recall": sum(first_search_recalls) / max(count, 1),
        "second_search_count": second_search_count,
        "second_search_useful_rate": useful_second_searches / max(second_search_count, 1),
        "successful_multiturn_episode_rate": successful_multiturn_episodes / max(count, 1),
        "accuracy_by_search_count": {
            str(calls): sum(values) / len(values)
            for calls, values in sorted(accuracy_by_search.items())
        },
        "failure_categories": dict(
            sorted(Counter(item["failure_category"] for item in categorized).items())
        ),
    }
    return report, categorized
