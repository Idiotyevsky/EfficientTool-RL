"""Behavioral analysis for stored HotpotQA agent trajectories."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..data import HotpotExample
from .metrics import answer_metrics, normalize_answer


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze_trajectories(
    trajectories: list[dict[str, Any]],
    examples: list[HotpotExample],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {example.example_id: example for example in examples}
    search_counts: Counter[int] = Counter()
    accuracy_by_search: dict[int, list[float]] = defaultdict(list)
    total_queries = 0
    duplicate_queries = 0
    early_answers = 0
    supporting_recalls: list[float] = []
    categorized: list[dict[str, Any]] = []

    for trajectory in trajectories:
        example = by_id[trajectory["episode_id"]]
        calls = int(trajectory.get("tool_calls", 0))
        search_counts[calls] += 1
        scores = answer_metrics(trajectory.get("final_answer") or "", example.answer)
        accuracy_by_search[calls].append(scores["exact_match"])
        queries: list[str] = []
        retrieved_titles: set[str] = set()
        for step in trajectory.get("steps", []):
            action = step.get("action", {})
            if action.get("kind") == "tool_call" and action.get("name") == "search":
                query = action.get("arguments", {}).get("query")
                if isinstance(query, str):
                    queries.append(normalize_answer(query))
                result = (step.get("observation") or {}).get("result", [])
                if isinstance(result, list):
                    retrieved_titles.update(
                        item["title"] for item in result if isinstance(item, dict) and "title" in item
                    )
        total_queries += len(queries)
        duplicate_queries += len(queries) - len(set(queries))
        early_answers += int(calls == 0)
        support = set(example.supporting_titles)
        recall = len(retrieved_titles & support) / max(len(support), 1)
        supporting_recalls.append(recall)

        category = "success"
        if not scores["exact_match"]:
            if trajectory.get("invalid_actions", 0):
                category = "malformed_action"
            elif calls == 0:
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
                "search_calls": calls,
                "supporting_title_recall": recall,
                "failure_category": category,
                "queries": queries,
                "retrieved_titles": sorted(retrieved_titles),
            }
        )

    count = len(trajectories)
    report = {
        "episodes": count,
        "search_count_distribution": dict(sorted(search_counts.items())),
        "duplicate_query_rate": duplicate_queries / max(total_queries, 1),
        "early_answer_rate": early_answers / max(count, 1),
        "avg_supporting_title_recall": sum(supporting_recalls) / max(count, 1),
        "accuracy_by_search_count": {
            str(calls): sum(values) / len(values)
            for calls, values in sorted(accuracy_by_search.items())
        },
        "failure_categories": dict(
            sorted(Counter(item["failure_category"] for item in categorized).items())
        ),
    }
    return report, categorized
