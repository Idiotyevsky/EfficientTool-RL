"""Success-gated cost-aware rewards for native HotpotQA tool trajectories."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .task import task_reward


_TOOL_RESPONSE_BLOCK = re.compile(r"<tool_response>(.*?)</tool_response>", flags=re.DOTALL)


def _as_mapping(extra_info: Any) -> Mapping[str, Any] | None:
    """Unwrap common array-like containers used by verl's batch plumbing."""
    if isinstance(extra_info, Mapping):
        return extra_info
    item = getattr(extra_info, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):
            return None
        if isinstance(value, Mapping):
            return value
    return None


def _supporting_titles(extra_info: Any) -> set[str]:
    info = _as_mapping(extra_info)
    if info is None:
        return set()
    value = info.get("supporting_titles", ())
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return {item for item in value if isinstance(item, str)}
    return set()


def _executed_search_payloads(solution_str: str) -> list[dict[str, Any]]:
    """Return successful native search responses from a rollout string."""
    payloads: list[dict[str, Any]] = []
    if not isinstance(solution_str, str):
        return payloads
    for block in _TOOL_RESPONSE_BLOCK.findall(solution_str):
        try:
            payload = json.loads(block)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        is_search = payload.get("tool") == "search" or (
            "tool" not in payload and "query" in payload and "results" in payload
        )
        if is_search:
            payloads.append(payload)
    return payloads


def search_usage(solution_str: str, extra_info: Any) -> dict[str, int]:
    """Count executed, useful, and wasted searches in one native trajectory.

    A search is useful when its successful response contains at least one new
    title from the sample's evaluation-only supporting-title annotation.
    Supporting titles are never read from the model prompt or tool kwargs.
    """
    support = _supporting_titles(extra_info)
    discovered: set[str] = set()
    useful = 0
    payloads = _executed_search_payloads(solution_str)
    for payload in payloads:
        results = payload.get("results", [])
        titles = {
            item["title"]
            for item in results
            if isinstance(item, Mapping) and isinstance(item.get("title"), str)
        } if isinstance(results, list) else set()
        newly_found = (titles & support) - discovered
        discovered.update(newly_found)
        useful += int(bool(newly_found))
    executed = len(payloads)
    return {
        "executed_search_calls": executed,
        "useful_search_calls": useful,
        "wasted_search_calls": max(executed - useful, 0),
    }


def cost_aware_reward(
    solution_str: str,
    ground_truth: str,
    *,
    extra_info: Any = None,
    lambda_cost: float = 0.05,
    require_supporting_titles: bool = True,
) -> dict[str, float | int]:
    """Compute task reward minus success-gated wasted-search cost.

    The task score is the existing '0.5 * EM + 0.5 * F1' signal. Useful
    searches are free; only executed searches that add no new supporting title
    are charged. Multiplying the penalty by task score prevents zero-quality
    guesses from being rewarded simply for making fewer searches.
    """
    if lambda_cost < 0:
        raise ValueError("lambda_cost must be non-negative")
    if require_supporting_titles and not _supporting_titles(extra_info):
        raise ValueError("supporting_titles metadata is required for cost-aware reward")

    task = task_reward(solution_str, ground_truth, alpha=0.5)
    usage = search_usage(solution_str, extra_info)
    penalty = float(lambda_cost) * task["score"] * usage["wasted_search_calls"]
    total = task["score"] - penalty
    return {
        **task,
        "task_reward": task["score"],
        "cost_penalty": penalty,
        "total_reward": total,
        "score": total,
        **usage,
        "cost_metadata_available": 1.0 if _supporting_titles(extra_info) else 0.0,
    }
