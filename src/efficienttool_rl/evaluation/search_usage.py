"""Executed, useful, and wasted search accounting for local agent episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def episode_search_usage(episode: Any, supporting_titles: Sequence[str]) -> dict[str, int]:
    """Count executed, useful, and wasted searches in one local episode.

    A search is useful when its executed observation contains at least one new
    title from the sample's evaluation-only supporting-title annotation. The
    same definition is applied to native verl rollouts by
    :func:`efficienttool_rl.rewards.cost_aware.search_usage`.
    """
    support = set(supporting_titles)
    discovered: set[str] = set()
    executed = 0
    useful = 0
    for step in episode.steps:
        if not step.tool_executed:
            continue
        action = step.action
        if action.get("kind") != "tool_call" or action.get("name") != "search":
            continue
        executed += 1
        observation = step.observation or {}
        raw_results = observation.get("result", [])
        titles = {
            item.get("title")
            for item in raw_results
            if isinstance(item, Mapping) and isinstance(item.get("title"), str)
        } if isinstance(raw_results, list) else set()
        new_titles = (titles & support) - discovered
        discovered.update(new_titles)
        useful += int(bool(new_titles))
    return {
        "executed_search_calls": executed,
        "useful_search_calls": useful,
        "wasted_search_calls": max(executed - useful, 0),
    }


__all__ = ["episode_search_usage"]
