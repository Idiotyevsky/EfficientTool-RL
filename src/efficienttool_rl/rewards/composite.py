"""Process-aware composite reward: answer + evidence coverage + format.

Objective::

    R = answer_weight * R_answer
      + evidence_weight * R_evidence
      + format_weight * R_format

- ``R_answer`` is the existing task-only signal ``0.5 * EM + 0.5 * F1``.
- ``R_evidence`` is document-level supporting-evidence coverage:
  the fraction of gold supporting titles that appear in successful search
  responses. HotpotQA supporting facts are sentence-level, but the search
  environment returns documents, so coverage is measured at document level.
  Coverage is saturating: once every gold title has been retrieved, further
  searches cannot increase the reward, and repeated retrievals of the same
  title are counted once.
- ``R_format`` averages three binary protocol checks: a single valid terminal
  ``<answer>`` block, no malformed or unknown tool calls, and a trajectory
  that terminates with the answer.

Gold supporting titles are read only from evaluation-only ``extra_info``
metadata; they are never present in prompts or tool kwargs. Supporting titles
never dominate training: weights are configurable and the answer component is
expected to carry the largest weight (pilot: 0.8 / 0.15 / 0.05).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .cost_aware import _supporting_titles, search_usage
from .parsing import classify_tool_calls, executed_search_payloads
from .task import extract_final_answer, task_reward

_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
_TOOL_RESPONSE_BLOCK = re.compile(r"<tool_response>.*?</tool_response>", flags=re.DOTALL)

DEFAULT_ANSWER_WEIGHT = 0.8
DEFAULT_EVIDENCE_WEIGHT = 0.15
DEFAULT_FORMAT_WEIGHT = 0.05


def evidence_coverage(
    retrieved_titles: Iterable[str],
    gold_titles: Iterable[str],
) -> dict[str, float]:
    """Compute saturating document-level supporting-evidence coverage."""
    gold = {title for title in gold_titles if isinstance(title, str) and title}
    retrieved = {title for title in retrieved_titles if isinstance(title, str) and title}
    if not gold:
        return {
            "evidence_coverage": 0.0,
            "evidence_gold_count": 0.0,
            "evidence_found_count": 0.0,
        }
    found = len(retrieved & gold)
    return {
        "evidence_coverage": found / len(gold),
        "evidence_gold_count": float(len(gold)),
        "evidence_found_count": float(found),
    }


def retrieved_titles_from_text(response: str) -> list[str]:
    """Return titles from successful native search responses."""
    titles: list[str] = []
    for payload in executed_search_payloads(response):
        results = payload.get("results", [])
        if not isinstance(results, list):
            continue
        for item in results:
            if isinstance(item, Mapping) and isinstance(item.get("title"), str):
                titles.append(item["title"])
    return titles


def format_components_from_text(response: str) -> dict[str, float]:
    """Return the three binary protocol checks for one native trajectory."""
    has_valid_answer = extract_final_answer(response) is not None
    calls = classify_tool_calls(response)
    no_malformed = float(
        calls["malformed_tool_call_count"] == 0 and calls["unknown_tool_call_count"] == 0
    )
    answer_matches = list(_ANSWER_BLOCK.finditer(response)) if isinstance(response, str) else []
    tool_response_ends = [
        match.end() for match in _TOOL_RESPONSE_BLOCK.finditer(response)
    ] if isinstance(response, str) else []
    ended_with_answer = float(
        bool(answer_matches)
        and (not tool_response_ends or answer_matches[-1].end() > tool_response_ends[-1])
    )
    return {
        "format_valid_answer": float(has_valid_answer),
        "format_no_malformed_calls": no_malformed,
        "format_ended_with_answer": ended_with_answer,
    }


def _check_weights(
    answer_weight: float, evidence_weight: float, format_weight: float
) -> None:
    weights = (answer_weight, evidence_weight, format_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("reward weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("at least one reward weight must be positive")


def composite_reward(
    response: str,
    reference: str,
    *,
    extra_info: Any = None,
    answer_weight: float = DEFAULT_ANSWER_WEIGHT,
    evidence_weight: float = DEFAULT_EVIDENCE_WEIGHT,
    format_weight: float = DEFAULT_FORMAT_WEIGHT,
) -> dict[str, float]:
    """Compute the composite reward for one native verl trajectory."""
    _check_weights(answer_weight, evidence_weight, format_weight)
    answer = task_reward(response, reference, alpha=0.5)
    gold = _supporting_titles(extra_info)
    coverage = evidence_coverage(retrieved_titles_from_text(response), gold)
    format_parts = format_components_from_text(response)
    usage = search_usage(response, extra_info)
    total = (
        answer_weight * answer["score"]
        + evidence_weight * coverage["evidence_coverage"]
        + format_weight * (sum(format_parts.values()) / len(format_parts))
    )
    return {
        "score": total,
        "answer_reward": answer["score"],
        "evidence_reward": coverage["evidence_coverage"],
        "format_reward": sum(format_parts.values()) / len(format_parts),
        "em": answer["em"],
        "f1": answer["f1"],
        "valid_answer": answer["valid_answer"],
        "evidence_metadata_available": 1.0 if gold else 0.0,
        **coverage,
        **format_parts,
        **usage,
    }


def retrieved_titles_from_episode(episode: Mapping[str, Any]) -> list[str]:
    """Return titles retrieved by executed searches in one local episode dict."""
    titles: list[str] = []
    for step in episode.get("steps", []):
        if not step.get("tool_executed"):
            continue
        action = step.get("action") or {}
        if action.get("kind") != "tool_call" or action.get("name") != "search":
            continue
        observation = step.get("observation") or {}
        results = observation.get("result", [])
        if not isinstance(results, list):
            continue
        for item in results:
            if isinstance(item, Mapping) and isinstance(item.get("title"), str):
                titles.append(item["title"])
    return titles


def episode_format_components(episode: Mapping[str, Any]) -> dict[str, float]:
    """Return the three binary protocol checks for one local episode dict."""
    return {
        "format_valid_answer": float(episode.get("final_answer") is not None),
        "format_no_malformed_calls": float(int(episode.get("invalid_actions", 0)) == 0),
        "format_ended_with_answer": float(
            episode.get("termination_reason") == "final_answer"
        ),
    }


def episode_search_usage_counts(episode: Mapping[str, Any]) -> dict[str, int]:
    """Executed/useful/wasted searches for one local episode dict."""
    executed = 0
    useful = 0
    seen: set[str] = set()
    for step in episode.get("steps", []):
        if not step.get("tool_executed"):
            continue
        action = step.get("action") or {}
        if action.get("kind") != "tool_call" or action.get("name") != "search":
            continue
        executed += 1
        observation = step.get("observation") or {}
        results = observation.get("result", [])
        titles = {
            item.get("title")
            for item in (results if isinstance(results, list) else [])
            if isinstance(item, Mapping) and isinstance(item.get("title"), str)
        }
        new_titles = {title for title in titles if title not in seen}
        seen.update(new_titles)
        useful += int(bool(new_titles))
    return {
        "executed_search_calls": executed,
        "useful_search_calls": useful,
        "wasted_search_calls": max(executed - useful, 0),
    }


def episode_composite_reward(
    episode: Mapping[str, Any],
    reference: str,
    *,
    supporting_titles: Sequence[str] = (),
    answer_weight: float = DEFAULT_ANSWER_WEIGHT,
    evidence_weight: float = DEFAULT_EVIDENCE_WEIGHT,
    format_weight: float = DEFAULT_FORMAT_WEIGHT,
) -> dict[str, float]:
    """Compute the composite reward for one stored local trajectory dict.

    Used for offline reward-distribution validation on fixed-policy
    trajectories; it applies the same components and weights as
    :func:`composite_reward` for native verl rollouts.
    """
    _check_weights(answer_weight, evidence_weight, format_weight)
    format_parts = episode_format_components(episode)
    answer_text = episode.get("final_answer")
    metrics = task_reward(
        f"<answer>{answer_text}</answer>" if answer_text else "",
        reference,
        alpha=0.5,
    )
    gold = {title for title in supporting_titles if isinstance(title, str) and title}
    coverage = evidence_coverage(retrieved_titles_from_episode(episode), gold)
    format_score = sum(format_parts.values()) / len(format_parts)
    usage = episode_search_usage_counts(episode)
    total = (
        answer_weight * metrics["score"]
        + evidence_weight * coverage["evidence_coverage"]
        + format_weight * format_score
    )
    return {
        "score": total,
        "answer_reward": metrics["score"],
        "evidence_reward": coverage["evidence_coverage"],
        "format_reward": format_score,
        "em": metrics["em"],
        "f1": metrics["f1"],
        "valid_answer": metrics["valid_answer"],
        "evidence_metadata_available": 1.0 if gold else 0.0,
        **coverage,
        **format_parts,
        **usage,
    }


__all__ = [
    "DEFAULT_ANSWER_WEIGHT",
    "DEFAULT_EVIDENCE_WEIGHT",
    "DEFAULT_FORMAT_WEIGHT",
    "composite_reward",
    "evidence_coverage",
    "episode_composite_reward",
    "episode_format_components",
    "episode_search_usage_counts",
    "format_components_from_text",
    "retrieved_titles_from_episode",
    "retrieved_titles_from_text",
]
