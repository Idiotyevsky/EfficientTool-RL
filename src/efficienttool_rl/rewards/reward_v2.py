"""Reward v2 for multi-turn search-agent GRPO.

The trajectory-level objective is::

    R = R_answer + beta_t * R_marginal_evidence
        - wasted_search_lambda * R_answer * N_wasted

``R_answer`` is the unchanged ``0.5 * EM + 0.5 * token-F1`` task reward.
Marginal evidence is accumulated in execution order: each successful search
receives the increase in unique gold supporting-title coverage, so duplicate
or irrelevant retrievals have zero gain.  The scalar sum is used as a
trajectory-level reward because stock GRPO converts sequence rewards into a
group-relative outcome advantage; assigning the scalar to individual search
tokens would not provide action-local credit without changing the algorithm.

Gold answers and supporting titles are reward-only metadata.  They are never
added to prompts, tool queries, observations, or other model-visible fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .cost_aware import _supporting_titles, search_usage
from .parsing import executed_search_payloads
from .task import task_reward

DEFAULT_BETA_START = 0.10
DEFAULT_BETA_END = 0.0
DEFAULT_WASTED_SEARCH_LAMBDA = 0.02


def linear_annealed_beta(
    global_step: int,
    total_training_steps: int,
    *,
    beta_start: float = DEFAULT_BETA_START,
    beta_end: float = DEFAULT_BETA_END,
) -> float:
    """Linearly anneal beta over optimizer steps, not agent turns.

    Step 1 receives ``beta_start`` and ``total_training_steps`` receives
    ``beta_end``.  Step 0 is the pre-training validation boundary and is
    clamped to the start value.
    """
    if total_training_steps < 1:
        raise ValueError("total_training_steps must be positive")
    if beta_start < 0 or beta_end < 0:
        raise ValueError("evidence beta values must be non-negative")

    step = int(global_step)
    if total_training_steps == 1:
        return float(beta_end if step >= 1 else beta_start)
    bounded_step = min(max(step, 1), total_training_steps)
    progress = (bounded_step - 1) / (total_training_steps - 1)
    return float(beta_start + progress * (beta_end - beta_start))


def _normalized_query(payload: Mapping[str, Any]) -> str | None:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return None
    return " ".join(query.casefold().split())


def marginal_evidence_progress(
    solution_str: str,
    extra_info: Any,
    *,
    require_supporting_titles: bool = True,
) -> dict[str, Any]:
    """Return per-search marginal gains and supporting-evidence diagnostics.

    For search ``t``, the gain is ``C_t - C_(t-1)``, where coverage is the
    fraction of unique gold supporting titles observed so far.  Search order
    is preserved, and a supporting title can contribute at most once.
    """
    support = _supporting_titles(extra_info)
    if require_supporting_titles and not support:
        raise ValueError("supporting_titles metadata is required for Reward v2")

    discovered: set[str] = set()
    gains: list[float] = []
    seen_queries: set[str] = set()
    repeated_search_count = 0
    payloads = executed_search_payloads(solution_str)

    for payload in payloads:
        results = payload.get("results", [])
        titles = {
            item["title"]
            for item in results
            if isinstance(item, Mapping) and isinstance(item.get("title"), str)
        } if isinstance(results, list) else set()
        newly_found = (titles & support) - discovered
        discovered.update(newly_found)
        gains.append(len(newly_found) / len(support) if support else 0.0)

        normalized_query = _normalized_query(payload)
        if normalized_query is not None:
            repeated_search_count += int(normalized_query in seen_queries)
            seen_queries.add(normalized_query)

    return {
        "gains": tuple(gains),
        "marginal_evidence_reward": float(sum(gains)),
        "unique_support_count": len(discovered),
        "gold_support_count": len(support),
        "repeated_search_count": repeated_search_count,
        "evidence_metadata_available": float(bool(support)),
    }


def reward_v2(
    solution_str: str,
    ground_truth: str,
    *,
    extra_info: Any = None,
    global_step: int = 0,
    total_training_steps: int,
    beta_start: float = DEFAULT_BETA_START,
    beta_end: float = DEFAULT_BETA_END,
    wasted_search_lambda: float = DEFAULT_WASTED_SEARCH_LAMBDA,
    require_supporting_titles: bool = True,
) -> dict[str, float | int]:
    """Compute the final Reward v2 trajectory score and scalar diagnostics."""
    if wasted_search_lambda < 0:
        raise ValueError("wasted_search_lambda must be non-negative")

    answer = task_reward(solution_str, ground_truth, alpha=0.5)
    # Reuse the project's canonical useful/wasted accounting for the penalty.
    usage = search_usage(solution_str, extra_info)
    evidence = marginal_evidence_progress(
        solution_str,
        extra_info,
        require_supporting_titles=require_supporting_titles,
    )
    gains = evidence["gains"]
    if usage["executed_search_calls"] != len(gains):
        raise RuntimeError("Reward v2 search accounting disagrees with parsed executions")
    if usage["useful_search_calls"] != sum(gain > 0 for gain in gains):
        raise RuntimeError("Reward v2 usefulness accounting disagrees with marginal gains")

    current_beta = linear_annealed_beta(
        global_step,
        total_training_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    answer_reward = float(answer["score"])
    marginal_reward = float(evidence["marginal_evidence_reward"])
    wasted_count = int(usage["wasted_search_calls"])
    wasted_penalty = float(wasted_search_lambda) * answer_reward * wasted_count
    total_reward = answer_reward + current_beta * marginal_reward - wasted_penalty
    search_count = int(usage["executed_search_calls"])

    return {
        "score": total_reward,
        "total_reward": total_reward,
        "answer_reward": answer_reward,
        "task_reward": answer_reward,
        "marginal_evidence_reward": marginal_reward,
        "current_beta": current_beta,
        "wasted_search_count": wasted_count,
        "wasted_search_penalty": wasted_penalty,
        "search_count": search_count,
        "useful_search_count": int(usage["useful_search_calls"]),
        "unique_support_count": int(evidence["unique_support_count"]),
        "multi_search": float(search_count >= 2),
        "evidence_gain_per_search": marginal_reward / max(search_count, 1),
        "repeated_search_count": int(evidence["repeated_search_count"]),
        "global_step": int(global_step),
        "em": float(answer["em"]),
        "f1": float(answer["f1"]),
        "valid_answer": float(answer["valid_answer"]),
        "evidence_metadata_available": float(evidence["evidence_metadata_available"]),
        **usage,
    }


__all__ = [
    "DEFAULT_BETA_END",
    "DEFAULT_BETA_START",
    "DEFAULT_WASTED_SEARCH_LAMBDA",
    "linear_annealed_beta",
    "marginal_evidence_progress",
    "reward_v2",
]
