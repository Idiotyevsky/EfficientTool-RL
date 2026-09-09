"""verl custom-reward adapter for the process-aware composite reward.

Weights are supplied through verl's ``custom_reward_function.reward_kwargs``
and default to the 0.8 / 0.15 / 0.05 pilot values.
"""

from __future__ import annotations

from typing import Any

from efficienttool_rl.rewards.composite import (
    DEFAULT_ANSWER_WEIGHT,
    DEFAULT_EVIDENCE_WEIGHT,
    DEFAULT_FORMAT_WEIGHT,
    composite_reward,
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    answer_weight: float = DEFAULT_ANSWER_WEIGHT,
    evidence_weight: float = DEFAULT_EVIDENCE_WEIGHT,
    format_weight: float = DEFAULT_FORMAT_WEIGHT,
    **kwargs: Any,
) -> dict[str, float]:
    del data_source, kwargs
    return composite_reward(
        solution_str,
        ground_truth,
        extra_info=extra_info,
        answer_weight=answer_weight,
        evidence_weight=evidence_weight,
        format_weight=format_weight,
    )
