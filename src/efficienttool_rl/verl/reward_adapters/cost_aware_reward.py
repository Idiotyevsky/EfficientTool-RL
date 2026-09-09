"""verl custom-reward adapter for the isolated M5 cost-aware smoke."""

from __future__ import annotations

from typing import Any

from efficienttool_rl.rewards.cost_aware import cost_aware_reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    lambda_cost: float = 0.05,
    require_supporting_titles: bool = True,
    **kwargs: Any,
) -> dict[str, float | int]:
    """Adapter called by verl's naive reward manager."""
    del data_source, kwargs
    return cost_aware_reward(
        solution_str,
        ground_truth,
        extra_info=extra_info,
        lambda_cost=lambda_cost,
        require_supporting_titles=require_supporting_titles,
    )
