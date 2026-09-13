"""verl custom-reward adapter for SearchAgent-RL Reward v2."""

from __future__ import annotations

from typing import Any

from efficienttool_rl.rewards.reward_v2 import (
    DEFAULT_BETA_END,
    DEFAULT_BETA_START,
    DEFAULT_WASTED_SEARCH_LAMBDA,
    reward_v2,
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    *,
    global_step: int = 0,
    total_training_steps: int,
    beta_start: float = DEFAULT_BETA_START,
    beta_end: float = DEFAULT_BETA_END,
    wasted_search_lambda: float = DEFAULT_WASTED_SEARCH_LAMBDA,
    require_supporting_titles: bool = True,
    **kwargs: Any,
) -> dict[str, float | int]:
    """Score one trajectory; the project-local manager supplies global_step."""
    del data_source, kwargs
    return reward_v2(
        solution_str,
        ground_truth,
        extra_info=extra_info,
        global_step=global_step,
        total_training_steps=total_training_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        wasted_search_lambda=wasted_search_lambda,
        require_supporting_titles=require_supporting_titles,
    )


__all__ = ["compute_score"]
