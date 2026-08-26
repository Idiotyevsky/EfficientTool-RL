"""verl custom-reward adapter for the approved task-only M3 objective."""

from __future__ import annotations

from typing import Any

from efficienttool_rl.rewards import task_reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    del data_source, extra_info, kwargs
    return task_reward(solution_str, ground_truth, alpha=0.5)
