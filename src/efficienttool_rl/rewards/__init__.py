"""Research reward functions."""

from .composite import (
    composite_reward,
    episode_composite_reward,
    evidence_coverage,
    format_components_from_text,
)
from .reward_v2 import (
    linear_annealed_beta,
    marginal_evidence_progress,
    reward_v2,
)
from .task import extract_final_answer, task_reward

__all__ = [
    "composite_reward",
    "episode_composite_reward",
    "evidence_coverage",
    "extract_final_answer",
    "format_components_from_text",
    "linear_annealed_beta",
    "marginal_evidence_progress",
    "reward_v2",
    "task_reward",
]
