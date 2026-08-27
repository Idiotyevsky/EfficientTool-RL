"""Deterministic dataset filters for information-demanding Hotpot episodes."""

from __future__ import annotations

from .hotpotqa import HotpotExample
from ..evaluation.metrics import normalize_answer
from ..tools.search import BM25Search


def is_two_hop_candidate(
    example: HotpotExample,
    *,
    max_observation_tokens: int = 384,
) -> bool:
    """Select bridge examples whose question-level first hop is incomplete.

    The filter is applied before rollout and does not alter observations. It
    keeps examples where the deterministic question query retrieves exactly one
    supporting title, while the answer is absent from that first passage and
    from the question itself. The second supporting passage remains available
    to a later targeted search.
    """
    if example.question_type != "bridge" or len(example.supporting_titles) < 2:
        return False
    first = BM25Search(
        example.passages,
        max_observation_tokens=max_observation_tokens,
    ).search(example.question, top_k=1)[0]
    first_support_count = int(first.title in set(example.supporting_titles))
    answer = normalize_answer(example.answer)
    return (
        first_support_count == 1
        and answer not in normalize_answer(first.passage)
        and answer not in normalize_answer(example.question)
    )
