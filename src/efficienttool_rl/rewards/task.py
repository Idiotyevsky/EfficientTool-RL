"""Task-only HotpotQA reward used by vanilla GRPO sanity experiments."""

from __future__ import annotations

import re

from ..evaluation.metrics import answer_metrics

_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
_TOOL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", flags=re.DOTALL)
_TOOL_RESPONSE_BLOCK = re.compile(r"<tool_response>.*?</tool_response>", flags=re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_ROLE_LINE = re.compile(r"(?m)^(?:system|user|assistant|tool)\s*$")


def extract_final_answer(response: str) -> str | None:
    """Return one terminal tagged answer from local or native verl output.

    Native multi-turn rollouts include serialized tool observations and role
    markers in ``solution_str``.  Those scaffolding tokens are not model
    answers, so remove them before applying the same strict single-answer
    format check used by the local agent runner.
    """
    if not isinstance(response, str):
        return None
    answer_only = _TOOL_BLOCK.sub("", response).strip()
    answer_only = _TOOL_RESPONSE_BLOCK.sub("", answer_only)
    answer_only = _THINK_BLOCK.sub("", answer_only)
    answer_only = _ROLE_LINE.sub("", answer_only).strip()
    if answer_only.count("<answer>") != 1 or answer_only.count("</answer>") != 1:
        return None
    match = _ANSWER_BLOCK.fullmatch(answer_only)
    if match is None or not match.group(1).strip():
        return None
    return match.group(1).strip()


def task_reward(response: str, reference: str, *, alpha: float = 0.5) -> dict[str, float]:
    """Compute alpha * EM + (1-alpha) * token F1 with no shaping terms."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    answer = extract_final_answer(response)
    if answer is None:
        return {"score": 0.0, "em": 0.0, "f1": 0.0, "valid_answer": 0.0}
    metrics = answer_metrics(answer, reference)
    score = alpha * metrics["exact_match"] + (1 - alpha) * metrics["f1"]
    return {
        "score": score,
        "em": metrics["exact_match"],
        "f1": metrics["f1"],
        "valid_answer": 1.0,
    }
