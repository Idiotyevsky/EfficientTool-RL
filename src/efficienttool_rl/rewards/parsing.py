"""Tool-call parsing shared by rewards and rollout analysis.

This module lives in the rewards layer so reward functions can parse native
multi-turn tool interactions without importing the evaluation layer, which
itself replays reward functions.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", flags=re.DOTALL)
_TOOL_CALL_OPENING = re.compile(r"<tool_call>")
_TOOL_RESPONSE_BLOCK = re.compile(r"<tool_response>(.*?)</tool_response>", flags=re.DOTALL)

_EMPTY_CLASSIFICATION: dict[str, int] = {
    "attempted_tool_call_count": 0,
    "tool_call_count": 0,
    "valid_tool_call_count": 0,
    "valid_search_call_count": 0,
    "executed_tool_call_count": 0,
    "executed_search_call_count": 0,
    "malformed_tool_call_count": 0,
    "unknown_tool_call_count": 0,
}


def tool_response_payloads(output: str) -> list[dict[str, Any]]:
    """Decode native tool responses; malformed responses are not executions."""
    payloads: list[dict[str, Any]] = []
    for block in _TOOL_RESPONSE_BLOCK.findall(output):
        try:
            payload = json.loads(block)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def executed_search_payloads(output: str) -> list[dict[str, Any]]:
    """Return successful native search responses from a rollout string."""
    return [
        payload
        for payload in tool_response_payloads(output)
        if payload.get("ok") is True
        and (
            payload.get("tool") == "search"
            or ("tool" not in payload and "query" in payload and "results" in payload)
        )
    ]


def classify_tool_calls(
    output: str, *, known_tool_names: frozenset[str] = frozenset({"search"})
) -> dict[str, int]:
    """Classify raw calls and successful responses without changing rollout.

    A literal opening is an attempt; a valid payload is a valid call; and a
    successful ``<tool_response>`` is an execution. ``malformed`` covers
    parser-level failures, while unknown names remain separately visible.
    """
    if not isinstance(output, str):
        return dict(_EMPTY_CLASSIFICATION)

    attempted = len(_TOOL_CALL_OPENING.findall(output))
    valid = 0
    valid_search = 0
    malformed = 0
    unknown = 0
    for block in _TOOL_CALL_BLOCK.findall(output):
        try:
            decoded = json.loads(block)
            if not isinstance(decoded, dict) or "name" not in decoded or "arguments" not in decoded:
                raise ValueError("tool call must contain name and arguments")
            name = decoded["name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool call name must be a non-empty string")
        except (TypeError, ValueError, json.JSONDecodeError):
            malformed += 1
            continue
        valid += 1
        valid_search += int(name == "search")
        if name not in known_tool_names:
            unknown += 1

    executed = 0
    executed_search = 0
    for payload in tool_response_payloads(output):
        if payload.get("ok") is not True:
            continue
        executed += 1
        executed_search += int(
            payload.get("tool") == "search"
            or ("tool" not in payload and "query" in payload and "results" in payload)
        )

    return {
        "attempted_tool_call_count": attempted,
        # Compatibility alias: this now explicitly means literal attempts.
        "tool_call_count": attempted,
        "valid_tool_call_count": valid,
        "valid_search_call_count": valid_search,
        "executed_tool_call_count": executed,
        "executed_search_call_count": executed_search,
        "malformed_tool_call_count": malformed,
        "unknown_tool_call_count": unknown,
    }


__all__ = [
    "classify_tool_calls",
    "executed_search_payloads",
    "tool_response_payloads",
]
