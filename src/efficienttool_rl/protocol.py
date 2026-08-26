"""Parsing for the strict M1 model action protocol."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "tool_call", "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class FinalAnswer:
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "answer", "answer": self.answer}


@dataclass(frozen=True)
class InvalidAction:
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "invalid", "code": self.code, "message": self.message}


ParsedAction: TypeAlias = ToolCall | FinalAnswer | InvalidAction

_OPENING_TAG = re.compile(r"<(?:tool_call|answer)>")
_ACTION_BLOCK = re.compile(
    r"<(tool_call|answer)>(.*?)</\1>",
    flags=re.DOTALL,
)


def invalid(code: str, message: str) -> InvalidAction:
    return InvalidAction(code=code, message=message)


def parse_action(text: str) -> ParsedAction:
    """Parse exactly one tagged action without raising on model-generated text."""
    if not isinstance(text, str):
        return invalid("invalid_output_type", "Model output must be a string.")

    openings = _OPENING_TAG.findall(text)
    matches = list(_ACTION_BLOCK.finditer(text))
    if not openings:
        return invalid("no_action", "No action block was found.")
    if len(openings) > 1 or len(matches) > 1:
        return invalid("multiple_actions", "Exactly one action block is allowed per turn.")
    if len(matches) != 1:
        return invalid("malformed_tag", "The action block is not closed correctly.")

    action_type, payload = matches[0].group(1), matches[0].group(2).strip()
    if action_type == "answer":
        if not payload:
            return invalid("empty_answer", "Final answer text must not be empty.")
        return FinalAnswer(answer=payload)

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        return invalid("malformed_json", f"Tool call JSON is invalid: {exc.msg}.")
    if not isinstance(decoded, dict):
        return invalid("invalid_tool_payload", "Tool call payload must be a JSON object.")

    name = decoded.get("name")
    if not isinstance(name, str) or not name.strip():
        return invalid("missing_tool_name", "Tool call requires a non-empty string name.")
    if "arguments" not in decoded:
        return invalid("missing_arguments", "Tool call requires an arguments object.")
    arguments = decoded["arguments"]
    if not isinstance(arguments, dict):
        return invalid("invalid_arguments", "Tool arguments must be a JSON object.")
    return ToolCall(name=name.strip(), arguments=arguments)
