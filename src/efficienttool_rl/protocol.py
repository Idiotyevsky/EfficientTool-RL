"""Parsing for the strict M1 model action protocol."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeAlias


SYSTEM_PROMPT = """You are a multi-hop question-answering tool agent.
Use search to gather evidence for every entity needed by the question. Prefer
concise, entity-specific queries and avoid repeating a query. Search again when
the current evidence leaves any required entity unresolved. Emit exactly one action per turn: either
<tool_call>{"name":"search","arguments":{"query":"...","top_k":3}}</tool_call>
or <answer>minimal answer span</answer>. The answer block must contain only the
answer, never an explanation or full sentence. For yes/no questions, output
exactly <answer>yes</answer> or <answer>no</answer>. Do not add text outside the
action block."""


SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search a deterministic local passage collection for evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise evidence-focused search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results, from 1 to 3.",
                },
            },
            "required": ["query"],
        },
    },
}


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


def canonicalize_action_text(text: str) -> str:
    """Keep only the first complete action block when it is unambiguous.

    Both local generation and native verl generation may continue emitting
    prose after a tagged action.  The agent protocol treats the first complete
    action as the turn boundary, while preserving invalid output for reward
    and failure analysis instead of silently repairing it.
    """
    action = parse_action(text)
    if isinstance(action, InvalidAction):
        return text
    match = _ACTION_BLOCK.search(text)
    if match is None:
        return text
    return text[: match.end()].strip()
