"""Minimal multi-turn tool-agent loop with structured trajectories."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .protocol import (
    FinalAnswer,
    InvalidAction,
    ToolCall,
    canonicalize_action_text,
    parse_action,
)

Message = dict[str, str]
ToolHandler = Callable[[dict[str, Any]], Any]


class Policy(Protocol):
    def generate(self, messages: Sequence[Mapping[str, str]]) -> str: ...


@dataclass(frozen=True)
class AgentConfig:
    max_turns: int = 5
    max_tool_calls: int = 3

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")


@dataclass
class EpisodeStep:
    turn: int
    model_output: str
    action: dict[str, Any]
    observation: dict[str, Any] | None
    terminated: bool


@dataclass
class EpisodeResult:
    episode_id: str
    prompt: str
    steps: list[EpisodeStep]
    final_answer: str | None
    termination_reason: str
    tool_calls: int
    invalid_actions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlTrajectoryWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, episode: EpisodeResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(episode.to_dict(), ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")


class AgentRunner:
    def __init__(
        self,
        policy: Policy,
        tools: Mapping[str, ToolHandler],
        config: AgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.policy = policy
        self.tools = dict(tools)
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt or (
            "Return exactly one <tool_call> JSON block or one <answer> block per turn."
        )

    def run(self, prompt: str, episode_id: str) -> EpisodeResult:
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        steps: list[EpisodeStep] = []
        tool_calls = 0
        invalid_actions = 0

        for turn in range(self.config.max_turns):
            model_output = canonicalize_action_text(self.policy.generate(tuple(messages)))
            action = parse_action(model_output)
            observation: dict[str, Any] | None = None
            termination_reason: str | None = None
            final_answer: str | None = None

            if isinstance(action, FinalAnswer):
                final_answer = action.answer
                termination_reason = "final_answer"
            elif isinstance(action, InvalidAction):
                invalid_actions += 1
                observation = self._error_observation(action.code, action.message)
            else:
                observation, executed, terminal = self._execute_tool(action, tool_calls)
                if executed:
                    tool_calls += 1
                if not observation["ok"]:
                    invalid_actions += int(observation["error"]["code"] == "unknown_tool")
                if terminal:
                    termination_reason = "max_tool_calls"

            steps.append(
                EpisodeStep(
                    turn=turn,
                    model_output=model_output,
                    action=action.to_dict(),
                    observation=observation,
                    terminated=termination_reason is not None,
                )
            )
            if termination_reason is not None:
                return EpisodeResult(
                    episode_id=episode_id,
                    prompt=prompt,
                    steps=steps,
                    final_answer=final_answer,
                    termination_reason=termination_reason,
                    tool_calls=tool_calls,
                    invalid_actions=invalid_actions,
                )

            messages.append({"role": "assistant", "content": model_output})
            assert observation is not None
            messages.append(
                {
                    "role": "tool" if isinstance(action, ToolCall) else "user",
                    "content": json.dumps(observation, ensure_ascii=False, sort_keys=True),
                }
            )

        if steps:
            steps[-1].terminated = True
        return EpisodeResult(
            episode_id=episode_id,
            prompt=prompt,
            steps=steps,
            final_answer=None,
            termination_reason="max_turns",
            tool_calls=tool_calls,
            invalid_actions=invalid_actions,
        )

    def _execute_tool(
        self,
        action: ToolCall,
        executed_tool_calls: int,
    ) -> tuple[dict[str, Any], bool, bool]:
        handler = self.tools.get(action.name)
        if handler is None:
            return self._error_observation("unknown_tool", f"Unknown tool: {action.name}"), False, False
        if executed_tool_calls >= self.config.max_tool_calls:
            return (
                self._error_observation("tool_budget_exhausted", "Tool-call budget is exhausted."),
                False,
                True,
            )
        try:
            result = handler(action.arguments)
            json.dumps(result)
        except Exception as exc:  # Tool failures become observations, not episode crashes.
            return self._error_observation("tool_error", str(exc)), True, False
        return {"ok": True, "tool": action.name, "result": result}, True, False

    @staticmethod
    def _error_observation(code: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {"code": code, "message": message},
            "expected_action_format": (
                "Return exactly one <tool_call>{JSON}</tool_call> or "
                "<answer>final text</answer> block."
            ),
        }
