"""Project-local canonical adapter for verl's async tool-agent loop.

The adapter keeps verl responsible for rollout, tool execution, and response
masks, while applying the same one-action parser used by the local evaluator.
It is intentionally a thin subclass: upstream verl remains an external,
editable dependency and is not modified by this project.
"""

from __future__ import annotations

from contextvars import ContextVar
import json
from collections.abc import Mapping
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall

from ..protocol import FinalAnswer, InvalidAction, ToolCall, parse_action


_ACTIVE_AGENT_DATA: ContextVar[AgentData | None] = ContextVar(
    "efficienttool_active_agent_data", default=None
)


def resolve_max_executed_search_calls(
    tools_kwargs: Mapping[str, Any] | None,
    tools: Mapping[str, Any],
    *,
    default: int = 3,
) -> int:
    """Resolve the per-trajectory search budget used by the native loop."""
    record_kwargs = (tools_kwargs or {}).get("search", {})
    create_kwargs = (
        record_kwargs.get("create_kwargs", {})
        if isinstance(record_kwargs, Mapping)
        else {}
    )
    if isinstance(create_kwargs, Mapping) and "max_executed_search_calls" in create_kwargs:
        value = create_kwargs["max_executed_search_calls"]
    else:
        search_tool = tools.get("search")
        tool_config = getattr(search_tool, "config", {})
        value = tool_config.get("max_executed_search_calls", default)
    value = int(value)
    if value < 0:
        raise ValueError("max_executed_search_calls must be non-negative")
    return value


def _find_subsequence(sequence: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(sequence):
        return None
    width = len(needle)
    for start in range(len(sequence) - width + 1):
        if sequence[start : start + width] == needle:
            return start
    return None


class CanonicalToolAgentLoop(ToolAgentLoop):
    """Use the project action protocol at every native generation boundary."""

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Expose the active trajectory to the tool-execution accounting hook."""
        token = _ACTIVE_AGENT_DATA.set(agent_data)
        try:
            return await super()._handle_processing_tools_state(agent_data)
        finally:
            _ACTIVE_AGENT_DATA.reset(token)

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
    ) -> tuple[Any, float, dict[str, Any]]:
        response, reward, metrics = await super()._call_tool(tool_call, tools_kwargs)
        agent_data = _ACTIVE_AGENT_DATA.get()
        if agent_data is not None and tool_call.name == "search":
            if int(metrics.get("executed_search_calls", 0)) > 0:
                key = "efficienttool_executed_search_calls"
                agent_data.metrics[key] = int(agent_data.metrics.get(key, 0)) + 1
        return response, reward, metrics

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination=ignore_termination
        )
        text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True),
        )
        action = parse_action(text)

        if state is AgentState.PROCESSING_TOOLS:
            # verl's Hermes parser accepts multiple blocks and silently drops
            # malformed ones.  The project protocol accepts exactly one known
            # tool call, so invalid or mixed turns terminate without executing
            # a different action than the local evaluator would execute.
            if not isinstance(action, ToolCall) or action.name not in self.tools:
                agent_data.tool_calls = []
                return AgentState.TERMINATED
            self._truncate_current_generation(agent_data)
            agent_data.tool_calls = [
                FunctionCall(
                    name=action.name,
                    arguments=json.dumps(action.arguments, ensure_ascii=False),
                )
            ]
            if action.name == "search":
                executed_search_calls = int(
                    agent_data.metrics.get("efficienttool_executed_search_calls", 0)
                )
                max_executed_search_calls = resolve_max_executed_search_calls(
                    agent_data.tools_kwargs, self.tools
                )
                if executed_search_calls >= max_executed_search_calls:
                    agent_data.tool_calls = []
                    agent_data.metrics["efficienttool_tool_budget_exhausted"] = 1
                    return AgentState.TERMINATED
            return AgentState.PROCESSING_TOOLS

        if state is AgentState.TERMINATED and isinstance(action, FinalAnswer):
            # A model can emit an answer followed by repetition before vLLM
            # returns.  Keep the same first-action boundary as AgentRunner.
            self._truncate_current_generation(agent_data)

        return state

    def _truncate_current_generation(self, agent_data: AgentData) -> None:
        """Drop tokens after the first complete action without changing masks."""
        response_ids = list(agent_data.response_ids)
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        action = parse_action(text)
        if isinstance(action, InvalidAction):
            return

        action_type = "answer" if isinstance(action, FinalAnswer) else "tool_call"
        closing_ids = list(
            self.tokenizer.encode(f"</{action_type}>", add_special_tokens=False)
        )
        marker_start = _find_subsequence(response_ids, closing_ids)
        if marker_start is None:
            return
        keep = marker_start + len(closing_ids)
        if keep >= len(response_ids):
            return

        removed = len(response_ids) - keep
        agent_data.response_ids = response_ids[:keep]
        agent_data.prompt_ids = agent_data.prompt_ids[:-removed]
        agent_data.response_mask = agent_data.response_mask[:-removed]
        if agent_data.response_logprobs:
            agent_data.response_logprobs = agent_data.response_logprobs[:-removed]
