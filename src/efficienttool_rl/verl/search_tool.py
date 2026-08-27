"""Per-trajectory deterministic HotpotQA search tool for verl."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from ..data import Passage
from ..tools import BM25Search


@dataclass
class _SearchInstance:
    index: BM25Search
    default_top_k: int
    max_top_k: int
    max_executed_search_calls: int
    executed_search_calls: int = 0


class HotpotSearchTool(BaseTool):
    """Build an isolated BM25 index for each verl trajectory."""

    def __init__(self, config: dict[str, Any], tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, _SearchInstance] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        create_kwargs = kwargs.get("create_kwargs", {})
        raw_passages = create_kwargs.get("passages", [])
        if not isinstance(raw_passages, list) or not raw_passages:
            raise ValueError("search.create_kwargs.passages must be a non-empty list")
        passages = []
        for item in raw_passages:
            if not isinstance(item, dict) or not isinstance(item.get("title"), str):
                raise ValueError("each passage requires a string title and text")
            if not isinstance(item.get("text"), str):
                raise ValueError("each passage requires a string title and text")
            passages.append(Passage(title=item["title"], text=item["text"]))
        instance_id = instance_id or str(uuid4())
        max_observation_tokens = int(
            create_kwargs.get(
                "max_observation_tokens", self.config.get("max_observation_tokens", 512)
            )
        )
        default_top_k = int(create_kwargs.get("top_k", self.config.get("top_k", 3)))
        max_top_k = int(
            create_kwargs.get("max_top_k", self.config.get("max_top_k", default_top_k))
        )
        max_executed_search_calls = int(
            create_kwargs.get(
                "max_executed_search_calls",
                self.config.get("max_executed_search_calls", 3),
            )
        )
        if max_observation_tokens < 1 or default_top_k < 1 or max_top_k < 1:
            raise ValueError("observation and top-k limits must be positive")
        if max_executed_search_calls < 0:
            raise ValueError("max_executed_search_calls must be non-negative")
        if default_top_k > max_top_k:
            default_top_k = max_top_k
        self._instances[instance_id] = _SearchInstance(
            index=BM25Search(passages, max_observation_tokens=max_observation_tokens),
            default_top_k=default_top_k,
            max_top_k=max_top_k,
            max_executed_search_calls=max_executed_search_calls,
        )
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        del kwargs
        instance = self._instances[instance_id]
        query = parameters.get("query")
        top_k = parameters.get("top_k", instance.default_top_k)
        if instance.executed_search_calls >= instance.max_executed_search_calls:
            payload = {
                "ok": False,
                "tool": "search",
                "error": {
                    "code": "tool_budget_exhausted",
                    "message": "Search-call budget is exhausted.",
                },
            }
            return (
                ToolResponse(text=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                0.0,
                {
                    "search_calls": 0,
                    "executed_search_calls": 0,
                    "tool_budget_exhausted": 1,
                },
            )
        try:
            if not isinstance(query, str):
                raise ValueError("search.query must be a string")
            if not isinstance(top_k, int) or isinstance(top_k, bool):
                raise ValueError("search.top_k must be an integer")
            top_k = min(top_k, instance.max_top_k)
            results = [result.to_dict() for result in instance.index.search(query, top_k=top_k)]
        except (TypeError, ValueError) as exc:
            payload = {
                "ok": False,
                "tool": "search",
                "error": {"code": "invalid_search", "message": str(exc)},
            }
            return (
                ToolResponse(text=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                0.0,
                {"search_calls": 0, "executed_search_calls": 0, "invalid": 1},
            )
        instance.executed_search_calls += 1
        payload = {"ok": True, "tool": "search", "query": query, "results": results}
        return (
            ToolResponse(text=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            0.0,
            {"search_calls": 1, "executed_search_calls": 1, "result_count": len(results)},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del kwargs
        self._instances.pop(instance_id, None)
