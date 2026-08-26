"""Per-trajectory deterministic HotpotQA search tool for verl."""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from ..data import Passage
from ..tools import BM25Search


class HotpotSearchTool(BaseTool):
    """Build an isolated BM25 index for each verl trajectory."""

    def __init__(self, config: dict[str, Any], tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, BM25Search] = {}

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
        self._instances[instance_id] = BM25Search(
            passages,
            max_observation_tokens=int(
                create_kwargs.get(
                    "max_observation_tokens", self.config.get("max_observation_tokens", 512)
                )
            ),
        )
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        del kwargs
        search = self._instances[instance_id]
        query = parameters.get("query")
        top_k = parameters.get("top_k", self.config.get("top_k", 3))
        try:
            if not isinstance(query, str):
                raise ValueError("search.query must be a string")
            if not isinstance(top_k, int) or isinstance(top_k, bool):
                raise ValueError("search.top_k must be an integer")
            top_k = min(top_k, int(self.config.get("max_top_k", 3)))
            results = [result.to_dict() for result in search.search(query, top_k=top_k)]
        except (TypeError, ValueError) as exc:
            payload = {"ok": False, "error": {"code": "invalid_search", "message": str(exc)}}
            return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0.0, {"invalid": 1}
        payload = {"ok": True, "query": query, "results": results}
        return (
            ToolResponse(text=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            0.0,
            {"search_calls": 1, "result_count": len(results)},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del kwargs
        self._instances.pop(instance_id, None)
