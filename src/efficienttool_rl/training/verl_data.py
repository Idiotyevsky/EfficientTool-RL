"""Convert normalized HotpotQA examples to verl RL records."""

from __future__ import annotations

from typing import Any

from ..data import HotpotExample

SYSTEM_PROMPT = """You are a multi-hop question-answering tool agent.
Use search to gather evidence for every entity needed by the question. Prefer
concise, entity-specific queries and search again when evidence is incomplete.
Emit exactly one action per turn:
<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"...\"}}</tool_call>
or <answer>minimal answer span</answer>. The answer block must contain only
the answer, never an explanation or full sentence. For yes/no questions, output
exactly yes or no inside the answer block."""


def to_verl_record(
    example: HotpotExample,
    *,
    index: int,
    max_observation_tokens: int = 512,
    max_top_k: int = 3,
) -> dict[str, Any]:
    """Create a JSON-serializable record with no answer in tool kwargs."""
    if max_observation_tokens < 1 or max_top_k < 1:
        raise ValueError("observation and top-k limits must be positive")
    passages = [{"title": passage.title, "text": passage.text} for passage in example.passages]
    return {
        "data_source": "hotpotqa_distractor",
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example.question},
        ],
        "ability": "multi_hop_qa",
        "reward_model": {"style": "rule", "ground_truth": example.answer},
        "extra_info": {
            "split": example.split,
            "index": index,
            "example_id": example.example_id,
            "question": example.question,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "passages": passages,
                        "max_observation_tokens": max_observation_tokens,
                        "max_top_k": max_top_k,
                    }
                }
            },
        },
    }
