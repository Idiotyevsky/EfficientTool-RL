"""Load HotpotQA examples from verl parquet records or normalized JSONL.

The parquet path reads the exact ``reward_model`` / ``extra_info`` columns used
by native verl training, so held-out evaluation always sees the same prompts,
supporting-title metadata, and per-trajectory tool configuration as training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hotpotqa import HotpotExample, Passage, load_hotpotqa


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parquet_examples(path: Path, split: str) -> list[HotpotExample]:
    """Load the same verl parquet records used by native training."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("pyarrow is required to read parquet input") from exc

    table = pq.read_table(str(path), columns=["reward_model", "extra_info"])
    examples: list[HotpotExample] = []
    for row_index, row in enumerate(table.to_pylist()):
        info = _as_dict(row.get("extra_info"))
        reward_model = _as_dict(row.get("reward_model"))
        tools_kwargs = _as_dict(info.get("tools_kwargs"))
        search_kwargs = _as_dict(_as_dict(tools_kwargs.get("search")).get("create_kwargs"))
        raw_passages = search_kwargs.get("passages")
        if not isinstance(raw_passages, list) or not raw_passages:
            raise ValueError(f"row {row_index}: missing search passages")
        passages: list[Passage] = []
        for passage_index, raw in enumerate(raw_passages):
            item = _as_dict(raw)
            if not isinstance(item.get("title"), str) or not isinstance(item.get("text"), str):
                raise ValueError(f"row {row_index}, passage {passage_index}: malformed passage")
            passages.append(Passage(title=item["title"], text=item["text"]))
        question = info.get("question")
        answer = reward_model.get("ground_truth")
        example_id = info.get("example_id")
        support = info.get("supporting_titles", [])
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"row {row_index}: missing question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"row {row_index}: missing ground-truth answer")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"row {row_index}: missing example_id")
        if not isinstance(support, (list, tuple)) or not all(
            isinstance(title, str) for title in support
        ):
            raise ValueError(f"row {row_index}: malformed supporting_titles")
        examples.append(
            HotpotExample(
                example_id=example_id,
                question=question,
                answer=answer,
                passages=tuple(passages),
                supporting_titles=tuple(support),
                split=str(info.get("split", split)),
                question_type=str(info.get("question_type", "unknown")),
                level=str(info.get("level", "unknown")),
            )
        )
    return examples


def load_verl_examples(path: str | Path, split: str = "validation") -> list[HotpotExample]:
    """Read verl parquet records or normalized HotpotQA JSON/JSONL by suffix."""
    path = Path(path)
    if path.suffix == ".parquet":
        return _parquet_examples(path, split)
    return load_hotpotqa(path, split=split)


__all__ = ["load_verl_examples"]
