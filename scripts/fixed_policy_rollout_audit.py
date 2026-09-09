#!/usr/bin/env python3
"""Sample fixed-policy tool-agent trajectories for cost-signal validation.

This runner deliberately performs inference only.  It reuses the local
AgentRunner, strict one-result search environment, and Transformers policy so
that a fixed checkpoint can be audited without starting a trainer or writing
checkpoints.  Rows are emitted in a format accepted by
``scripts/audit_cost_signal.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from efficienttool_rl.agent import AgentConfig, AgentRunner
from efficienttool_rl.data import HotpotExample, Passage, load_hotpotqa
from efficienttool_rl.evaluation.metrics import answer_metrics
from efficienttool_rl.policies import TransformersToolPolicy
from efficienttool_rl.protocol import SYSTEM_PROMPT
from efficienttool_rl.tools import BM25Search


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_examples(path: Path, split: str) -> list[HotpotExample]:
    if path.suffix == ".parquet":
        return _parquet_examples(path, split)
    return load_hotpotqa(path, split=split)


def _local_search_usage(episode: Any, supporting_titles: Sequence[str]) -> dict[str, int]:
    support = set(supporting_titles)
    discovered: set[str] = set()
    executed = 0
    useful = 0
    for step in episode.steps:
        if not step.tool_executed:
            continue
        action = step.action
        if action.get("kind") != "tool_call" or action.get("name") != "search":
            continue
        executed += 1
        observation = step.observation or {}
        raw_results = observation.get("result", [])
        titles = {
            item.get("title")
            for item in raw_results
            if isinstance(item, Mapping) and isinstance(item.get("title"), str)
        } if isinstance(raw_results, list) else set()
        new_titles = (titles & support) - discovered
        discovered.update(new_titles)
        useful += int(bool(new_titles))
    return {
        "executed_search_calls": executed,
        "useful_search_calls": useful,
        "wasted_search_calls": max(executed - useful, 0),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-search-calls", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-top-k", type=int, default=1)
    parser.add_argument("--max-observation-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.start_index < 0 or args.limit < 1:
        raise ValueError("start-index must be non-negative and limit must be positive")
    if args.rollouts_per_prompt < 1:
        raise ValueError("rollouts-per-prompt must be positive")
    if args.shard_id < 0 or args.num_shards < 1 or args.shard_id >= args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    if args.max_turns < 1 or args.max_search_calls < 0:
        raise ValueError("invalid turn or search budget")
    if args.top_k < 1 or args.max_top_k < args.top_k:
        raise ValueError("max-top-k must be at least top-k and both must be positive")
    if args.max_observation_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("token limits must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be positive and top-p must be in (0, 1]")
    if args.output.exists() and args.output.is_dir():
        raise IsADirectoryError(args.output)

    examples = load_examples(args.data, args.split)
    selected = examples[args.start_index : args.start_index + args.limit]
    if len(selected) != args.limit:
        raise ValueError("requested range exceeds the dataset")
    shard = selected[args.shard_id :: args.num_shards]

    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing[str(row["trajectory_id"])] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = TransformersToolPolicy(
        args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed + args.shard_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    completed = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for example_offset, example in enumerate(shard):
            search = BM25Search(
                example.passages,
                max_observation_tokens=args.max_observation_tokens,
            )

            def bounded_search(arguments: dict[str, object]) -> list[dict[str, object]]:
                supplied = dict(arguments)
                top_k = supplied.get("top_k", args.top_k)
                if isinstance(top_k, int) and not isinstance(top_k, bool):
                    supplied["top_k"] = min(top_k, args.max_top_k)
                return search.tool(supplied)

            runner = AgentRunner(
                policy,
                tools={"search": bounded_search},
                config=AgentConfig(
                    max_turns=args.max_turns,
                    max_tool_calls=args.max_search_calls,
                ),
                system_prompt=SYSTEM_PROMPT,
            )
            for rollout_index in range(args.rollouts_per_prompt):
                trajectory_id = f"{example.example_id}__rollout_{rollout_index}"
                if trajectory_id in existing:
                    continue
                episode = runner.run(
                    example.question,
                    episode_id=trajectory_id,
                )
                scores = answer_metrics(episode.final_answer or "", example.answer)
                usage = _local_search_usage(episode, example.supporting_titles)
                record: dict[str, Any] = {
                    "trajectory_id": trajectory_id,
                    "input": example.question,
                    "episode_id": episode.episode_id,
                    "reference_answer": example.answer,
                    "task_reward": 0.5 * scores["exact_match"] + 0.5 * scores["f1"],
                    "exact_match": scores["exact_match"],
                    "f1": scores["f1"],
                    "valid_answer": float(bool(episode.final_answer)),
                    "wasted_search_calls": usage["wasted_search_calls"],
                    "useful_search_calls": usage["useful_search_calls"],
                    "executed_search_calls": usage["executed_search_calls"],
                    "attempted_tool_calls": episode.attempted_tool_calls,
                    "valid_tool_calls": episode.valid_tool_calls,
                    "invalid_actions": episode.invalid_actions,
                    "generated_tokens": sum(
                        policy.count_tokens(step.model_output) for step in episode.steps
                    ),
                    "turns": len(episode.steps),
                    "termination_reason": episode.termination_reason,
                    "trajectory": episode.to_dict(),
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                existing[trajectory_id] = record
                completed += 1
                print(
                    json.dumps(
                        {
                            "global_index": args.start_index + args.shard_id + example_offset * args.num_shards,
                            "example_id": example.example_id,
                            "rollout": rollout_index,
                            "completed": completed,
                            "em": scores["exact_match"],
                            "f1": scores["f1"],
                            **usage,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if args.config_output:
        _write_json(
            args.config_output,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data": str(args.data.resolve()),
                "data_sha256": _sha256(args.data),
                "model": str(args.model.resolve()),
                "device": args.device,
                "start_index": args.start_index,
                "limit": args.limit,
                "rollouts_per_prompt": args.rollouts_per_prompt,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "max_turns": args.max_turns,
                "max_search_calls": args.max_search_calls,
                "top_k": args.top_k,
                "max_top_k": args.max_top_k,
                "max_observation_tokens": args.max_observation_tokens,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed + args.shard_id,
                "inference_only": True,
            },
        )


if __name__ == "__main__":
    main()
