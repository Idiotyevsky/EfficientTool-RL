#!/usr/bin/env python3
"""Check sampled rollout diversity and task-only reward variance before GRPO."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from efficienttool_rl.agent import AgentConfig, AgentRunner, JsonlTrajectoryWriter
from efficienttool_rl.data import load_hotpotqa
from efficienttool_rl.policies import TransformersToolPolicy
from efficienttool_rl.rewards import task_reward
from efficienttool_rl.tools import BM25Search
from efficienttool_rl.training.verl_data import SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-search-calls", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.prompts < 1 or args.group_size < 2:
        raise ValueError("prompts must be positive and group-size must be at least 2")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_hotpotqa(args.data, split="train")
    selected = examples[args.start_index : args.start_index + args.prompts]
    if len(selected) != args.prompts:
        raise ValueError("requested prompt range exceeds input dataset")
    policy = TransformersToolPolicy(
        args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    writer = JsonlTrajectoryWriter(args.output_dir / "trajectories.jsonl")
    groups: list[dict[str, object]] = []
    all_episodes = 0
    for prompt_offset, example in enumerate(selected):
        rewards: list[float] = []
        signatures: list[str] = []
        for member in range(args.group_size):
            policy.set_seed(args.seed + prompt_offset * args.group_size + member)
            search = BM25Search(example.passages, max_observation_tokens=512)

            def search_tool(arguments: dict[str, object]) -> list[dict[str, object]]:
                return search.tool(arguments)

            runner = AgentRunner(
                policy,
                tools={"search": search_tool},
                config=AgentConfig(
                    max_turns=args.max_turns,
                    max_tool_calls=args.max_search_calls,
                ),
                system_prompt=SYSTEM_PROMPT,
            )
            episode = runner.run(example.question, episode_id=f"{example.example_id}-g{member}")
            writer.append(episode)
            response = "\n".join(step.model_output for step in episode.steps)
            reward = task_reward(response, example.answer)
            rewards.append(reward["score"])
            signatures.append(json.dumps(episode.to_dict(), sort_keys=True))
            all_episodes += 1
        groups.append(
            {
                "example_id": example.example_id,
                "rewards": rewards,
                "reward_mean": statistics.mean(rewards),
                "reward_variance": statistics.pvariance(rewards),
                "distinct_trajectory_count": len(set(signatures)),
                "distinct_reward_count": len(set(rewards)),
            }
        )

    variances = [float(group["reward_variance"]) for group in groups]
    reward_values = [float(reward) for group in groups for reward in group["rewards"]]
    report = {
        "episodes": all_episodes,
        "groups": len(groups),
        "group_size": args.group_size,
        "mean_reward": statistics.mean(reward_values),
        "reward_std": statistics.pstdev(reward_values),
        "mean_group_reward_variance": statistics.mean(variances),
        "zero_variance_group_ratio": sum(value == 0 for value in variances) / len(variances),
        "nontrivial_reward_group_ratio": sum(
            int(group["distinct_reward_count"]) > 1 for group in groups
        )
        / len(groups),
        "all_groups_have_trajectory_diversity": all(
            int(group["distinct_trajectory_count"]) > 1 for group in groups
        ),
        "reward_histogram": dict(sorted(Counter(reward_values).items())),
        "groups_detail": groups,
    }
    (args.output_dir / "rollout_diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
