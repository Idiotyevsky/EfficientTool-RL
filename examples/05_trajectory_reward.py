"""Connect an episode trajectory to the real task reward implementation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from efficienttool_rl.agent import AgentConfig, AgentRunner
from efficienttool_rl.data import Passage
from efficienttool_rl.evaluation import summarize_episodes
from efficienttool_rl.rewards import task_reward
from efficienttool_rl.tools import BM25Search


class OneSearchPolicy:
    """A deterministic policy adapter for inspecting one complete episode."""

    def __init__(self) -> None:
        self._responses = iter(
            (
                '<tool_call>{"name":"search","arguments":'
                '{"query":"Ada Lovelace","top_k":1}}</tool_call>',
                "<answer>Analytical Engine</answer>",
            )
        )

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        del messages
        return next(self._responses)


def main() -> None:
    search = BM25Search(
        [
            Passage(
                title="Ada Lovelace",
                text="Ada Lovelace wrote notes on Charles Babbage's Analytical Engine.",
            ),
            Passage(
                title="Analytical Engine",
                text="The Analytical Engine was designed by Charles Babbage.",
            ),
        ],
        max_observation_tokens=48,
    )
    episode = AgentRunner(
        policy=OneSearchPolicy(),
        tools={"search": search.tool},
        config=AgentConfig(max_turns=3, max_tool_calls=2),
    ).run(
        "What machine did Ada Lovelace write notes about?",
        episode_id="trajectory-reward-demo",
    )

    scored_response = f"<answer>{episode.final_answer or ''}</answer>"
    reward = task_reward(scored_response, "Analytical Engine", alpha=0.5)
    record = {
        **episode.to_dict(),
        "exact_match": reward["em"],
        "f1": reward["f1"],
    }

    print("Trajectory:")
    print(json.dumps(episode.to_dict(), indent=2, ensure_ascii=False))
    print("\nTask reward (0.5 EM + 0.5 token F1):")
    print(json.dumps(reward, indent=2))
    print("\nBehavior summary:")
    print(json.dumps(summarize_episodes([record]), indent=2))
    print("\nThis is a bounded inspection demo, not model training.")


if __name__ == "__main__":
    main()
