"""Compare useful and wasteful searches with the production analyzer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from efficienttool_rl.agent import AgentConfig, AgentRunner
from efficienttool_rl.data import HotpotExample, Passage
from efficienttool_rl.evaluation.trajectory_analysis import analyze_trajectories
from efficienttool_rl.tools import BM25Search


class FixedPolicy:
    def __init__(self, responses: tuple[str, ...]) -> None:
        self._responses = iter(responses)

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        del messages
        return next(self._responses)


PASSAGES = (
    Passage(
        title="Ada Lovelace",
        text="Ada Lovelace wrote notes on Charles Babbage's Analytical Engine.",
    ),
    Passage(
        title="Analytical Engine",
        text="The Analytical Engine was designed by Charles Babbage.",
    ),
)


def make_example(example_id: str) -> HotpotExample:
    return HotpotExample(
        example_id=example_id,
        question="What machine did Ada Lovelace write notes about?",
        answer="Analytical Engine",
        passages=PASSAGES,
        supporting_titles=("Ada Lovelace", "Analytical Engine"),
        split="demo",
    )


def run_episode(example_id: str, queries: tuple[str, ...]) -> dict[str, object]:
    search = BM25Search(PASSAGES, max_observation_tokens=48)
    responses = tuple(
        f'<tool_call>{{"name":"search","arguments":{{"query":"{query}","top_k":1}}}}</tool_call>'
        for query in queries
    ) + ("<answer>Analytical Engine</answer>",)
    episode = AgentRunner(
        policy=FixedPolicy(responses),
        tools={"search": search.tool},
        config=AgentConfig(max_turns=4, max_tool_calls=3),
    ).run(
        "What machine did Ada Lovelace write notes about?",
        episode_id=example_id,
    )
    return episode.to_dict()


def main() -> None:
    useful_path = run_episode(
        "useful-path",
        ("Ada Lovelace", "Analytical Engine"),
    )
    wasteful_path = run_episode(
        "wasteful-path",
        ("Ada Lovelace", "Ada Lovelace"),
    )
    report, categorized = analyze_trajectories(
        [useful_path, wasteful_path],
        [make_example("useful-path"), make_example("wasteful-path")],
    )

    selected = {
        key: report[key]
        for key in (
            "attempted_tool_call_count",
            "valid_tool_call_count",
            "executed_search_call_count",
            "useful_search_call_count",
            "wasted_search_call_count",
            "tool_efficiency",
            "duplicate_query_rate",
        )
    }
    print("Production analyzer summary:")
    print(json.dumps(selected, indent=2))
    print("\nPer-episode classification:")
    print(
        json.dumps(
            [
                {
                    "episode_id": item["episode_id"],
                    "useful_search_calls": item["useful_search_calls"],
                    "wasted_search_calls": item["wasted_search_calls"],
                    "queries": item["queries"],
                }
                for item in categorized
            ],
            indent=2,
            ensure_ascii=False,
        )
    )
    print("\nSupporting titles are used offline by the analyzer, never as observations.")


if __name__ == "__main__":
    main()
