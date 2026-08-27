"""Run a transparent two-search episode through the real AgentRunner."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from efficienttool_rl.agent import AgentConfig, AgentRunner
from efficienttool_rl.data import Passage
from efficienttool_rl.protocol import SYSTEM_PROMPT
from efficienttool_rl.tools import BM25Search


class ScriptedPolicy:
    """A tiny deterministic policy adapter for demonstrating loop mechanics.

    It deliberately ignores the messages and emits a recorded teaching path.
    This is not a language-model result and is labeled as such in the output.
    """

    def __init__(self) -> None:
        self._responses = iter(
            (
                '<tool_call>{"name":"search","arguments":'
                '{"query":"Ada Lovelace Babbage","top_k":1}}</tool_call>',
                '<tool_call>{"name":"search","arguments":'
                '{"query":"Analytical Engine","top_k":1}}</tool_call>',
                "<answer>Analytical Engine</answer>",
            )
        )

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        del messages
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise RuntimeError("scripted episode requested too many turns") from exc


def main() -> None:
    passages = [
        Passage(
            title="Ada Lovelace",
            text="Ada Lovelace wrote notes on Charles Babbage's Analytical Engine.",
        ),
        Passage(
            title="Analytical Engine",
            text="The Analytical Engine was a general-purpose mechanical computer designed by Charles Babbage.",
        ),
        Passage(
            title="Unrelated passage",
            text="A short passage about a different historical topic.",
        ),
    ]
    search = BM25Search(passages, max_observation_tokens=48)
    runner = AgentRunner(
        policy=ScriptedPolicy(),
        tools={"search": search.tool},
        config=AgentConfig(max_turns=4, max_tool_calls=3),
        system_prompt=SYSTEM_PROMPT,
    )

    print("Recorded trajectory demonstration (not model output):")
    episode = runner.run(
        "What machine did the person who wrote notes on Babbage's work write about?",
        episode_id="example-multiturn",
    )

    for step in episode.steps:
        print(f"\nTurn {step.turn + 1}: {step.model_output}")
        if step.observation is not None:
            print("Observation:")
            print(json.dumps(step.observation, indent=2, ensure_ascii=False))

    print("\nEpisode summary:")
    print(
        json.dumps(
            {
                "final_answer": episode.final_answer,
                "termination_reason": episode.termination_reason,
                "attempted_tool_calls": episode.attempted_tool_calls,
                "valid_tool_calls": episode.valid_tool_calls,
                "executed_search_calls": episode.executed_search_calls,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
