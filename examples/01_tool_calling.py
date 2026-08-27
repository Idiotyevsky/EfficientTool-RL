"""Make one tool-call protocol step visible without loading a model."""

from __future__ import annotations

import json

from efficienttool_rl.data import Passage
from efficienttool_rl.protocol import ToolCall, parse_action
from efficienttool_rl.tools import BM25Search

# This is an explicitly simulated model emission for teaching the protocol.
SIMULATED_MODEL_OUTPUT = (
    '<tool_call>{"name":"search","arguments":'
    '{"query":"Ada Lovelace","top_k":1}}</tool_call>'
)


def main() -> None:
    print("Model output (simulated):")
    print(SIMULATED_MODEL_OUTPUT)

    action = parse_action(SIMULATED_MODEL_OUTPUT)
    print("\nParsed action:")
    print(json.dumps(action.to_dict(), indent=2, ensure_ascii=False))

    if not isinstance(action, ToolCall):
        raise RuntimeError(f"expected a tool call, got {type(action).__name__}")

    search = BM25Search(
        [
            Passage(
                title="Ada Lovelace",
                text="Ada Lovelace wrote notes on Charles Babbage's Analytical Engine.",
            ),
            Passage(
                title="Alan Turing",
                text="Alan Turing studied computation and theoretical computer science.",
            ),
        ],
        max_observation_tokens=32,
    )
    observation = search.tool(action.arguments)

    print("\nSearch observation:")
    print(json.dumps(observation, indent=2, ensure_ascii=False))
    print("\nThis example uses the real parser and BM25 tool.")
    print("The model emission above is a labeled teaching input, not a prediction.")


if __name__ == "__main__":
    main()
