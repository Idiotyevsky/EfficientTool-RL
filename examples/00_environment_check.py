"""Check Learn Track prerequisites without loading a model."""

from __future__ import annotations

import importlib.util
import platform
import sys

from efficienttool_rl.protocol import parse_action


OPTIONAL_COMPONENTS = (
    "torch",
    "transformers",
    "datasets",
    "hydra",
    "ray",
    "verl",
    "vllm",
)


def main() -> None:
    print("MiniAgentRL Learn Track environment check")
    print(f"Python:     {sys.version.split()[0]}")
    print(f"Platform:   {platform.platform()}")
    print(f"Executable: {sys.executable}")

    parsed = parse_action("<answer>environment check passed</answer>")
    print(f"Core package: PASS ({parsed.to_dict()})")

    print("\nOptional components:")
    for name in OPTIONAL_COMPONENTS:
        state = "available" if importlib.util.find_spec(name) else "not installed"
        print(f"  {name:12} {state}")

    print("\nNext step: run examples/01_tool_calling.py.")
    print("Missing optional components are expected for the CPU-only lessons.")


if __name__ == "__main__":
    main()
