from __future__ import annotations

from types import SimpleNamespace

from scripts.fixed_policy_rollout_audit import _local_search_usage


def _step(*, executed: bool, titles: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        tool_executed=executed,
        action={"kind": "tool_call", "name": "search", "arguments": {"query": "q"}},
        observation={"ok": True, "result": [{"title": title} for title in titles]},
    )


def test_local_usage_counts_new_support_and_waste() -> None:
    episode = SimpleNamespace(
        steps=[
            _step(executed=True, titles=["A"]),
            _step(executed=True, titles=["B"]),
            _step(executed=True, titles=["A"]),
        ]
    )

    assert _local_search_usage(episode, ["A", "B"]) == {
        "executed_search_calls": 3,
        "useful_search_calls": 2,
        "wasted_search_calls": 1,
    }


def test_local_usage_ignores_unexecuted_or_non_search_steps() -> None:
    episode = SimpleNamespace(
        steps=[
            _step(executed=False, titles=["A"]),
            SimpleNamespace(
                tool_executed=True,
                action={"kind": "tool_call", "name": "calculator", "arguments": {}},
                observation={"ok": True, "result": []},
            ),
        ]
    )

    assert _local_search_usage(episode, ["A"]) == {
        "executed_search_calls": 0,
        "useful_search_calls": 0,
        "wasted_search_calls": 0,
    }
