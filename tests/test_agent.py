import json
from collections.abc import Mapping, Sequence

from efficienttool_rl.agent import AgentConfig, AgentRunner, JsonlTrajectoryWriter


class ScriptedPolicy:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        return next(self.outputs)


def tool_call(query: str = "q") -> str:
    return f'<tool_call>{{"name":"search","arguments":{{"query":"{query}"}}}}</tool_call>'


def test_multi_turn_tool_episode_reaches_final_answer() -> None:
    policy = ScriptedPolicy([tool_call("evidence"), "<answer>M1_AGENT_OK</answer>"])
    runner = AgentRunner(
        policy=policy,
        tools={"search": lambda arguments: {"passage": arguments["query"]}},
    )

    result = runner.run("Find evidence.", episode_id="episode-1")

    assert result.termination_reason == "final_answer"
    assert result.final_answer == "M1_AGENT_OK"
    assert result.tool_calls == 1
    assert result.attempted_tool_calls == 1
    assert result.valid_tool_calls == 1
    assert result.executed_tool_calls == 1
    assert result.executed_search_calls == 1
    assert result.invalid_actions == 0
    assert result.steps[0].observation == {
        "ok": True,
        "tool": "search",
        "result": {"passage": "evidence"},
    }
    assert result.steps[-1].terminated is True


def test_malformed_action_does_not_crash_episode() -> None:
    policy = ScriptedPolicy(["<tool_call>{bad}</tool_call>", "<answer>recovered</answer>"])
    result = AgentRunner(policy, tools={}).run("Recover.", episode_id="episode-2")

    assert result.final_answer == "recovered"
    assert result.invalid_actions == 1
    assert result.attempted_tool_calls == 1
    assert result.valid_tool_calls == 0
    assert result.executed_tool_calls == 0
    assert result.steps[0].observation["error"]["code"] == "malformed_json"


def test_unknown_tool_is_logged_and_episode_continues() -> None:
    unknown = '<tool_call>{"name":"calculator","arguments":{}}</tool_call>'
    policy = ScriptedPolicy([unknown, "<answer>fallback</answer>"])
    result = AgentRunner(policy, tools={}).run("Try.", episode_id="episode-3")

    assert result.final_answer == "fallback"
    assert result.invalid_actions == 1
    assert result.attempted_tool_calls == 1
    assert result.valid_tool_calls == 1
    assert result.executed_tool_calls == 0
    assert result.steps[0].observation["error"]["code"] == "unknown_tool"


def test_repeated_calls_terminate_when_budget_is_exceeded() -> None:
    policy = ScriptedPolicy([tool_call(), tool_call(), tool_call()])
    runner = AgentRunner(
        policy,
        tools={"search": lambda arguments: []},
        config=AgentConfig(max_turns=4, max_tool_calls=2),
    )
    result = runner.run("Repeat.", episode_id="episode-4")

    assert result.termination_reason == "max_tool_calls"
    assert result.tool_calls == 2
    assert result.steps[-1].observation["error"]["code"] == "tool_budget_exhausted"


def test_plain_text_terminates_at_max_turns() -> None:
    policy = ScriptedPolicy(["plain", "still plain"])
    runner = AgentRunner(policy, tools={}, config=AgentConfig(max_turns=2))
    result = runner.run("No action.", episode_id="episode-5")

    assert result.termination_reason == "max_turns"
    assert result.invalid_actions == 2
    assert result.steps[-1].terminated is True


def test_tool_error_becomes_observation() -> None:
    def broken_tool(arguments: dict[str, object]) -> object:
        raise RuntimeError("localized failure")

    policy = ScriptedPolicy([tool_call(), "<answer>handled</answer>"])
    result = AgentRunner(policy, tools={"search": broken_tool}).run(
        "Handle failure.", episode_id="episode-6"
    )

    assert result.final_answer == "handled"
    assert result.tool_calls == 1
    assert result.steps[0].observation["error"] == {
        "code": "tool_error",
        "message": "localized failure",
    }


def test_jsonl_writer_records_structured_episode(tmp_path) -> None:
    policy = ScriptedPolicy(["<answer>done</answer>"])
    result = AgentRunner(policy, tools={}).run("Answer.", episode_id="episode-7")
    output = tmp_path / "trajectories.jsonl"

    JsonlTrajectoryWriter(output).append(result)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["episode_id"] == "episode-7"
    assert payload["termination_reason"] == "final_answer"
    assert payload["steps"][0]["action"] == {"kind": "answer", "answer": "done"}
