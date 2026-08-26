from collections.abc import Mapping, Sequence

from efficienttool_rl.agent import AgentConfig, AgentRunner


class InvalidPolicy:
    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        return "plain text"


def test_invalid_action_observation_repeats_expected_format() -> None:
    result = AgentRunner(
        InvalidPolicy(),
        tools={},
        config=AgentConfig(max_turns=1),
    ).run("test", episode_id="recovery-format")

    observation = result.steps[0].observation
    assert observation is not None
    assert "<tool_call>" in observation["expected_action_format"]
    assert "<answer>" in observation["expected_action_format"]
