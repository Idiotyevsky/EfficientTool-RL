import pytest

from efficienttool_rl.protocol import (
    FinalAnswer,
    InvalidAction,
    ToolCall,
    canonicalize_action_text,
    count_tool_call_attempts,
    parse_action,
)


def test_parses_valid_tool_call() -> None:
    action = parse_action(
        '<tool_call>{"name":"search","arguments":{"query":"hotpot"}}</tool_call>'
    )
    assert action == ToolCall(name="search", arguments={"query": "hotpot"})


def test_allows_reasoning_around_one_action() -> None:
    action = parse_action('reasoning\n<answer>  final text  </answer>\ntrailing')
    assert action == FinalAnswer(answer="final text")


def test_canonicalizes_valid_action_at_first_complete_block() -> None:
    assert (
        canonicalize_action_text('<answer>Paris</answer> repeated answer text')
        == '<answer>Paris</answer>'
    )
    assert canonicalize_action_text(
        'reasoning\n<tool_call>{"name":"search","arguments":{"query":"Paris"}}</tool_call>tail'
    ).endswith('</tool_call>')


def test_canonicalization_preserves_invalid_output_for_diagnostics() -> None:
    malformed = '<answer>Paris</answer><answer>London</answer>'
    assert canonicalize_action_text(malformed) == malformed


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("plain text", "no_action"),
        ("<tool_call>{bad}</tool_call>", "malformed_json"),
        ('<tool_call>{"name":"search"}</tool_call>', "missing_arguments"),
        ('<tool_call>{"name":"search","arguments":[]}</tool_call>', "invalid_arguments"),
        ("<answer></answer>", "empty_answer"),
        ("<answer>x", "malformed_tag"),
        ("<answer>x</answer><answer>y</answer>", "multiple_actions"),
    ],
)
def test_invalid_actions_are_structured(text: str, code: str) -> None:
    action = parse_action(text)
    assert isinstance(action, InvalidAction)
    assert action.code == code


def test_non_string_output_is_invalid() -> None:
    action = parse_action(None)  # type: ignore[arg-type]
    assert isinstance(action, InvalidAction)
    assert action.code == "invalid_output_type"


def test_tool_call_attempt_count_includes_unclosed_and_repeated_blocks() -> None:
    text = '<tool_call>{bad}</tool_call><tool_call>{"name":"search"}'
    assert count_tool_call_attempts(text) == 2
