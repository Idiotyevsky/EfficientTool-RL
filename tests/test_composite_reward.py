import pytest

from efficienttool_rl.rewards.composite import (
    composite_reward,
    episode_composite_reward,
    evidence_coverage,
    episode_format_components,
    format_components_from_text,
    retrieved_titles_from_text,
)
from efficienttool_rl.rewards.task import task_reward


def native_search(title: str, query: str) -> str:
    return (
        "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\""
        + query
        + "\"}}</tool_call>"
        f"<tool_response>{{\"ok\":true,\"tool\":\"search\",\"query\":\"{query}\","
        f"\"results\":[{{\"title\":\"{title}\",\"text\":\"evidence\",\"score\":1.0}}]}}"
        "</tool_response>"
    )


def native_response(titles_and_queries, answer="Paris") -> str:
    parts = [native_search(title, query) for title, query in titles_and_queries]
    parts.append(f"<answer>{answer}</answer>")
    return "".join(parts)


EXTRA = {"supporting_titles": ["Paris", "France"]}


def test_evidence_coverage_is_saturating_and_deduplicates():
    assert evidence_coverage([], ["A", "B"])["evidence_coverage"] == 0.0
    assert evidence_coverage(["A"], ["A", "B"])["evidence_coverage"] == 0.5
    full = evidence_coverage(["A", "B", "A", "C"], ["A", "B"])
    assert full["evidence_coverage"] == 1.0
    assert full["evidence_found_count"] == 2.0
    assert evidence_coverage(["A"], [])["evidence_coverage"] == 0.0


def test_retrieved_titles_from_text_ignores_failed_payloads():
    text = (
        native_search("A", "one")
        + "<tool_response>{\"ok\":false,\"error\":{}}</tool_response>"
        + native_search("B", "two")
    )
    assert retrieved_titles_from_text(text) == ["A", "B"]


def test_format_components_positive_for_clean_trajectory():
    parts = format_components_from_text(native_response([("Paris", "q1")]))
    assert parts == {
        "format_valid_answer": 1.0,
        "format_no_malformed_calls": 1.0,
        "format_ended_with_answer": 1.0,
    }


def test_format_components_flag_malformed_call_and_missing_answer():
    parts = format_components_from_text(
        "<tool_call>{not json}</tool_call> some prose"
    )
    assert parts["format_no_malformed_calls"] == 0.0
    assert parts["format_valid_answer"] == 0.0
    assert parts["format_ended_with_answer"] == 0.0


def test_format_components_detect_tool_response_after_answer():
    text = "<answer>Paris</answer>" + native_search("A", "late")
    parts = format_components_from_text(text)
    assert parts["format_ended_with_answer"] == 0.0
    assert parts["format_valid_answer"] == 1.0


def test_composite_reward_components_and_weights():
    response = native_response([("Paris", "capital")])
    result = composite_reward(response, "Paris", extra_info=EXTRA)
    answer = task_reward(response, "Paris", alpha=0.5)
    assert result["answer_reward"] == answer["score"] == result["em"]
    assert result["evidence_reward"] == 0.5
    assert result["format_reward"] == 1.0
    expected = 0.8 * answer["score"] + 0.15 * 0.5 + 0.05 * 1.0
    assert result["score"] == pytest.approx(expected)
    assert result["executed_search_calls"] == 1
    assert result["useful_search_calls"] == 1


def test_more_searches_do_not_increase_saturated_evidence_reward():
    one = composite_reward(native_response([("Paris", "a")]), "Paris", extra_info=EXTRA)
    many = composite_reward(
        native_response([("Paris", "a"), ("Paris", "b"), ("France", "c")]),
        "Paris",
        extra_info=EXTRA,
    )
    assert many["evidence_reward"] == 1.0 > one["evidence_reward"]
    assert many["executed_search_calls"] == 3
    assert many["useful_search_calls"] == 2
    assert many["evidence_reward"] <= one["evidence_reward"] + 0.5


def test_missing_metadata_yields_zero_evidence_with_flag():
    result = composite_reward(native_response([("Paris", "a")]), "Paris", extra_info={})
    assert result["evidence_reward"] == 0.0
    assert result["evidence_metadata_available"] == 0.0
    assert result["answer_reward"] > 0.0


def test_zero_weights_beyond_answer_are_allowed_but_degenerate_rejected():
    only_answer = composite_reward(
        native_response([("Paris", "a")]),
        "Paris",
        extra_info=EXTRA,
        evidence_weight=0.0,
        format_weight=0.0,
    )
    assert only_answer["score"] == pytest.approx(0.8 * only_answer["answer_reward"])
    # Zero weights remove a component from the total but the raw component
    # values stay visible for diagnostics.
    assert only_answer["evidence_reward"] == 0.5
    assert only_answer["format_reward"] == 1.0
    with pytest.raises(ValueError):
        composite_reward("x", "y", answer_weight=0.0, evidence_weight=0.0, format_weight=0.0)
    with pytest.raises(ValueError):
        composite_reward("x", "y", answer_weight=-1.0)


def test_episode_composite_reward_matches_component_semantics():
    episode = {
        "final_answer": "Paris",
        "termination_reason": "final_answer",
        "invalid_actions": 0,
        "steps": [
            {
                "tool_executed": True,
                "action": {"kind": "tool_call", "name": "search", "arguments": {}},
                "observation": {
                    "ok": True,
                    "result": [{"title": "Paris", "text": "e", "score": 1.0}],
                },
            }
        ],
    }
    result = episode_composite_reward(episode, "Paris", supporting_titles=["Paris", "France"])
    assert result["em"] == 1.0
    assert result["evidence_reward"] == 0.5
    assert result["format_reward"] == 1.0
    assert result["executed_search_calls"] == 1
    expected = 0.8 * 1.0 + 0.15 * 0.5 + 0.05 * 1.0
    assert result["score"] == pytest.approx(expected)


def test_episode_format_components_flag_incomplete_trajectory():
    episode = {
        "final_answer": None,
        "termination_reason": "max_turns",
        "invalid_actions": 1,
        "steps": [],
    }
    parts = episode_format_components(episode)
    assert parts == {
        "format_valid_answer": 0.0,
        "format_no_malformed_calls": 0.0,
        "format_ended_with_answer": 0.0,
    }
