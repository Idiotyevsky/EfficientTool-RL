import pytest

from efficienttool_rl.rewards import extract_final_answer, task_reward


def test_extracts_one_answer_after_tool_call():
    response = '<tool_call>{"name":"search","arguments":{"query":"q"}}</tool_call>\n<answer>London</answer>'
    assert extract_final_answer(response) == "London"


def test_extracts_answer_from_native_verl_tool_trajectory():
    response = """<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"q\"}}</tool_call>
user
<tool_response>{\"ok\":true}</tool_response>
assistant
<think>The evidence says Delhi.</think>
<answer>Delhi</answer>"""
    assert extract_final_answer(response) == "Delhi"
    assert task_reward(response, "Delhi")["score"] == 1.0


def test_rejects_untagged_empty_and_multiple_answers():
    assert extract_final_answer("London") is None
    assert extract_final_answer("<answer> </answer>") is None
    assert extract_final_answer("<answer>a</answer><answer>b</answer>") is None
    assert extract_final_answer("explanation <answer>a</answer>") is None


def test_task_reward_is_half_em_half_f1():
    reward = task_reward("<answer>red blue</answer>", "red green")
    assert reward == {
        "score": pytest.approx(0.25),
        "em": 0.0,
        "f1": pytest.approx(0.5),
        "valid_answer": 1.0,
    }


def test_correct_answer_has_maximum_reward():
    assert task_reward("<answer>The Eiffel Tower</answer>", "Eiffel Tower")["score"] == 1.0


def test_invalid_format_gets_zero_reward():
    assert task_reward("Eiffel Tower", "Eiffel Tower")["score"] == 0.0


def test_alpha_must_be_valid():
    with pytest.raises(ValueError, match="alpha"):
        task_reward("<answer>x</answer>", "x", alpha=1.1)
