import pytest

from efficienttool_rl.rewards.cost_aware import cost_aware_reward, search_usage


def native_response(*, title: str, query: str) -> str:
    return (
        '<tool_response>{"ok":true,"query":'
        + repr(query).replace("'", '"')
        + ',"results":[{"title":'
        + repr(title).replace("'", '"')
        + ',"passage":"evidence"}],"tool":"search"}</tool_response>'
    )


def test_search_usage_counts_new_supporting_titles_only_once():
    response = (
        native_response(title="A", query="first")
        + native_response(title="A", query="repeat")
        + native_response(title="B", query="second")
    )
    assert search_usage(response, {"supporting_titles": ["A", "B"]}) == {
        "executed_search_calls": 3,
        "useful_search_calls": 2,
        "wasted_search_calls": 1,
    }


def test_cost_aware_reward_charges_waste_but_not_useful_search():
    response = (
        native_response(title="A", query="first")
        + native_response(title="A", query="repeat")
        + "<answer>London</answer>"
    )
    reward = cost_aware_reward(
        response,
        "London",
        extra_info={"supporting_titles": ["A", "B"]},
        lambda_cost=0.05,
    )
    assert reward["task_reward"] == pytest.approx(1.0)
    assert reward["cost_penalty"] == pytest.approx(0.05)
    assert reward["score"] == pytest.approx(0.95)
    assert reward["executed_search_calls"] == 2
    assert reward["useful_search_calls"] == 1
    assert reward["wasted_search_calls"] == 1


def test_failed_answer_does_not_receive_a_search_cost_penalty():
    response = native_response(title="A", query="first") + "<answer>wrong</answer>"
    reward = cost_aware_reward(
        response,
        "London",
        extra_info={"supporting_titles": ["A"]},
        lambda_cost=0.3,
    )
    assert reward["task_reward"] == 0.0
    assert reward["cost_penalty"] == 0.0
    assert reward["score"] == 0.0


def test_cost_aware_reward_fails_closed_without_annotation():
    with pytest.raises(ValueError, match="supporting_titles"):
        cost_aware_reward("<answer>London</answer>", "London")
