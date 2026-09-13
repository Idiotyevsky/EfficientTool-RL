import json

import pytest

from efficienttool_rl.data import HotpotExample, Passage
from efficienttool_rl.rewards.reward_v2 import (
    linear_annealed_beta,
    marginal_evidence_progress,
    reward_v2,
)
from efficienttool_rl.training import to_verl_record


def search(title: str, query: str) -> str:
    payload = {
        "ok": True,
        "tool": "search",
        "query": query,
        "results": [{"title": title, "passage": "evidence", "score": 1.0}],
    }
    return f"<tool_response>{json.dumps(payload)}</tool_response>"


EXTRA = {"supporting_titles": ["A", "B"]}


def test_new_supporting_evidence_has_positive_marginal_gain():
    progress = marginal_evidence_progress(search("A", "first"), EXTRA)
    assert progress["gains"] == (0.5,)
    assert progress["marginal_evidence_reward"] == pytest.approx(0.5)
    assert progress["unique_support_count"] == 1


def test_duplicate_support_has_zero_marginal_gain():
    progress = marginal_evidence_progress(
        search("A", "first") + search("A", "different query"), EXTRA
    )
    assert progress["gains"] == (0.5, 0.0)
    assert progress["marginal_evidence_reward"] == pytest.approx(0.5)


def test_second_new_support_receives_another_positive_gain():
    progress = marginal_evidence_progress(
        search("A", "first") + search("B", "bridge"), EXTRA
    )
    assert progress["gains"] == (0.5, 0.5)
    assert progress["marginal_evidence_reward"] == pytest.approx(1.0)
    assert progress["unique_support_count"] == 2


def test_irrelevant_retrieval_has_zero_marginal_gain():
    progress = marginal_evidence_progress(search("C", "irrelevant"), EXTRA)
    assert progress["gains"] == (0.0,)
    assert progress["marginal_evidence_reward"] == 0.0
    assert progress["unique_support_count"] == 0


def test_wasted_search_reuses_useful_search_accounting():
    response = (
        search("A", "first")
        + search("A", "first")
        + search("C", "irrelevant")
        + "<answer>London</answer>"
    )
    result = reward_v2(
        response,
        "London",
        extra_info=EXTRA,
        global_step=1,
        total_training_steps=62,
    )
    assert result["search_count"] == 3
    assert result["useful_search_count"] == 1
    assert result["wasted_search_count"] == 2
    assert result["repeated_search_count"] == 1


def test_zero_answer_reward_has_zero_wasted_penalty():
    response = search("C", "bad") + "<answer>wrong</answer>"
    result = reward_v2(
        response,
        "London",
        extra_info=EXTRA,
        global_step=1,
        total_training_steps=62,
    )
    assert result["answer_reward"] == 0.0
    assert result["wasted_search_count"] == 1
    assert result["wasted_search_penalty"] == 0.0
    assert result["total_reward"] == 0.0


def test_correct_answer_with_waste_receives_weak_success_gated_penalty():
    response = search("C", "irrelevant") + "<answer>London</answer>"
    result = reward_v2(
        response,
        "London",
        extra_info=EXTRA,
        global_step=62,
        total_training_steps=62,
        wasted_search_lambda=0.02,
    )
    assert result["answer_reward"] == 1.0
    assert result["wasted_search_penalty"] == pytest.approx(0.02)
    assert result["total_reward"] == pytest.approx(0.98)


def test_beta_linearly_anneals_from_point_one_to_zero():
    assert linear_annealed_beta(0, 11) == pytest.approx(0.10)
    assert linear_annealed_beta(1, 11) == pytest.approx(0.10)
    assert linear_annealed_beta(6, 11) == pytest.approx(0.05)
    assert linear_annealed_beta(11, 11) == pytest.approx(0.0)
    assert linear_annealed_beta(100, 11) == pytest.approx(0.0)


def test_beta_zero_without_waste_matches_task_only_reward():
    response = search("A", "first") + "<answer>London</answer>"
    result = reward_v2(
        response,
        "London",
        extra_info=EXTRA,
        global_step=62,
        total_training_steps=62,
    )
    assert result["current_beta"] == 0.0
    assert result["wasted_search_count"] == 0
    assert result["total_reward"] == result["answer_reward"] == 1.0
    assert "format_reward" not in result


def test_reward_metadata_is_not_model_visible():
    example = HotpotExample(
        example_id="q1",
        question="Where was the scientist born?",
        answer="London",
        passages=(
            Passage("A", "The scientist was born in London."),
            Passage("B", "A bridge document."),
        ),
        supporting_titles=("A", "B"),
        split="train",
    )
    record = to_verl_record(example, index=0, max_top_k=1)
    model_prompt = record["prompt"]
    tool_kwargs = record["extra_info"]["tools_kwargs"]

    assert "supporting_titles" not in json.dumps(model_prompt)
    assert "supporting_titles" not in json.dumps(tool_kwargs)
    assert "ground_truth" not in json.dumps(model_prompt)
    assert "ground_truth" not in json.dumps(tool_kwargs)
    assert record["reward_model"]["ground_truth"] == "London"
    assert record["extra_info"]["supporting_titles"] == ["A", "B"]
