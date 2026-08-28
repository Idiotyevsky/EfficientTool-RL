import pytest

from scripts.analyze_cost_reward import (
    TrajectoryScore,
    build_pair_checks,
    build_reward_landscape,
    counterfactual_reward,
    pairwise_rank_stats,
)


def make_record(
    episode_id: str,
    task_reward: float,
    *,
    em: float = 0.0,
    executed: int = 0,
    useful: int = 0,
) -> TrajectoryScore:
    return TrajectoryScore(
        source="test",
        episode_id=episode_id,
        task_reward=task_reward,
        em=em,
        f1=task_reward,
        valid_answer=float(task_reward > 0),
        executed_search_calls=executed,
        useful_search_calls=useful,
        wasted_search_calls=executed - useful,
    )


def test_counterfactual_reward_only_charges_waste():
    assert counterfactual_reward(1.0, 0, 0.3) == {
        "task_reward": 1.0,
        "cost_penalty": 0.0,
        "total_reward": 1.0,
    }
    assert counterfactual_reward(1.0, 1, 0.3)["total_reward"] == pytest.approx(0.7)
    assert counterfactual_reward(0.0, 4, 0.3)["cost_penalty"] == 0.0


def test_counterfactual_reward_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="task_score"):
        counterfactual_reward(1.1, 0, 0.1)
    with pytest.raises(ValueError, match="lambda_cost"):
        counterfactual_reward(1.0, 0, -0.1)
    with pytest.raises(ValueError, match="wasted_search_calls"):
        counterfactual_reward(1.0, -1, 0.1)


def test_landscape_protects_zero_task_and_preserves_useful_calls():
    landscape = build_reward_landscape([0.0, 0.3])
    rows = {row["case"]: row for row in landscape["rows"]}
    assert rows["correct_2_useful_0_waste"]["values"][1]["total_reward"] == 1.0
    assert rows["correct_2_useful_1_waste"]["values"][1]["total_reward"] == pytest.approx(0.7)
    assert rows["partial_task_reward_0.50_1_waste"]["values"][1]["total_reward"] == pytest.approx(0.35)
    assert rows["wrong_0_search"]["values"][1]["total_reward"] == 0.0


def test_pairwise_stats_prefer_lower_waste_without_reversing_correctness():
    records = [
        make_record("correct_clean", 1.0, em=1.0, executed=2, useful=2),
        make_record("correct_waste", 1.0, em=1.0, executed=3, useful=2),
        make_record("wrong_zero", 0.0, executed=0, useful=0),
    ]
    stats = pairwise_rank_stats(records, 0.3)
    assert stats["task_order_inversions"] == 0
    assert stats["lower_waste_preference_pairs"] == 1
    assert stats["correct_vs_wrong_inversions"] == 0
    assert stats["correct_two_search_vs_wrong_zero_inversions"] == 0


def test_pair_checks_require_same_task_quality_for_isolated_comparisons():
    records = [
        make_record("clean", 1.0, em=1.0, executed=2, useful=2),
        make_record("waste", 1.0, em=1.0, executed=3, useful=2),
        make_record("wrong", 0.0, executed=0),
        make_record("partial", 0.5, executed=2, useful=1),
    ]
    checks = build_pair_checks({"test": records}, [0.0, 0.3])["test"]
    statuses = {item["name"]: item["status"] for item in checks}
    assert statuses["correct_useful_vs_correct_wasted"] == "PASS"
    assert statuses["correct_two_search_vs_wrong_zero"] == "PASS"
    assert statuses["partial_with_waste_vs_wrong_zero"] == "PASS"
    assert statuses["same_task_quality_no_waste"] == "UNAVAILABLE"
