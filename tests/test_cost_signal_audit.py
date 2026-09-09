import pytest

from scripts.audit_cost_signal import audit_batch


def row(prompt: str, task: float, waste: int) -> dict[str, object]:
    return {
        "input": prompt,
        "task_reward": task,
        "wasted_search_calls": waste,
    }


def test_cost_active_group_and_advantage_delta_are_distinct_from_row_rate():
    report = audit_batch(
        [
            row("p1", 1.0, 1),
            row("p1", 0.8, 0),
            row("p1", 0.2, 0),
            row("p1", 0.0, 0),
            row("p2", 0.0, 3),
            row("p2", 0.0, 0),
        ],
        lambda_cost=0.05,
    )

    assert report["rows"] == 6
    assert report["groups"] == 2
    assert report["nonzero_penalty_count"] == 1
    assert report["cost_active_group_count"] == 1
    assert report["zero_cost_group_rate"] == pytest.approx(0.5)
    assert report["advantage_changed_group_count"] == 1
    assert report["mean_abs_advantage_delta"] > 0


def test_equal_task_quality_gets_a_lower_waste_tiebreak_without_flip():
    report = audit_batch(
        [
            row("p1", 1.0, 0),
            row("p1", 1.0, 1),
            row("p1", 0.0, 0),
            row("p1", 0.0, 0),
        ],
        lambda_cost=0.1,
    )

    assert report["ranking_flip_count"] == 0
    assert report["lower_waste_tiebreak_count"] == 1


def test_batches_are_grouped_by_prompt_not_row_position():
    report = audit_batch(
        [
            row("p1", 1.0, 0),
            row("p2", 0.0, 0),
            row("p1", 0.0, 1),
            row("p2", 1.0, 0),
        ],
        lambda_cost=0.05,
    )

    assert report["groups"] == 2
    assert report["group_size_histogram"] == {2: 2}
