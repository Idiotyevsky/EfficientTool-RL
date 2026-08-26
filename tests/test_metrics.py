import pytest

from efficienttool_rl.evaluation.metrics import answer_metrics, exact_match, token_f1


def test_exact_match_normalizes_case_articles_and_punctuation():
    assert exact_match("The Eiffel Tower!", "eiffel tower") == 1.0


def test_token_f1_uses_token_overlap():
    assert token_f1("red blue", "red green") == pytest.approx(0.5)


def test_answer_metrics_reports_both_values():
    assert answer_metrics("London", "London") == {"exact_match": 1.0, "f1": 1.0}
