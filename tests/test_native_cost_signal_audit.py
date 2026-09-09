from scripts.audit_native_cost_signal import (
    native_rows_to_audit_rows,
    question_from_input,
)


def test_question_from_native_input():
    assert question_from_input("system\nuser\nWhat year?\nassistant\n") == "What year?"


def test_native_rows_replay_task_and_wasted_search_counts():
    output = (
        '<tool_response>{"ok":true,"tool":"search","query":"a",'
        '"results":[{"title":"A","passage":"evidence"}]}</tool_response>'
        '<tool_response>{"ok":true,"tool":"search","query":"repeat",'
        '"results":[{"title":"A","passage":"evidence"}]}</tool_response>'
        '<answer>Paris</answer>'
    )
    rows = native_rows_to_audit_rows(
        [
            {
                "input": "system\nuser\nWhat city?\nassistant\n",
                "output": output,
                "gts": "Paris",
            }
        ],
        {"What city?": ("A", "B")},
    )

    assert rows == [
        {
            "input": "system\nuser\nWhat city?\nassistant\n",
            "task_reward": 1.0,
            "wasted_search_calls": 1,
            "executed_search_calls": 2,
            "useful_search_calls": 1,
        }
    ]
