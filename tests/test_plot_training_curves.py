import json

from scripts.plot_training_curves import (
    aggregate_jsonl,
    load_run_directory,
    parse_console_metrics,
    to_long_rows,
)


def test_parse_console_metrics_handles_ansi_and_resumed_steps(tmp_path):
    first = tmp_path / "first.log"
    first.write_text(
        "\x1b[36m(Runner pid=1)\x1b[0m step:1 - actor/entropy:0.25 "
        "- actor/grad_norm:3.5 - training/global_step:1\n",
        encoding="utf-8",
    )
    resumed = tmp_path / "resumed.log"
    resumed.write_text(
        "step:1 - actor/entropy:0.20 - actor/grad_norm:3.0\n"
        "step:2 - actor/entropy:1.2e-1 - actor/grad_norm:2.5\n",
        encoding="utf-8",
    )
    metrics = parse_console_metrics([first, resumed])
    assert metrics[1]["actor/entropy"] == 0.20
    assert metrics[1]["actor/grad_norm"] == 3.0
    assert metrics[2]["actor/entropy"] == 0.12
    assert metrics[2]["training/global_step"] == 2.0


def test_aggregate_jsonl_computes_means_and_group_variance(tmp_path):
    path = tmp_path / "1.jsonl"
    rows = [
        {"input": "q1", "score": 1.0, "search_count": 2, "multi_search": 1.0},
        {"input": "q1", "score": 0.0, "search_count": 1, "multi_search": 0.0},
        {"input": "q2", "score": 0.5, "search_count": 2, "multi_search": 1.0},
        {"input": "q2", "score": 0.5, "search_count": 2, "multi_search": 1.0},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metrics = aggregate_jsonl(path)
    assert metrics["total_reward"] == 0.5
    assert metrics["search_count"] == 1.75
    assert metrics["multi_search"] == 0.75
    assert metrics["zero_variance_group_ratio"] == 0.5
    assert metrics["trajectory_count"] == 4.0
    assert metrics["group_count"] == 2.0


def test_aggregate_jsonl_recovers_legacy_search_metrics_from_output(tmp_path):
    path = tmp_path / "1.jsonl"
    rows = [
        {
            "input": "system\n...\nuser\nquestion one\nassistant",
            "gts": "answer",
            "output": (
                '<tool_call>{"name":"search","arguments":{"query":"first"}}'
                "</tool_call>\n"
                '<tool_response>{"ok":true,"tool":"search","query":"first",'
                '"results":[]}</tool_response>\n'
                '<tool_call>{"name":"search","arguments":{"query":"second"}}'
                "</tool_call>\n"
                '<tool_response>{"ok":true,"tool":"search","query":"second",'
                '"results":[]}</tool_response>\n'
                "<answer>answer</answer>"
            ),
            "score": 1.0,
        },
        {
            "input": "system\n...\nuser\nquestion two\nassistant",
            "gts": "answer",
            "output": "<answer>answer</answer>",
            "score": 1.0,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    metrics = aggregate_jsonl(path)

    assert metrics["search_count"] == 1.0
    assert metrics["multi_search"] == 0.5


def test_load_run_directory_keeps_observed_steps_only(tmp_path):
    rollouts, validation = tmp_path / "rollouts", tmp_path / "validation"
    rollouts.mkdir()
    validation.mkdir()
    (rollouts / "1.jsonl").write_text(
        json.dumps({"input": "q", "answer_reward": 0.5}) + "\n",
        encoding="utf-8",
    )
    (rollouts / "3.jsonl").write_text(
        json.dumps({"input": "q", "answer_reward": 1.0}) + "\n",
        encoding="utf-8",
    )
    (validation / "0.jsonl").write_text(
        json.dumps({"input": "q", "em": 0.0, "f1": 0.5}) + "\n",
        encoding="utf-8",
    )
    run = load_run_directory(tmp_path)
    rows = to_long_rows({"run": run})
    assert set(run["train"]) == {1, 3}
    assert set(run["validation"]) == {0}
    assert any(row["metric"] == "f1" and row["split"] == "validation" for row in rows)
