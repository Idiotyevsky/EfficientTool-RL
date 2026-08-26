from efficienttool_rl.evaluation.verl_analysis import analyze_verl_rollouts, classify_tool_calls


def test_classify_tool_calls_mirrors_native_parser_boundary():
    output = (
        '<tool_call>{"name":"search","arguments":{"query":"Paris"}}</tool_call>'
        '<tool_call>{"name":"search"}</tool_call>'
        '<tool_call>{"name":"calculator","arguments":{}}</tool_call>'
    )

    assert classify_tool_calls(output) == {
        "tool_call_count": 3,
        "valid_tool_call_count": 2,
        "malformed_tool_call_count": 1,
        "unknown_tool_call_count": 1,
    }


def test_native_rollouts_are_grouped_by_prompt_and_reward_is_replayed():
    rows = [
        {
            "input": "prompt-a",
            "output": '<tool_call>{}</tool_call>\n<answer>Delhi</answer>',
            "gts": "Delhi",
        },
        {"input": "prompt-b", "output": "untagged", "gts": "Delhi"},
        {
            "input": "prompt-a",
            "output": '<tool_call>{}</tool_call>\n<answer>wrong</answer>',
            "gts": "Delhi",
        },
    ]

    report = analyze_verl_rollouts(rows)

    assert report["episodes"] == 3
    assert report["groups"] == 2
    assert report["group_size_histogram"] == {1: 1, 2: 1}
    assert report["mean_reward"] == 1 / 3
    assert report["nontrivial_reward_group_ratio"] == 0.5
    assert report["all_groups_have_trajectory_diversity"] is False
    assert report["literal_tool_call_count_mean"] == 2 / 3
    assert report["malformed_tool_call_count_mean"] == 2 / 3
    assert report["malformed_tool_call_episode_rate"] == 2 / 3
    assert report["malformed_tool_call_rate"] == 1.0
