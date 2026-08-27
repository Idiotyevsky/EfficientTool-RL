from efficienttool_rl.evaluation.verl_analysis import (
    analyze_verl_behavior,
    analyze_verl_rollouts,
    classify_tool_calls,
)


def test_classify_tool_calls_mirrors_native_parser_boundary():
    output = (
        '<tool_call>{"name":"search","arguments":{"query":"Paris"}}</tool_call>'
        '<tool_call>{"name":"search"}</tool_call>'
        '<tool_call>{"name":"calculator","arguments":{}}</tool_call>'
    )

    assert classify_tool_calls(output) == {
        "attempted_tool_call_count": 3,
        "tool_call_count": 3,
        "valid_tool_call_count": 2,
        "valid_search_call_count": 1,
        "executed_tool_call_count": 0,
        "executed_search_call_count": 0,
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


class _ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def test_behavior_summary_reports_search_turn_and_length_distributions():
    rows = [
        {
            "output": (
                '<tool_call>{"name":"search","arguments":{"query":"Paris"}}</tool_call>'
                'user\n<tool_response>{"ok":true,"tool":"search","query":"Paris","results":[]}'
                '</tool_response>\nassistant\n<answer>Paris</answer>'
            ),
            "gts": "Paris",
        },
        {
            "output": (
                '<tool_call>{"name":"search","arguments":{"query":"Paris"}}</tool_call>'
                'user\n<tool_response>{"ok":true,"tool":"search","query":"Paris","results":[]}'
                '</tool_response>\nassistant\n'
                '<tool_call>{"name":"search","arguments":{"query":"Paris"}}</tool_call>'
                'user\n<tool_response>{"ok":true,"tool":"search","query":"Paris","results":[]}'
                '</tool_response>\nassistant\n<answer>London</answer>'
            ),
            "gts": "London",
        },
    ]

    report = analyze_verl_behavior(rows, tokenizer=_ToyTokenizer())

    assert report["search_count_distribution"] == {1: 1, 2: 1}
    assert report["executed_search_call_count"] == 3
    assert report["attempted_tool_call_count"] == 3
    assert report["turn_distribution"] == {2: 1, 3: 1}
    assert report["avg_assistant_turns"] == 2.5
    assert report["verl_num_turn_distribution_estimate"] == {4: 1, 6: 1}
    assert report["duplicate_query_rate"] == 1 / 3
    assert report["generated_token_stats"]["count"] == 2
    assert report["accuracy_by_search_count"]["em"][1]["mean"] == 1.0


def test_behavior_summary_counts_useful_searches_from_successful_responses():
    rows = [
        {
            "input": "system\nuser\nWhich city?\nassistant\n",
            "output": (
                '<tool_call>{"name":"search","arguments":{"query":"city"}}</tool_call>'
                'user\n<tool_response>{"ok":true,"tool":"search","query":"city",'
                '"results":[{"title":"Bridge","passage":"..."}]}</tool_response>\n'
                'assistant\n<answer>Paris</answer>'
            ),
            "gts": "Paris",
        }
    ]
    report = analyze_verl_behavior(
        rows,
        supporting_titles_by_question={"Which city?": ("Bridge", "Answer")},
    )

    assert report["supporting_title_metadata_available"] is True
    assert report["attempted_tool_call_count"] == 1
    assert report["valid_tool_call_count"] == 1
    assert report["executed_search_call_count"] == 1
    assert report["useful_search_call_count"] == 1
    assert report["wasted_search_call_count"] == 0
    assert report["tool_efficiency"] == 1.0
    assert report["first_search_support_recall"] == 0.5
