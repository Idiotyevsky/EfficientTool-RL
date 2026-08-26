from efficienttool_rl.evaluation import summarize_episodes


def test_summarize_episode_behavior():
    episodes = [
        {
            "termination_reason": "final_answer",
            "tool_calls": 2,
            "invalid_actions": 0,
            "steps": [{}, {}, {}],
        },
        {
            "termination_reason": "max_turns",
            "tool_calls": 0,
            "invalid_actions": 1,
            "steps": [{}, {}],
        },
    ]
    assert summarize_episodes(episodes) == {
        "episodes": 2,
        "completion_rate": 0.5,
        "avg_search_calls": 1.0,
        "avg_turns": 2.5,
        "invalid_action_rate": 0.2,
    }


def test_summarize_empty_episode_set():
    assert summarize_episodes([])["episodes"] == 0
