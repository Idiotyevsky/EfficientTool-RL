from efficienttool_rl.agent import EpisodeResult, EpisodeStep
from efficienttool_rl.evaluation import episode_search_usage


def episode(steps: list[EpisodeStep]) -> EpisodeResult:
    return EpisodeResult(
        episode_id="e1",
        prompt="q",
        steps=steps,
        final_answer=None,
        termination_reason="max_turns",
        tool_calls=0,
        invalid_actions=0,
        attempted_tool_calls=0,
        valid_tool_calls=0,
        executed_tool_calls=0,
        executed_search_calls=0,
    )


def tool_call_step(title: str, *, executed: bool = True) -> EpisodeStep:
    return EpisodeStep(
        turn=0,
        model_output="<tool_call>{}</tool_call>",
        action={"kind": "tool_call", "name": "search", "arguments": {"query": title}},
        observation={
            "ok": True,
            "tool": "search",
            "result": [{"title": title, "text": "passage", "score": 1.0}],
        },
        terminated=False,
        tool_executed=executed,
    )


def test_counts_executed_useful_and_wasted_searches():
    steps = [
        tool_call_step("Gold A"),
        tool_call_step("Gold A"),
        tool_call_step("Irrelevant"),
    ]
    usage = episode_search_usage(episode(steps), supporting_titles=["Gold A", "Gold B"])
    assert usage == {
        "executed_search_calls": 3,
        "useful_search_calls": 1,
        "wasted_search_calls": 2,
    }


def test_ignores_non_search_and_unexecuted_steps():
    answer_step = EpisodeStep(
        turn=1,
        model_output="<answer>x</answer>",
        action={"kind": "answer", "answer": "x"},
        observation=None,
        terminated=True,
    )
    failed_step = tool_call_step("Gold A", executed=False)
    usage = episode_search_usage(episode([answer_step, failed_step]), ["Gold A"])
    assert usage == {
        "executed_search_calls": 0,
        "useful_search_calls": 0,
        "wasted_search_calls": 0,
    }
