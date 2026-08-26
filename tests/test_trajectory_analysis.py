from efficienttool_rl.data import HotpotExample, Passage
from efficienttool_rl.evaluation.trajectory_analysis import analyze_trajectories


def test_analysis_detects_duplicate_search_and_evidence_recall():
    example = HotpotExample(
        example_id="q1",
        question="q",
        answer="yes",
        passages=(Passage("Gold", "text"),),
        supporting_titles=("Gold",),
        split="validation",
    )
    search_step = {
        "action": {"kind": "tool_call", "name": "search", "arguments": {"query": "Gold?"}},
        "observation": {"result": [{"title": "Gold", "passage": "text", "score": 1.0}]},
    }
    trajectory = {
        "episode_id": "q1",
        "final_answer": "no",
        "tool_calls": 2,
        "invalid_actions": 0,
        "steps": [search_step, search_step],
    }
    report, records = analyze_trajectories([trajectory], [example])
    assert report["duplicate_query_rate"] == 0.5
    assert report["avg_supporting_title_recall"] == 1.0
    assert records[0]["failure_category"] == "repeated_search"
