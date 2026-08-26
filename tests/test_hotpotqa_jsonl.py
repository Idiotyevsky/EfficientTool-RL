import json

from efficienttool_rl.data import load_hotpotqa


def test_load_normalized_jsonl(tmp_path):
    path = tmp_path / "dev.jsonl"
    row = {
        "_id": "q1",
        "question": "Question?",
        "answer": "Answer",
        "supporting_facts": [["Title", 0]],
        "context": [["Title", ["Answer sentence."]]],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_hotpotqa(path, "validation")
    assert len(loaded) == 1
    assert loaded[0].passages[0].title == "Title"
