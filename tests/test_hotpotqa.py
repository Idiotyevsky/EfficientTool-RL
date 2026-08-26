import json

import pytest

from efficienttool_rl.data import load_hotpotqa


def test_load_official_hotpot_shape(tmp_path):
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "Where was Ada born?",
                    "answer": "London",
                    "supporting_facts": [["Ada Lovelace", 0], ["London", 1]],
                    "context": [
                        ["Ada Lovelace", ["Ada was born in London."]],
                        ["London", ["London is in England."]],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    example = load_hotpotqa(path, split="dev")[0]
    assert example.example_id == "q1"
    assert example.split == "dev"
    assert example.passages[0].text == "Ada was born in London."
    assert example.supporting_titles == ("Ada Lovelace", "London")


def test_loader_rejects_duplicate_ids(tmp_path):
    item = {"_id": "same", "question": "q", "answer": "a", "context": []}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([item, item]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_hotpotqa(path, split="train")
