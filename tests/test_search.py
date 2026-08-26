import pytest

from efficienttool_rl.data import Passage
from efficienttool_rl.tools import BM25Search


def test_bm25_ranks_relevant_passage_and_is_deterministic():
    passages = [
        Passage("Paris", "Paris is the capital of France."),
        Passage("Berlin", "Berlin is the capital of Germany."),
        Passage("Other", "A passage about marine biology."),
    ]
    search = BM25Search(passages)
    first = search.search("capital France", top_k=2)
    second = search.search("capital France", top_k=2)
    assert first == second
    assert first[0].title == "Paris"
    assert first[0].score > first[1].score


def test_search_returns_structured_bounded_results():
    search = BM25Search([Passage("One", "one two three four")], max_observation_tokens=2)
    result = search.tool({"query": "one", "top_k": 1})[0]
    assert result["title"] == "One"
    assert result["passage"] == "one two"
    assert result["score"] > 0


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(query):
    search = BM25Search([Passage("One", "text")])
    with pytest.raises(ValueError, match="query"):
        search.search(query)
