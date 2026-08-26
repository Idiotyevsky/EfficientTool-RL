"""Dependency-free, deterministic Okapi BM25 retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

from ..data.hotpotqa import Passage

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


@dataclass(frozen=True)
class SearchResult:
    title: str
    passage: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BM25Search:
    def __init__(
        self,
        passages: Iterable[Passage],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        max_observation_tokens: int = 512,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("require k1 > 0 and 0 <= b <= 1")
        if max_observation_tokens < 1:
            raise ValueError("max_observation_tokens must be positive")
        self.k1 = k1
        self.b = b
        self.max_observation_tokens = max_observation_tokens
        self.passages = tuple(passages)
        if not self.passages:
            raise ValueError("at least one passage is required")

        self._tokens = [tokenize(f"{p.title} {p.text}") for p in self.passages]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._avg_length = sum(map(len, self._tokens)) / len(self._tokens)
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        count = len(self.passages)
        self._idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self._term_frequencies):
            length = len(self._tokens[index])
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                norm = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self._avg_length, 1.0)
                )
                score += self._idf.get(term, 0.0) * frequency * (self.k1 + 1) / norm
            scored.append((score, index))

        # Corpus order is the explicit deterministic tie-breaker.
        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[SearchResult] = []
        remaining = self.max_observation_tokens
        for score, index in scored[: min(top_k, len(scored))]:
            passage = self.passages[index]
            words = passage.text.split()
            bounded = " ".join(words[:remaining])
            remaining -= len(bounded.split())
            results.append(
                SearchResult(title=passage.title, passage=bounded, score=round(score, 8))
            )
            if remaining == 0:
                break
        return results

    def tool(self, arguments: dict[str, object]) -> list[dict[str, object]]:
        query = arguments.get("query")
        top_k = arguments.get("top_k", 3)
        if not isinstance(query, str):
            raise ValueError("search.query must be a string")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("search.top_k must be an integer")
        return [result.to_dict() for result in self.search(query, top_k)]
