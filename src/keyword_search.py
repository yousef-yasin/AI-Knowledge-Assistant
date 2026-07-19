from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi


@dataclass
class KeywordMatch:
    """Stores one BM25 keyword search result."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    
    score: float


class BM25Index:
    """Keyword search index used in hybrid retrieval."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []

    def build(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not (
            len(ids)
            == len(documents)
            == len(metadatas)
        ):
            raise ValueError(
                "ids, documents, and metadatas "
                "must have equal lengths."
            )

        self._ids = list(ids)
        self._documents = list(documents)
        self._metadatas = [
            dict(metadata or {})
            for metadata in metadatas
        ]

        if not documents:
            self._bm25 = None
            self._tokenized_corpus = []
            return

        self._tokenized_corpus = [
            self._tokenize(document)
            for document in documents
        ]

        self._bm25 = BM25Okapi(
            self._tokenized_corpus
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[KeywordMatch]:
        if top_k <= 0:
            raise ValueError(
                "top_k must be a positive integer."
            )

        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = [
            float(score)
            for score in self._bm25.get_scores(
                query_tokens
            )
        ]

        # BM25Okapi may return all-zero scores for very
        # small corpora. Use token overlap as fallback.
        if not any(score > 0 for score in scores):
            query_set = set(query_tokens)

            scores = [
                float(
                    len(
                        query_set.intersection(
                            document_tokens
                        )
                    )
                )
                for document_tokens
                in self._tokenized_corpus
            ]

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[KeywordMatch] = []

        for index in ranked_indices:
            score = scores[index]

            if score <= 0:
                continue

            results.append(
                KeywordMatch(
                    chunk_id=self._ids[index],
                    text=self._documents[index],
                    metadata=self._metadatas[index],
                    score=score,
                )
            )

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\w+",
            text.casefold(),
            flags=re.UNICODE,
        )