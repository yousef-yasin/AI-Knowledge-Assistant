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
    # Raw BM25 score — not on a fixed 0-1 range, only meaningful for ranking within this index.
    score: float


class BM25Index:
    """Keyword search index built with BM25, used as the keyword half of hybrid search."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    def build(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Build the BM25 index from the full set of stored documents.

        Args:
            ids: Unique id for each chunk, same order as documents.
            documents: Raw chunk text, same order as ids.
            metadatas: Metadata dict for each chunk, same order as ids.
        """

        if not documents:
            self._bm25 = None
            self._ids, self._documents, self._metadatas = [], [], []
            return

        self._ids = ids
        self._documents = documents
        self._metadatas = metadatas

        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[KeywordMatch]:
        """
        Search the index for the top_k best keyword matches.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            KeywordMatch objects, highest BM25 score first. Chunks with
            zero keyword overlap are excluded rather than returned with a
            meaningless zero score.
        """

        if self._bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            KeywordMatch(
                chunk_id=self._ids[i],
                text=self._documents[i],
                metadata=self._metadatas[i],
                score=float(scores[i]),
            )
            for i in ranked_indices
            if scores[i] > 0
        ]

    def _tokenize(self, text: str) -> list[str]:
        """Lowercases and splits text into simple word tokens for BM25 matching."""
        return re.findall(r"\w+", text.lower())
