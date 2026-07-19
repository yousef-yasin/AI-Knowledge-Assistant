from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.citations import CitationManager
from src.embeddings import EmbeddingGenerator
from src.keyword_search import BM25Index
from src.query_rewriter import QueryRewriter
from src.vector_store import VectorDB


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float


class RetrievalEngine:
    """
    Semantic, keyword, hybrid, and multi-query retrieval.
    """

    def __init__(
        self,
        vector_db: VectorDB,
        embedding_generator: (
            EmbeddingGenerator | None
        ) = None,
        citation_manager: (
            CitationManager | None
        ) = None,
        query_rewriter: (
            QueryRewriter | None
        ) = None,
    ) -> None:
        self.vector_db = vector_db

        self.embedding_generator = (
            embedding_generator
            or EmbeddingGenerator()
        )

        self.citation_manager = (
            citation_manager
            or CitationManager()
        )

        self.query_rewriter = (
            query_rewriter
            or QueryRewriter()
        )

        self.keyword_index = BM25Index()
        self.refresh_keyword_index()

    def refresh_keyword_index(self) -> None:
        """
        Rebuild BM25 from the vector database.

        Fake databases used in tests may not implement
        get_all_documents, so keyword search is disabled
        gracefully in that case.
        """

        get_all_documents = getattr(
            self.vector_db,
            "get_all_documents",
            None,
        )

        if not callable(get_all_documents):
            self.keyword_index.build(
                ids=[],
                documents=[],
                metadatas=[],
            )
            return

        stored = get_all_documents() or {}

        self.keyword_index.build(
            ids=list(
                stored.get("ids", [])
            ),
            documents=list(
                stored.get("documents", [])
            ),
            metadatas=list(
                stored.get("metadatas", [])
            ),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        rewrite_query: bool = True,
        use_hybrid: bool = True,
        use_multi_query: bool = True,
    ) -> list[RetrievedChunk]:
        cleaned = query.strip()

        if not cleaned:
            raise ValueError(
                "Cannot retrieve results "
                "for an empty query."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be a positive integer."
            )

        if rewrite_query:
            rewritten_result = (
                self.query_rewriter.rewrite(
                    cleaned
                )
            )

            rewritten = (
                rewritten_result.rewritten.strip()
                or cleaned
            )
        else:
            rewritten = cleaned

        multi_query_generator = getattr(
            self.query_rewriter,
            "generate_multi_queries",
            None,
        )

        if (
            use_multi_query
            and callable(multi_query_generator)
        ):
            queries = multi_query_generator(
                rewritten
            )
        else:
            queries = [rewritten]

        queries = list(
            dict.fromkeys(
                query_text.strip()
                for query_text in queries
                if query_text
                and query_text.strip()
            )
        )

        if not queries:
            queries = [rewritten]

        fused: dict[
            str,
            dict[str, Any],
        ] = {}

        keyword_limit = max(
            top_k * 3,
            10,
        )

        for query_index, query_text in enumerate(
            queries
        ):
            semantic_chunks = (
                self._semantic_search(
                    query=query_text,
                    top_k=top_k,
                    where=where,
                )
            )

            self._merge_ranked(
                fused=fused,
                chunks=semantic_chunks,
                source="semantic",
                query_index=query_index,
            )

            if (
                use_hybrid
                and where is None
            ):
                keyword_matches = (
                    self.keyword_index.search(
                        query_text,
                        top_k=keyword_limit,
                    )
                )

                keyword_chunks = [
                    RetrievedChunk(
                        text=match.text,
                        metadata=match.metadata,
                        score=match.score,
                    )
                    for match in keyword_matches
                ]

                self._merge_ranked(
                    fused=fused,
                    chunks=keyword_chunks,
                    source="keyword",
                    query_index=query_index,
                )

        return self._build_results(
            fused=fused,
            top_k=top_k,
        )

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        embedding = (
            self.embedding_generator
            .generate_embedding(query)
        )

        raw_results = self.vector_db.query(
            embedding,
            top_k,
            where,
        )

        return self._parse_results(
            raw_results
        )

    def _merge_ranked(
        self,
        fused: dict[str, dict[str, Any]],
        chunks: list[RetrievedChunk],
        source: str,
        query_index: int,
    ) -> None:
        if source not in {
            "semantic",
            "keyword",
        }:
            raise ValueError(
                f"Unsupported source: {source}"
            )

        source_weight = (
            1.0
            if source == "semantic"
            else 0.75
        )

        query_weight = (
            1.0
            / (
                1.0
                + query_index * 0.15
            )
        )

        for rank, chunk in enumerate(
            chunks,
            start=1,
        ):
            key = self._chunk_key(chunk)

            item = fused.setdefault(
                key,
                {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "fusion": 0.0,
                    "best_semantic": 0.0,
                    "semantic_hits": 0,
                    "keyword_hits": 0,
                },
            )

            item["fusion"] += (
                source_weight
                * query_weight
                / (60 + rank)
            )

            if source == "semantic":
                item["semantic_hits"] += 1

                item["best_semantic"] = max(
                    float(
                        item["best_semantic"]
                    ),
                    self._clamp_score(
                        chunk.score
                    ),
                )
            else:
                item["keyword_hits"] += 1

    def _build_results(
        self,
        fused: dict[str, dict[str, Any]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not fused:
            return []

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                float(item["fusion"]),
                float(
                    item["best_semantic"]
                ),
            ),
            reverse=True,
        )

        max_fusion = max(
            float(item["fusion"])
            for item in ranked
        )

        results: list[RetrievedChunk] = []

        for item in ranked:
            semantic_score = (
                self._clamp_score(
                    item.get(
                        "best_semantic",
                        0.0,
                    )
                )
            )

            normalized_rank_score = (
                float(item["fusion"])
                / max_fusion
                if max_fusion > 0
                else 0.0
            )

            normalized_rank_score = (
                self._clamp_score(
                    normalized_rank_score
                )
            )

            semantic_hits = int(
                item.get(
                    "semantic_hits",
                    0,
                )
            )

            keyword_hits = int(
                item.get(
                    "keyword_hits",
                    0,
                )
            )

            if semantic_hits > 0:
                has_consensus = (
                    semantic_hits > 1
                    or keyword_hits > 0
                )

                bonus = (
                    (
                        1.0 - semantic_score
                    )
                    * 0.15
                    * normalized_rank_score
                    if has_consensus
                    else 0.0
                )

                final_score = (
                    semantic_score
                    + bonus
                )
            else:
                final_score = (
                    normalized_rank_score
                    * 0.55
                )

            final_score = (
                self._clamp_score(
                    final_score
                )
            )

            results.append(
                RetrievedChunk(
                    text=str(
                        item["text"]
                    ),
                    metadata=dict(
                        item.get(
                            "metadata"
                        )
                        or {}
                    ),
                    score=final_score,
                )
            )

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _chunk_key(
        chunk: RetrievedChunk,
    ) -> str:
        metadata = chunk.metadata or {}

        values = [
            metadata.get(
                "document_name"
            ),
            metadata.get(
                "page_number"
            ),
            metadata.get(
                "chunk_number"
            ),
        ]

        has_identifier = any(
            value not in (
                None,
                "",
            )
            for value in values
        )

        if has_identifier:
            return "|".join(
                (
                    ""
                    if value is None
                    else str(value)
                )
                for value in values
            )

        return chunk.text.strip()

    def retrieve_with_citations(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ):
        chunks = self.retrieve(
            query=query,
            top_k=top_k,
            where=where,
        )

        citations = [
            self.citation_manager
            .create_citation(
                chunk.metadata,
                chunk.score,
            )
            for chunk in chunks
        ]

        return chunks, citations

    @staticmethod
    def _parse_results(
        results: dict[str, Any],
    ) -> list[RetrievedChunk]:
        document_batches = (
            results.get("documents")
            or [[]]
        )

        metadata_batches = (
            results.get("metadatas")
            or [[]]
        )

        distance_batches = (
            results.get("distances")
            or [[]]
        )

        documents = (
            document_batches[0]
            if document_batches
            else []
        )

        metadatas = (
            metadata_batches[0]
            if metadata_batches
            else []
        )

        distances = (
            distance_batches[0]
            if distance_batches
            else []
        )

        parsed_results: list[
            RetrievedChunk
        ] = []

        for (
            text,
            metadata,
            distance,
        ) in zip(
            documents,
            metadatas,
            distances,
        ):
            try:
                numeric_distance = float(
                    distance
                )
            except (
                TypeError,
                ValueError,
            ):
                numeric_distance = 1.0

            similarity = (
                RetrievalEngine
                ._clamp_score(
                    1.0
                    - numeric_distance
                )
            )

            parsed_results.append(
                RetrievedChunk(
                    text=str(text),
                    metadata=dict(
                        metadata or {}
                    ),
                    score=similarity,
                )
            )

        return parsed_results

    @staticmethod
    def _clamp_score(
        score: Any,
    ) -> float:
        try:
            numeric_score = float(score)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                numeric_score,
            ),
        )