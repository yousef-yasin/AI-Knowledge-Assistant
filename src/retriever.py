from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.citations import Citation, CitationManager
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
    """Semantic, keyword, hybrid, and multi-query retrieval."""

    def __init__(self, vector_db: VectorDB,
                 embedding_generator: EmbeddingGenerator | None = None,
                 citation_manager: CitationManager | None = None,
                 query_rewriter: QueryRewriter | None = None) -> None:
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.citation_manager = citation_manager or CitationManager()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.keyword_index = BM25Index()
        self.refresh_keyword_index()

    def refresh_keyword_index(self) -> None:
        stored = self.vector_db.get_all_documents()
        self.keyword_index.build(
            ids=list(stored.get("ids", [])),
            documents=list(stored.get("documents", [])),
            metadatas=list(stored.get("metadatas", [])),
        )

    def retrieve(self, query: str, top_k: int = 5,
                 where: dict[str, Any] | None = None,
                 rewrite_query: bool = True,
                 use_hybrid: bool = True,
                 use_multi_query: bool = True) -> list[RetrievedChunk]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Cannot retrieve results for an empty query.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        rewritten = self.query_rewriter.rewrite(cleaned).rewritten if rewrite_query else cleaned
        queries = (
            self.query_rewriter.generate_multi_queries(rewritten)
            if use_multi_query else [rewritten]
        )
        fused: dict[str, dict[str, Any]] = {}
        for query_index, query_text in enumerate(queries):
            semantic = self._semantic_search(query_text, max(top_k * 3, 10), where)
            self._merge_ranked(fused, semantic, source="semantic", query_index=query_index)
            if use_hybrid and where is None:
                keyword = self.keyword_index.search(query_text, top_k=max(top_k * 3, 10))
                keyword_chunks = [RetrievedChunk(m.text, m.metadata, m.score) for m in keyword]
                self._merge_ranked(fused, keyword_chunks, source="keyword", query_index=query_index)

        ranked = sorted(fused.values(), key=lambda item: item["fusion"], reverse=True)
        results: list[RetrievedChunk] = []
        for item in ranked:
            semantic_score = item.get("best_semantic", 0.0)
            rank_score = min(item["fusion"] * 4.0, 1.0)
            final_score = max(0.0, min(1.0, semantic_score * 0.75 + rank_score * 0.25))
            results.append(RetrievedChunk(item["text"], item["metadata"], final_score))
            if len(results) >= top_k:
                break
        return results

    def _semantic_search(self, query: str, top_k: int,
                         where: dict[str, Any] | None) -> list[RetrievedChunk]:
        embedding = self.embedding_generator.generate_embedding(query)
        return self._parse_results(self.vector_db.query(embedding, top_k, where))

    def _merge_ranked(self, fused: dict[str, dict[str, Any]],
                      chunks: list[RetrievedChunk], source: str,
                      query_index: int) -> None:
        # Reciprocal Rank Fusion is robust across incompatible score scales.
        for rank, chunk in enumerate(chunks, start=1):
            key = self._chunk_key(chunk)
            item = fused.setdefault(key, {
                "text": chunk.text, "metadata": chunk.metadata,
                "fusion": 0.0, "best_semantic": 0.0,
            })
            weight = 1.0 if source == "semantic" else 0.75
            query_weight = 1.0 / (1.0 + query_index * 0.15)
            item["fusion"] += weight * query_weight / (60 + rank)
            if source == "semantic":
                item["best_semantic"] = max(item["best_semantic"], chunk.score)

    @staticmethod
    def _chunk_key(chunk: RetrievedChunk) -> str:
        metadata = chunk.metadata
        return "|".join(str(metadata.get(key, "")) for key in
                        ("document_name", "page_number", "chunk_number")) or chunk.text

    def retrieve_with_citations(self, query: str, top_k: int = 5,
                                where: dict[str, Any] | None = None):
        chunks = self.retrieve(query, top_k=top_k, where=where)
        citations = [self.citation_manager.create_citation(c.metadata, c.score) for c in chunks]
        return chunks, citations

    @staticmethod
    def _parse_results(results: dict[str, Any]) -> list[RetrievedChunk]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [RetrievedChunk(text, metadata or {}, max(0.0, min(1.0, 1 - distance)))
                for text, metadata, distance in zip(documents, metadatas, distances)]
