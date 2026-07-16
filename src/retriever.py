from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorDB
from src.citations import Citation, CitationManager
from src.query_rewriter import QueryRewriter
from src.keyword_search import BM25Index


@dataclass
class RetrievedChunk:
    """Stores a single retrieved chunk together with its relevance score."""

    text: str
    metadata: dict[str, Any]
    score: float


class RetrievalEngine:
    """Finds the most relevant document chunks for a given query."""

    def __init__(
        self,
        vector_db: VectorDB,
        embedding_generator: EmbeddingGenerator | None = None,
        citation_manager: CitationManager | None = None,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        """
        Args:
            vector_db: The VectorDB instance to search against.
            embedding_generator: Generator used to embed the query. Created
                automatically if not provided, so it stays consistent with
                how document embeddings were generated.
            citation_manager: Used to build citations from retrieved chunks.
                Created automatically if not provided.
            query_rewriter: Optimizes the raw query before embedding, per
                AKA-12. Created automatically if not provided.
        """

        self.vector_db = vector_db
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.citation_manager = citation_manager or CitationManager()
        self.query_rewriter = query_rewriter or QueryRewriter()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Args:
            query: The user's natural language question.
            top_k: Number of chunks to return.
            where: Optional metadata filter, e.g. {"document_name": "file.pdf"}.

        Returns:
            A list of RetrievedChunk objects, most relevant first.
        """

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("Cannot retrieve results for an empty query.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        rewritten = self.query_rewriter.rewrite(cleaned_query)
        # Optimizes the query before embedding, per AKA-12 — e.g. "can you
        # tell me about the faq" becomes "About the frequently asked questions".

        query_embedding = self.embedding_generator.generate_embedding(
            rewritten.rewritten)

        results = self.vector_db.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        return self._parse_results(results)

    def retrieve_with_citations(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievedChunk], list[Citation]]:
        """
        Retrieve chunks and build a matching Citation for each one.

        Args:
            query: The user's natural language question.
            top_k: Number of chunks to return.
            where: Optional metadata filter.

        Returns:
            A tuple of (chunks, citations) — citations[i] corresponds to
            chunks[i], built from that chunk's metadata and similarity score.
        """

        chunks = self.retrieve(query, top_k=top_k, where=where)

        citations = [
            self.citation_manager.create_citation(
                metadata=chunk.metadata,
                similarity_score=chunk.score,
            )
            for chunk in chunks
        ]
        # Builds one Citation per chunk, reusing the chunk's own similarity
        # score as the confidence source — no re-fetching or re-scoring needed.

        return chunks, citations

    def _parse_results(self, results: dict[str, Any]) -> list[RetrievedChunk]:
        """Converts Chroma's raw query response into RetrievedChunk objects."""

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        # Chroma wraps each field in an outer list (one per query); [0] unwraps our single query.

        retrieved_chunks: list[RetrievedChunk] = []

        for text, metadata, distance in zip(documents, metadatas, distances):
            # Converts cosine distance to a similarity score.
            similarity = 1 - distance
            retrieved_chunks.append(
                RetrievedChunk(text=text, metadata=metadata, score=similarity)
            )

        return retrieved_chunks
