from __future__ import annotations

# Creates a simple class for storing a retrieved result.
from dataclasses import dataclass  # auto generate _init_, _repr_
from typing import Any  # Allows metadata values to have different data types.

from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorDB


@dataclass
class RetrievedChunk:
    """Stores a single retrieved chunk together with its relevance score."""

    text: str  # The original chunk text.
    metadata: dict[str, Any]  # Document, page, and chunk info.
    score: float  # Similarity score; higher means more relevant.


class RetrievalEngine:
    """Finds the most relevant document chunks for a given query."""

    def __init__(
        self,
        vector_db: VectorDB,  # passes the DB we Built
        # optional, if not provided, it creates a default generator
        embedding_generator: EmbeddingGenerator | None = None,
    ) -> None:
        """
        Initialize the retrieval engine.

        Args:
            vector_db: The VectorDB instance to search against.
            embedding_generator: Generator used to embed the query. Created
                automatically with the default model if not provided, so it
                stays consistent with how document embeddings were generated.
        """

        self.vector_db = vector_db
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def retrieve(
        self,
        query: str,  # the user's question
        top_k: int = 5,  # how many results to return
        where: dict[str, Any] | None = None,  # optional filter
    ) -> list[RetrievedChunk]:
        """
        Returns:
            A list of RetrievedChunk objects, most relevant first.
        """

        # Removes unnecessary spaces around the query.
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("Cannot retrieve results for an empty query.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_embedding = self.embedding_generator.generate_embedding(
            cleaned_query)
        # Embeds the query using the same model as the documents, so vectors are comparable.

        results = self.vector_db.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        return self._parse_results(results)

    def _parse_results(
        self,
        results: dict[str, Any],
    ) -> list[RetrievedChunk]:
        """
        Convert Chroma's raw query response into a list of RetrievedChunk.

        Chroma returns cosine distance (lower = more similar). We convert
        this to a similarity score (higher = more relevant) to match
        typical retrieval conventions.
        """

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        # Chroma wraps each field in an outer list (one per query); [0] unwraps our single query.

        retrieved_chunks: list[RetrievedChunk] = []

        for text, metadata, distance in zip(documents, metadatas, distances):
            # Converts cosine distance to a similarity score.
            similarity = 1 - distance
            retrieved_chunks.append(
                RetrievedChunk(
                    text=text,
                    metadata=metadata,
                    score=similarity,
                )
            )

        return retrieved_chunks
