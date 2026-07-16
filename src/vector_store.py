from __future__ import annotations

# Used to generate a fallback unique ID when metadata is incomplete.
import uuid

import chromadb  # to import the DB

from src.embeddings import EmbeddedChunk


class VectorDB:
    def __init__(
        self,
        # initialize the database, persist_path is the path where the database will be stored
        persist_path: str = "./chroma_db",
        # collection_name to specify the name of the collection in the database
        collection_name: str = "knowledge_base",
    ):
        # actual connection to the database, using PersistentClient to ensure data is saved to disk
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(  # if the collection with the specified name does not exist, it will be created
            name=collection_name,
            # metadata to specify the distance metric used for similarity search, in this case, cosine similarity
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def add_embedded_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """
        Store a list of embedded chunks directly, without manually unpacking
        ids, embeddings, documents, and metadatas yourself.

        Args:
            chunks: Embedded chunks produced by EmbeddingGenerator.
        """

        if not chunks:
            return

        ids = [self._build_chunk_id(chunk, index)
               for index, chunk in enumerate(chunks)]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def _build_chunk_id(self, chunk: EmbeddedChunk, index: int) -> str:
        """
        Builds a stable, readable ID from chunk metadata, using the real
        keys produced by metadata.py: 'document_name' and 'chunk_number'.
        Falls back to a random UUID if either is missing.
        """

        document_name = chunk.metadata.get("document_name")
        chunk_number = chunk.metadata.get("chunk_number", index)

        if document_name is not None:
            return f"{document_name}_{chunk_number}"

        return str(uuid.uuid4())

    def count(self) -> int:  # returns the number of documents in the collection
        return self.collection.count()

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        """
        Search the collection for the most similar embeddings.

        Args:
            query_embedding: The embedding vector to search against.
            n_results: How many results to return.
            where: Optional metadata filter, e.g. {"document_name": "file.pdf"}.

        Returns:
            Chroma's raw query response (documents, metadatas, distances).
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
