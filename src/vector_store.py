from __future__ import annotations

import uuid

import chromadb

from src.embeddings import EmbeddedChunk


class VectorDB:
    """
    Manage storage and semantic search using ChromaDB.
    """

    def __init__(
        self,
        persist_path: str = "./chroma_db",
        collection_name: str = "knowledge_base",
    ) -> None:
        """
        Initialize the persistent ChromaDB collection.
        """

        self.client = chromadb.PersistentClient(
            path=persist_path
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
            },
        )

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """
        Add new documents or update existing documents.

        Using upsert prevents duplicate-ID errors when
        the indexing process is executed more than once.
        """

        if not ids:
            return

        if not (
            len(ids)
            == len(embeddings)
            == len(documents)
            == len(metadatas)
        ):
            raise ValueError(
                "IDs, embeddings, documents, and metadata "
                "must have the same length."
            )

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def add_embedded_chunks(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:
        """
        Store embedded chunks in ChromaDB.
        """

        if not chunks:
            return

        ids = [
            self._build_chunk_id(
                chunk,
                index,
            )
            for index, chunk in enumerate(chunks)
        ]

        embeddings = [
            chunk.embedding
            for chunk in chunks
        ]

        documents = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            chunk.metadata
            for chunk in chunks
        ]

        self.add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def _build_chunk_id(
        self,
        chunk: EmbeddedChunk,
        index: int,
    ) -> str:
        """
        Build a unique and stable ID for every chunk.

        Page number is included because chunk numbering may
        restart on every PDF page.
        """

        document_name = str(
            chunk.metadata.get(
                "document_name",
                "unknown_document",
            )
        )

        page_number = chunk.metadata.get(
            "page_number"
        )

        chunk_number = chunk.metadata.get(
            "chunk_number",
            index + 1,
        )

        safe_document_name = (
            document_name
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        page_part = (
            f"page_{page_number}"
            if page_number is not None
            else "no_page"
        )

        if not document_name:
            return str(uuid.uuid4())

        return (
            f"{safe_document_name}_"
            f"{page_part}_"
            f"chunk_{chunk_number}"
        )

    def count(self) -> int:
        """
        Return the number of chunks stored in the collection.
        """

        return self.collection.count()

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        """
        Search for the chunks most similar to a query embedding.
        """

        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        stored_count = self.count()

        if stored_count == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        safe_n_results = min(
            n_results,
            stored_count,
        )

        query_arguments = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": safe_n_results,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where:
            query_arguments["where"] = where

        return self.collection.query(
            **query_arguments
        )

    def get_all_documents(self) -> dict:
        """
        Retrieve all stored documents and metadata.
        """

        return self.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

    def delete_document(
        self,
        document_name: str,
    ) -> None:
        """
        Delete all chunks belonging to one document.
        """

        if not document_name.strip():
            raise ValueError(
                "Document name cannot be empty."
            )

        self.collection.delete(
            where={
                "document_name": document_name,
            }
        )

    def reset(self) -> None:
        """
        Delete all chunks from the current collection.
        """

        existing = self.collection.get()

        ids = existing.get(
            "ids",
            [],
        )

        if ids:
            self.collection.delete(
                ids=ids
            )