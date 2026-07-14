from __future__ import annotations

from dataclasses import dataclass  # Used to create a simple class for storing chunk data.
from typing import Any  # Allows metadata values to have different data types.

from src.knowledge_base.loader import LoadedDocument
from src.knowledge_base.metadata import add_chunk_metadata


@dataclass  # Automatically creates the constructor and utility methods.
class DocumentChunk:
    """Stores one text chunk and its metadata."""

    text: str  # Stores the chunk text.
    metadata: dict[str, Any]  # Stores the document, page, and chunk information.


class DocumentChunker:
    """Splits loaded documents into smaller overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        """
        Initialize the chunker settings.

        Args:
            chunk_size: Maximum number of characters in each chunk.
            chunk_overlap: Number of repeated characters between chunks.
        """

        if chunk_size < 1:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size  # Stores the maximum chunk size.
        self.chunk_overlap = chunk_overlap  # Stores the overlap between chunks.

    def chunk_document(
        self,
        document: LoadedDocument,
    ) -> list[DocumentChunk]:
        """
        Split one loaded document into smaller chunks.

        Args:
            document: Loaded document containing text and metadata.

        Returns:
            A list of DocumentChunk objects.
        """

        text = document.text.strip()  # Removes extra spaces around the document text.

        if not text:
            return []  # Returns an empty list if the document has no useful text.

        chunks: list[DocumentChunk] = []
        start = 0  # Marks the beginning position of the current chunk.
        chunk_number = 1  # Starts chunk numbering from 1.

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )  # Calculates the ending position of the current chunk.

            chunk_text = text[start:end].strip()
            # Extracts the current part of the document text.

            if chunk_text:
                chunk_metadata = add_chunk_metadata(
                    document.metadata,
                    chunk_number=chunk_number,
                )
                # Adds the chunk number without changing the original metadata.

                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        metadata=chunk_metadata,
                    )
                )
                # Stores the chunk text and its metadata.

                chunk_number += 1

            if end == len(text):
                break
                # Stops the loop after reaching the end of the document.

            start = end - self.chunk_overlap
            # Moves to the next chunk while keeping some overlapping text.

        return chunks  # Returns all generated chunks.

    def chunk_documents(
        self,
        documents: list[LoadedDocument],
    ) -> list[DocumentChunk]:
        """
        Split multiple loaded documents into chunks.

        Args:
            documents: List of loaded documents.

        Returns:
            A combined list containing all generated chunks.
        """

        all_chunks: list[DocumentChunk] = []

        for document in documents:
            document_chunks = self.chunk_document(document)
            # Splits the current loaded document.

            all_chunks.extend(document_chunks)
            # Adds the generated chunks to the final list.

        return all_chunks


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """
    Helper function for chunking documents in one line.

    Example:
        chunks = chunk_documents(documents)
    """

    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunker.chunk_documents(documents)