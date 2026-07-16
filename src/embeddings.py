from __future__ import annotations

from dataclasses import dataclass  # Creates a simple class for storing chunk data and its embedding.
from typing import Any  # Allows metadata values to have different data types.

from sentence_transformers import SentenceTransformer  # Converts text into numerical vectors.

from src.knowledge_base.chunker import DocumentChunk


@dataclass
class EmbeddedChunk:
    """Stores a document chunk together with its embedding vector."""

    text: str  # Stores the original chunk text.
    embedding: list[float]  # Stores the numerical vector generated from the text.
    metadata: dict[str, Any]  # Stores document, page, and chunk information.


class EmbeddingGenerator:
    """Generates embeddings for document chunks."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Load the embedding model.

        Args:
            model_name: Name of the Sentence Transformer model.
        """

        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        self.model_name = model_name

        # Loads the embedding model.
        # The model may be downloaded automatically during the first run.
        self.model = SentenceTransformer(model_name)

    def generate_embedding(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate one embedding vector from a text.

        Args:
            text: Text that will be converted into a vector.

        Returns:
            A list of floating-point numbers representing the text.
        """

        cleaned_text = text.strip()  # Removes unnecessary spaces around the text.

        if not cleaned_text:
            raise ValueError("Cannot generate an embedding for empty text.")

        embedding = self.model.encode(
            cleaned_text,
            normalize_embeddings=True,
        )
        # Converts the text into a normalized numerical vector.

        return embedding.tolist()
        # Converts the NumPy array into a normal Python list.

    def generate_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts at once.

        Args:
            texts: A list of texts.

        Returns:
            A list containing one embedding vector for each text.
        """

        if not texts:
            return []

        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]
        # Removes empty texts and unnecessary spaces.

        if not cleaned_texts:
            return []

        embeddings = self.model.encode(
            cleaned_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Generates embeddings for all texts in one batch.

        return embeddings.tolist()

    def embed_chunk(
        self,
        chunk: DocumentChunk,
    ) -> EmbeddedChunk:
        """
        Generate an embedding for one document chunk.

        Args:
            chunk: Document chunk containing text and metadata.

        Returns:
            An EmbeddedChunk containing the original data and its vector.
        """

        embedding = self.generate_embedding(chunk.text)

        return EmbeddedChunk(
            text=chunk.text,
            embedding=embedding,
            metadata=chunk.metadata.copy(),
        )
        # Returns the chunk text, embedding, and a copy of its metadata.

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for multiple document chunks.

        Args:
            chunks: List of document chunks.

        Returns:
            A list of EmbeddedChunk objects.
        """

        if not chunks:
            return []

        valid_chunks = [
            chunk
            for chunk in chunks
            if chunk.text and chunk.text.strip()
        ]
        # Removes chunks that contain no useful text.

        if not valid_chunks:
            return []

        texts = [
            chunk.text
            for chunk in valid_chunks
        ]

        embeddings = self.generate_embeddings(texts)

        embedded_chunks: list[EmbeddedChunk] = []

        for chunk, embedding in zip(
            valid_chunks,
            embeddings,
        ):
            embedded_chunks.append(
                EmbeddedChunk(
                    text=chunk.text,
                    embedding=embedding,
                    metadata=chunk.metadata.copy(),
                )
            )
            # Connects every chunk with its generated embedding.

        return embedded_chunks


def embed_chunks(
    chunks: list[DocumentChunk],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[EmbeddedChunk]:
    """
    Helper function for generating embeddings in one line.

    Example:
        embedded_chunks = embed_chunks(chunks)
    """

    generator = EmbeddingGenerator(
        model_name=model_name,
    )

    return generator.embed_chunks(chunks)