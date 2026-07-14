import pytest

from src.embeddings import EmbeddingGenerator
from src.knowledge_base.chunker import DocumentChunk


@pytest.fixture
def generator() -> EmbeddingGenerator:
    # Creates one embedding generator for the tests.
    return EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
    )


def test_generate_single_embedding(
    generator: EmbeddingGenerator,
) -> None:
    # Generates an embedding for one text.
    embedding = generator.generate_embedding(
        "Artificial intelligence is changing the world."
    )

    # Checks that the result is a Python list.
    assert isinstance(embedding, list)

    # Checks that the embedding contains numerical values.
    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)


def test_generate_multiple_embeddings(
    generator: EmbeddingGenerator,
) -> None:
    texts = [
        "Artificial intelligence",
        "Machine learning",
        "Deep learning",
    ]

    embeddings = generator.generate_embeddings(texts)

    # Checks that every text generated one embedding.
    assert len(embeddings) == len(texts)

    # Checks that all embedding vectors have the same size.
    dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    assert len(dimensions) == 1


def test_empty_text_raises_error(
    generator: EmbeddingGenerator,
) -> None:
    # Empty text should not generate an embedding.
    with pytest.raises(
        ValueError,
        match="Cannot generate an embedding for empty text",
    ):
        generator.generate_embedding("   ")


def test_empty_text_list_returns_empty_list(
    generator: EmbeddingGenerator,
) -> None:
    # An empty list should return no embeddings.
    embeddings = generator.generate_embeddings([])

    assert embeddings == []


def test_embed_chunk_preserves_metadata(
    generator: EmbeddingGenerator,
) -> None:
    chunk = DocumentChunk(
        text="Retrieval-Augmented Generation uses external knowledge.",
        metadata={
            "document_name": "rag.pdf",
            "page_number": 3,
            "chunk_number": 1,
        },
    )

    embedded_chunk = generator.embed_chunk(chunk)

    # Checks that the original text is preserved.
    assert embedded_chunk.text == chunk.text

    # Checks that metadata is preserved.
    assert embedded_chunk.metadata["document_name"] == "rag.pdf"
    assert embedded_chunk.metadata["page_number"] == 3
    assert embedded_chunk.metadata["chunk_number"] == 1

    # Checks that an embedding vector was generated.
    assert len(embedded_chunk.embedding) > 0


def test_embed_multiple_chunks(
    generator: EmbeddingGenerator,
) -> None:
    chunks = [
        DocumentChunk(
            text="First chunk text",
            metadata={
                "document_name": "sample.txt",
                "chunk_number": 1,
            },
        ),
        DocumentChunk(
            text="Second chunk text",
            metadata={
                "document_name": "sample.txt",
                "chunk_number": 2,
            },
        ),
    ]

    embedded_chunks = generator.embed_chunks(chunks)

    # Checks that every valid chunk generated one embedded chunk.
    assert len(embedded_chunks) == 2

    # Checks that chunk order is preserved.
    assert embedded_chunks[0].metadata["chunk_number"] == 1
    assert embedded_chunks[1].metadata["chunk_number"] == 2


def test_invalid_model_name() -> None:
    # The model name cannot be empty.
    with pytest.raises(
        ValueError,
        match="model_name cannot be empty",
    ):
        EmbeddingGenerator(model_name="   ")