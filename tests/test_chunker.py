from pathlib import Path

import pytest

from src.knowledge_base.chunker import DocumentChunker, chunk_documents
from src.knowledge_base.loader import LoadedDocument


def test_chunk_document_splits_text() -> None:
    # Creates a sample loaded document with text and metadata.
    document = LoadedDocument(
        text="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        metadata={
            "document_name": "sample.txt",
            "page_number": None,
        },
    )

    # Creates a chunker with small values to make the test easy to verify.
    chunker = DocumentChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    chunks = chunker.chunk_document(document)

    # Checks that the text was split into more than one chunk.
    assert len(chunks) > 1

    # Checks that the first chunk contains the expected text.
    assert chunks[0].text == "ABCDEFGHIJ"

    # Checks that the second chunk starts with the overlapping characters.
    assert chunks[1].text.startswith("IJ")

    # Checks that chunk numbering starts from 1.
    assert chunks[0].metadata["chunk_number"] == 1

    # Checks that the original document metadata is preserved.
    assert chunks[0].metadata["document_name"] == "sample.txt"


def test_empty_document_returns_empty_list() -> None:
    # Creates a document containing only empty spaces.
    document = LoadedDocument(
        text="   ",
        metadata={
            "document_name": "empty.txt",
            "page_number": None,
        },
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(document)

    # Empty documents should not generate chunks.
    assert chunks == []


def test_chunk_documents_combines_results() -> None:
    # Creates two loaded documents.
    documents = [
        LoadedDocument(
            text="First document text",
            metadata={
                "document_name": "first.txt",
                "page_number": None,
            },
        ),
        LoadedDocument(
            text="Second document text",
            metadata={
                "document_name": "second.txt",
                "page_number": None,
            },
        ),
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=10,
        chunk_overlap=2,
    )

    # Checks that chunks were generated for both documents.
    document_names = {
        chunk.metadata["document_name"]
        for chunk in chunks
    }

    assert "first.txt" in document_names
    assert "second.txt" in document_names


def test_invalid_chunk_size() -> None:
    # chunk_size must be greater than zero.
    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than 0",
    ):
        DocumentChunker(
            chunk_size=0,
            chunk_overlap=0,
        )


def test_negative_chunk_overlap() -> None:
    # chunk_overlap cannot be negative.
    with pytest.raises(
        ValueError,
        match="chunk_overlap cannot be negative",
    ):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    # chunk_overlap must always be smaller than chunk_size.
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=100,
        )