import pytest

from src.citations import (
    Citation,
    CitationManager,
    format_answer_with_citations,
)


def test_calculate_confidence() -> None:
    # Creates a citation manager.
    manager = CitationManager()

    # Calculates the confidence score.
    confidence = manager.calculate_confidence(0.85)

    # Checks that the confidence score is converted correctly.
    assert confidence == 85.0


def test_confidence_is_limited_to_valid_range() -> None:
    # Creates a citation manager.
    manager = CitationManager()

    # Checks that values greater than 1 become 100%.
    assert manager.calculate_confidence(1.5) == 100.0

    # Checks that negative values become 0%.
    assert manager.calculate_confidence(-0.5) == 0.0


def test_create_citation_preserves_metadata() -> None:
    # Creates a citation manager.
    manager = CitationManager()

    # Creates one citation from metadata.
    citation = manager.create_citation(
        metadata={
            "document_name": "book.pdf",
            "page_number": 7,
            "chunk_number": 2,
        },
        similarity_score=0.92,
    )

    # Checks that all metadata values are preserved.
    assert citation.document_name == "book.pdf"
    assert citation.page_number == 7
    assert citation.chunk_number == 2

    # Checks that the confidence score is correct.
    assert citation.confidence_score == 92.0


def test_calculate_overall_confidence() -> None:
    # Creates sample citations.
    citations = [
        Citation(
            document_name="first.pdf",
            page_number=1,
            chunk_number=1,
            confidence_score=80.0,
        ),
        Citation(
            document_name="second.pdf",
            page_number=2,
            chunk_number=2,
            confidence_score=90.0,
        ),
    ]

    # Calculates rank-weighted retrieval confidence.
    confidence = CitationManager.calculate_overall_confidence(
        citations
    )

    # The first source receives twice the weight of the second.
    assert confidence == 83.33


def test_format_citation() -> None:
    # Creates one citation.
    citation = Citation(
        document_name="report.pdf",
        page_number=4,
        chunk_number=3,
        confidence_score=88.5,
    )

    # Formats the citation into readable text.
    formatted = CitationManager.format_citation(citation)

    # Checks that all citation information appears.
    assert "report.pdf" in formatted
    assert "Page: 4" in formatted
    assert "Chunk: 3" in formatted
    assert "Confidence: 88.50%" in formatted


def test_format_response_with_sources() -> None:
    # Creates a citation manager.
    manager = CitationManager()

    # Creates sample citations.
    citations = [
        Citation(
            document_name="book.pdf",
            page_number=5,
            chunk_number=1,
            confidence_score=90.0,
        )
    ]

    # Generates the final response.
    response = manager.format_response(
        answer="Machine learning is a subset of AI.",
        citations=citations,
    )

    # Checks that the answer contains the expected information.
    assert "Machine learning is a subset of AI." in response
    assert "Sources:" in response
    assert "book.pdf" in response
    assert "Overall Confidence: 90.00%" in response


def test_format_response_without_sources() -> None:
    # Creates a citation manager.
    manager = CitationManager()

    # Generates a response without citations.
    response = manager.format_response(
        answer="No supported information was found.",
        citations=[],
    )

    # Checks that the default message is returned.
    assert "No supporting sources were found" in response
    assert "Overall Confidence: 0.00%" in response


def test_empty_answer_raises_error() -> None:
    # Creates a citation manager.
    manager = CitationManager()

    # Checks that an empty answer raises an error.
    with pytest.raises(
        ValueError,
        match="answer cannot be empty",
    ):
        manager.format_response(
            answer="   ",
            citations=[],
        )


def test_helper_function() -> None:
    # Uses the helper function to format an answer.
    result = format_answer_with_citations(
        answer="Artificial intelligence is a broad field.",
        retrieved_results=[
            {
                "metadata": {
                    "document_name": "ai.pdf",
                    "page_number": 3,
                    "chunk_number": 1,
                },
                "similarity_score": 0.95,
            }
        ],
    )

    # Checks that the helper function returns the expected output.
    assert "ai.pdf" in result
    assert "95.00%" in result

def test_calculate_answer_confidence_with_passed_validation() -> None:
    citations = [
        Citation("first.pdf", 1, 1, 80.0),
        Citation("second.pdf", 2, 2, 60.0),
    ]

    confidence = CitationManager.calculate_answer_confidence(
        citations=citations,
        validation_supported=True,
    )

    # Rank-weighted retrieval is 73.33%; combined with a PASS validator.
    assert confidence == 84.0


def test_calculate_answer_confidence_with_failed_validation() -> None:
    citations = [Citation("first.pdf", 1, 1, 80.0)]

    confidence = CitationManager.calculate_answer_confidence(
        citations=citations,
        validation_supported=False,
    )

    assert confidence == 48.0
