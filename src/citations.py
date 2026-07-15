from __future__ import annotations

from dataclasses import dataclass  # Creates a simple class for storing citation data.
from typing import Any  # Allows metadata values to have different data types.


@dataclass  # Automatically creates the constructor and utility methods.
class Citation:
    """Stores the source information for one retrieved chunk."""

    document_name: str  # Stores the source document name.
    page_number: int | None  # Stores the page number if available.
    chunk_number: int | None  # Stores the chunk number if available.
    confidence_score: float  # Stores the confidence score as a percentage.


class CitationManager:
    """Creates citations and confidence scores from retrieved results."""

    @staticmethod
    def calculate_confidence(similarity_score: float) -> float:
        """
        Convert a similarity score into a percentage.

        Args:
            similarity_score:
                A similarity value between 0 and 1.

        Returns:
            A confidence percentage between 0 and 100.
        """

        if not isinstance(similarity_score, (int, float)):
            raise TypeError("similarity_score must be a number.")

        normalized_score = max(
            0.0,
            min(float(similarity_score), 1.0),
        )
        # Keeps the similarity score within the valid range.

        return round(
            normalized_score * 100,
            2,
        )
        # Converts the similarity score into a percentage.

    def create_citation(
        self,
        metadata: dict[str, Any],
        similarity_score: float,
    ) -> Citation:
        """
        Create one Citation object from metadata and a similarity score.

        Args:
            metadata:
                Information about the retrieved document chunk.

            similarity_score:
                Similarity score returned by the retrieval system.

        Returns:
            A Citation object.
        """

        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        document_name = str(
            metadata.get("document_name", "Unknown document")
        )
        # Retrieves the document name or uses a default value.

        page_number = metadata.get("page_number")
        # Retrieves the page number if available.

        chunk_number = metadata.get("chunk_number")
        # Retrieves the chunk number if available.

        confidence_score = self.calculate_confidence(
            similarity_score
        )
        # Calculates the confidence score.

        return Citation(
            document_name=document_name,
            page_number=page_number,
            chunk_number=chunk_number,
            confidence_score=confidence_score,
        )
        # Returns a Citation object containing all source information.

    def create_citations(
        self,
        retrieved_results: list[dict[str, Any]],
    ) -> list[Citation]:
        """
        Create citations for multiple retrieved results.

        Each result should contain:
        - metadata
        - similarity_score

        Args:
            retrieved_results:
                Results returned by the retrieval system.

        Returns:
            A list of Citation objects.
        """

        citations: list[Citation] = []
        # Stores all generated citations.

        for result in retrieved_results:
            metadata = result.get("metadata", {})
            similarity_score = result.get(
                "similarity_score",
                0.0,
            )

            citation = self.create_citation(
                metadata=metadata,
                similarity_score=similarity_score,
            )
            # Creates a citation for the current retrieved result.

            citations.append(citation)
            # Adds the generated citation to the final list.

        return citations

    @staticmethod
    def calculate_overall_confidence(
        citations: list[Citation],
    ) -> float:
        """
        Calculate the average confidence score for all citations.

        Args:
            citations:
                List of generated citations.

        Returns:
            Average confidence percentage.
        """

        if not citations:
            return 0.0

        total_score = sum(
            citation.confidence_score
            for citation in citations
        )
        # Calculates the total confidence score.

        return round(
            total_score / len(citations),
            2,
        )
        # Returns the average confidence percentage.

    @staticmethod
    def format_citation(citation: Citation) -> str:
        """
        Convert one citation into readable text.

        Args:
            citation:
                Citation object that will be displayed.

        Returns:
            A formatted citation string.
        """

        citation_parts = [
            f"Document: {citation.document_name}"
        ]
        # Starts building the citation text.

        if citation.page_number is not None:
            citation_parts.append(
                f"Page: {citation.page_number}"
            )
            # Adds the page number if available.

        if citation.chunk_number is not None:
            citation_parts.append(
                f"Chunk: {citation.chunk_number}"
            )
            # Adds the chunk number if available.

        citation_parts.append(
            f"Confidence: {citation.confidence_score:.2f}%"
        )
        # Adds the confidence percentage.

        return " | ".join(citation_parts)
        # Combines all citation parts into one readable string.

    def format_response(
        self,
        answer: str,
        citations: list[Citation],
    ) -> str:
        """
        Add citations and confidence information to an answer.

        Args:
            answer:
                The generated AI response.

            citations:
                Sources supporting the response.

        Returns:
            The final formatted response.
        """

        cleaned_answer = answer.strip()
        # Removes unnecessary spaces from the answer.

        if not cleaned_answer:
            raise ValueError("answer cannot be empty.")

        if not citations:
            return (
                f"{cleaned_answer}\n\n"
                "Sources: No supporting sources were found.\n"
                "Overall Confidence: 0.00%"
            )
            # Returns the answer if no supporting sources are available.

        formatted_citations = [
            f"{index}. {self.format_citation(citation)}"
            for index, citation in enumerate(
                citations,
                start=1,
            )
        ]
        # Formats all citations into readable text.

        overall_confidence = self.calculate_overall_confidence(
            citations
        )
        # Calculates the average confidence score.

        return (
            f"{cleaned_answer}\n\n"
            "Sources:\n"
            + "\n".join(formatted_citations)
            + f"\n\nOverall Confidence: {overall_confidence:.2f}%"
        )
        # Returns the final response with citations and confidence score.


def format_answer_with_citations(
    answer: str,
    retrieved_results: list[dict[str, Any]],
) -> str:
    """
    Helper function for formatting an answer with citations in one line.

    Example:
        final_response = format_answer_with_citations(
            answer,
            retrieved_results,
        )
    """

    manager = CitationManager()
    # Creates the citation manager.

    citations = manager.create_citations(
        retrieved_results
    )
    # Generates citations from the retrieved results.

    return manager.format_response(
        answer=answer,
        citations=citations,
    )
    # Returns the final formatted answer with citations.