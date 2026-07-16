from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retriever import RetrievedChunk
from src.citations import Citation, CitationManager


@dataclass
class ContextResult:
    """Stores the assembled context ready for the LLM prompt."""

    context: str  # The final formatted context string.
    # One Citation per chunk used in the context.
    citations: list[Citation] = field(default_factory=list)
    chunks_used: int = 0  # How many chunks made it into the final context.
    # How many were dropped (low score, duplicates, or over budget).
    chunks_dropped: int = 0


class ContextBuilder:
    """Builds an optimized context string from retrieved chunks before sending it to Gemini."""

    def __init__(
        self,
        max_tokens: int = 3000,
        chars_per_token: float = 4.0,
        citation_manager: CitationManager | None = None,
    ) -> None:
        """
        Args:
            max_tokens: Approximate max tokens the context may use.
            chars_per_token: Rough chars-to-token ratio (no real Gemini
                tokenizer wired in yet; ~4 chars/token is a common estimate
                for English).
            citation_manager: Used to build citations for the chunks that
                end up in the final context. Created automatically if not
                provided, so callers don't need to wire it up manually.
        """

        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be a positive number.")

        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self.citation_manager = citation_manager or CitationManager()

    def build_context(
        self,
        chunks: list[RetrievedChunk],
        min_score: float = 0.0,
    ) -> ContextResult:
        """
        Assemble retrieved chunks into a single optimized context string.

        Args:
            chunks: Chunks from RetrievalEngine.retrieve(), assumed sorted
                highest score first.
            min_score: Chunks below this similarity score are dropped.

        Returns:
            A ContextResult with the formatted context, citations, and counts.
        """

        if not chunks:
            return ContextResult(context="", citations=[], chunks_used=0, chunks_dropped=0)

        filtered = [c for c in chunks if c.score >= min_score]
        deduped = self._deduplicate(filtered)
        selected, dropped_by_budget = self._apply_token_budget(deduped)

        total_dropped = (len(chunks) - len(filtered)) + \
            (len(filtered) - len(deduped)) + dropped_by_budget

        citations = [
            self.citation_manager.create_citation(
                metadata=chunk.metadata,
                similarity_score=chunk.score,
            )
            for chunk in selected
        ]
        # Builds one Citation per selected chunk, reusing citations.py's
        # existing logic instead of duplicating source-label formatting here.

        return ContextResult(
            context=self._format_context(selected, citations),
            citations=citations,
            chunks_used=len(selected),
            chunks_dropped=total_dropped,
        )

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Removes chunks with identical text, keeping the first (highest-scoring) occurrence."""

        seen: set[str] = set()
        unique: list[RetrievedChunk] = []

        for chunk in chunks:
            normalized = chunk.text.strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(chunk)

        return unique

    def _apply_token_budget(self, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], int]:
        """Keeps adding chunks, highest relevance first, until the token budget would be exceeded."""

        selected: list[RetrievedChunk] = []
        used_tokens = 0

        for chunk in chunks:
            estimated = self._estimate_tokens(chunk.text)
            if used_tokens + estimated > self.max_tokens:
                break
            selected.append(chunk)
            used_tokens += estimated

        return selected, len(chunks) - len(selected)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate based on character length."""
        return int(len(text) / self.chars_per_token)

    def _format_context(self, chunks: list[RetrievedChunk], citations: list[Citation]) -> str:
        """
        Formats selected chunks into a numbered, source-labeled string for
        the prompt, using citations.py's Citation objects for source labels
        instead of reading raw metadata directly.
        """

        if not chunks:
            return ""

        sections = [
            f"[{i}] {self._build_inline_label(citation)}\n{chunk.text.strip()}"
            for i, (chunk, citation) in enumerate(zip(chunks, citations), start=1)
        ]

        return "\n\n".join(sections)

    def _build_inline_label(self, citation: Citation) -> str:
        """
        Builds a short source label for inline use in the context string
        (no confidence score here — that's for the final answer's citation
        list via CitationManager.format_citation, not the raw context fed
        into the model).
        """

        label = f"Source: {citation.document_name}"

        if citation.page_number is not None:
            label += f" (page {citation.page_number})"

        return label
