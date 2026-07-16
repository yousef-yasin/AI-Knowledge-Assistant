
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retriever import RetrievedChunk
# purpose of this class is to take the output of the retriever and transform it


@dataclass
class ContextResult:
    """Stores the assembled context ready for the LLM prompt."""

    context: str  # The final formatted context string.
    # Deduplicated source metadata used in the context.
    sources: list[dict[str, Any]] = field(default_factory=list)
    chunks_used: int = 0  # How many chunks made it into the final context.
    # How many were dropped (low score, duplicates, or over budget).
    chunks_dropped: int = 0


class ContextBuilder:
    """Builds an optimized context string from retrieved chunks before sending it to Gemini."""

    def __init__(
        self,
        max_tokens: int = 3000,  # how much text gets sent into gemini
        chars_per_token: float = 4.0,
    ) -> None:

        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be a positive number.")

        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token

    def build_context(
        self,
        chunks: list[RetrievedChunk],
        min_score: float = 0.0,
    ) -> ContextResult:

        if not chunks:
            # if the retriver finds nothing
            return ContextResult(context="", sources=[], chunks_used=0, chunks_dropped=0)
            # returns an empty result instead of error

        # drops any chunk whose smiliarity score is below min_score
        filtered = [c for c in chunks if c.score >= min_score]
        deduped = self._deduplicate(filtered)  # removes duplicates first
        selected, dropped_by_budget = self._apply_token_budget(
            deduped)  # fit what's left into token budget

        total_dropped = (len(chunks) - len(filtered)) + \
            (len(filtered) - len(deduped)) + \
            dropped_by_budget  # arithmetic sum

        return ContextResult(
            context=self._format_context(selected),
            sources=self._extract_sources(selected),
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

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Formats selected chunks into a numbered, source-labeled string for the prompt."""

        if not chunks:
            return ""

        sections = [
            f"[{i}] Source: {self._build_source_label(chunk.metadata)}\n{chunk.text.strip()}"
            for i, chunk in enumerate(chunks, start=1)
        ]

        return "\n\n".join(sections)

    def _build_source_label(self, metadata: dict[str, Any]) -> str:
        """Builds a human-readable source label from chunk metadata."""

        source = metadata.get("source", "unknown")
        page = metadata.get("page")

        return f"{source} (page {page})" if page is not None else str(source)

    def _extract_sources(self, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
        """Builds a deduplicated list of source metadata for citation display."""

        seen: set[str] = set()
        sources: list[dict[str, Any]] = []

        for chunk in chunks:
            label = self._build_source_label(chunk.metadata)
            if label in seen:
                continue
            seen.add(label)
            sources.append(chunk.metadata)

        return sources
