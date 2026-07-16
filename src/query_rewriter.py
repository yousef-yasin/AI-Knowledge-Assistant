from __future__ import annotations

import re
from dataclasses import dataclass, field


# Filler phrases users often prepend to a real question; stripping these
# gives the embedding model a cleaner, more direct query to work with.
FILLER_PATTERNS: list[str] = [
    r"^can you (please )?",
    r"^could you (please )?",
    r"^i was wondering (if )?",
    r"^i want to know ",
    r"^please tell me ",
    r"^tell me ",
    r"^i'd like to know ",
    r"^do you know ",
]

# Common abbreviations expanded so the query is more likely to match the
# full terms actually used in source documents.
ABBREVIATION_MAP: dict[str, str] = {
    "faq": "frequently asked questions",
    "docs": "documentation",
    "info": "information",
    "config": "configuration",
    "auth": "authentication",
}


@dataclass
class RewrittenQuery:
    """Stores the original query alongside its optimized version."""

    original: str  # The user's raw question, unchanged.
    rewritten: str  # The cleaned-up, retrieval-optimized version.
    # Which rules actually changed something, for debugging.
    applied_rules: list[str] = field(default_factory=list)


class QueryRewriter:
    """
    Rewrites raw user questions into cleaner, more retrieval-friendly queries.

    Currently rule-based, with no external API dependency — this keeps AKA-12
    unblocked ahead of AKA-14 (Gemini integration, Sprint 2). An LLM-based
    rewriter can be swapped in later without changing how RetrievalEngine
    calls it: just implement a class with the same `rewrite(query) -> RewrittenQuery`
    signature and pass it in instead of this one.
    """

    def __init__(
        self,
        filler_patterns: list[str] | None = None,
        abbreviation_map: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            filler_patterns: Regex patterns matched against the start of a
                query and stripped out. Uses the module defaults if not provided.
            abbreviation_map: Lowercase word -> expanded phrase. Uses the
                module defaults if not provided.
        """

        self.filler_patterns = filler_patterns or FILLER_PATTERNS
        self.abbreviation_map = abbreviation_map or ABBREVIATION_MAP

    def rewrite(self, query: str) -> RewrittenQuery:
        """
        Rewrite a raw user query into an optimized search query.

        Args:
            query: The user's raw question.

        Returns:
            A RewrittenQuery with the optimized version and which rules fired.
        """

        cleaned = query.strip()

        if not cleaned:
            raise ValueError("Cannot rewrite an empty query.")

        applied_rules: list[str] = []

        stripped = self._strip_filler(cleaned)
        if stripped != cleaned:
            applied_rules.append("stripped_filler_phrase")

        expanded = self._expand_abbreviations(stripped)
        if expanded != stripped:
            applied_rules.append("expanded_abbreviations")

        normalized = self._normalize_whitespace(expanded)

        final_query = normalized[0].upper(
        ) + normalized[1:] if normalized else normalized
        # Capitalizes the first letter for readability; doesn't affect
        # embedding quality since the model is case-insensitive-ish, but
        # keeps logged/displayed queries tidy.

        return RewrittenQuery(
            original=cleaned,
            rewritten=final_query,
            applied_rules=applied_rules,
        )

    def _strip_filler(self, text: str) -> str:
        """Removes common conversational lead-ins from the start of the query."""

        result = text
        for pattern in self.filler_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        return result.strip()

    def _expand_abbreviations(self, text: str) -> str:
        """Replaces known abbreviations with their full form, word by word."""

        words = text.split()
        expanded_words = [self.abbreviation_map.get(
            word.lower(), word) for word in words]
        return " ".join(expanded_words)

    def _normalize_whitespace(self, text: str) -> str:
        """Collapses repeated spaces left behind by stripping/expanding."""
        return re.sub(r"\s+", " ", text).strip()


def rewrite_query(query: str) -> RewrittenQuery:
    """
    Helper function for rewriting a query in one line.

    Example:
        result = rewrite_query("can you tell me about the faq")
    """

    rewriter = QueryRewriter()
    return rewriter.rewrite(query)
