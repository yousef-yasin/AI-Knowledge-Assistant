from __future__ import annotations

import re


class RetrievalDecisionAgent:
    """Decide whether a question needs document retrieval or memory only."""

    DOCUMENT_TERMS = {
        "policy", "document", "handbook", "leave", "vacation", "salary",
        "department", "working hours", "company", "rule", "procedure",
        "benefit", "employee", "annual", "sick", "maternity",
    }
    FOLLOW_UP_PATTERNS = (
        r"^(and|what about|how about)\b",
        r"\b(it|that|those|this|previous answer|you said)\b",
    )

    def decide(self, question: str, history: list[dict]) -> str:
        cleaned = question.strip().lower()
        if not history:
            return "retrieval"
        if any(term in cleaned for term in self.DOCUMENT_TERMS):
            return "retrieval"
        if any(re.search(pattern, cleaned) for pattern in self.FOLLOW_UP_PATTERNS):
            return "memory"
        return "retrieval"


def needs_retrieval(question: str, history: list[dict]) -> bool:
    return RetrievalDecisionAgent().decide(question, history) == "retrieval"
