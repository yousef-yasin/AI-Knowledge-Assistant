from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.knowledge_base.loader import LoadedDocument


@dataclass
class SummarizedDocument:
    text: str
    summary: str
    metadata: dict[str, Any]


class DocumentSummarizer:
    """Create deterministic extractive summaries during indexing.

    The implementation intentionally avoids extra API calls during indexing.
    It scores sentences using document term frequency and keeps the most
    informative sentences in their original order.
    """

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
        "is", "are", "was", "were", "be", "by", "with", "that", "this",
        "as", "at", "from", "it", "its", "their", "they", "them", "can",
        "will", "may", "must", "should", "have", "has", "had",
    }

    def __init__(self, max_sentences: int = 4, max_characters: int = 1200) -> None:
        if max_sentences < 1 or max_characters < 50:
            raise ValueError("Invalid summarizer limits.")
        self.max_sentences = max_sentences
        self.max_characters = max_characters

    def summarize_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("Cannot summarize empty text.")
        if len(cleaned) <= self.max_characters:
            return cleaned

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
            if sentence.strip()
        ]
        if len(sentences) <= self.max_sentences:
            return cleaned[: self.max_characters].rstrip()

        words = re.findall(r"[A-Za-z0-9_]+", cleaned.lower())
        frequencies = Counter(
            word for word in words if word not in self.STOP_WORDS and len(word) > 2
        )

        ranked: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            sentence_words = re.findall(r"[A-Za-z0-9_]+", sentence.lower())
            score = sum(frequencies[word] for word in sentence_words)
            score = score / max(len(sentence_words), 1)
            ranked.append((score, index, sentence))

        chosen = sorted(
            sorted(ranked, reverse=True)[: self.max_sentences],
            key=lambda item: item[1],
        )
        summary = " ".join(item[2] for item in chosen)
        return summary[: self.max_characters].rstrip()

    def summarize_document(self, document: LoadedDocument) -> SummarizedDocument:
        summary = self.summarize_text(document.text)
        metadata = document.metadata.copy()
        metadata["content_type"] = "document_summary"
        metadata["is_summary"] = True
        metadata["chunk_number"] = 0
        return SummarizedDocument(
            text=document.text,
            summary=summary,
            metadata=metadata,
        )

    def summarize_documents(self, documents: list[LoadedDocument]) -> list[SummarizedDocument]:
        return [self.summarize_document(document) for document in documents if document.text.strip()]
