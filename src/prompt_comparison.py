"""Compare basic and advanced prompting strategies and export the results."""
from __future__ import annotations

import csv
from pathlib import Path

from src.context_builder import ContextBuilder
from src.embeddings import EmbeddingGenerator
from src.evaluation import calculate_answer_similarity, calculate_confidence
from src.generator import generate_answer
from src.retriever import RetrievalEngine
from src.vector_store import VectorDB
from src.evaluation import load_evaluation_questions


def run(limit: int = 10) -> Path:
    cases = load_evaluation_questions("evaluation_questions.json")[:limit]
    retriever = RetrievalEngine(VectorDB("./chroma_db", "knowledge_base"), EmbeddingGenerator())
    builder = ContextBuilder(max_tokens=3000)
    rows = []

    for case in cases:
        chunks = retriever.retrieve(case["question"], top_k=3,
                                    use_hybrid=True, use_multi_query=True)
        context = builder.build_context(chunks, min_score=0.20)
        texts = [chunk.text for chunk in chunks]
        basic = generate_answer(case["question"], context.context, [], "basic")
        advanced = generate_answer(case["question"], context.context, [], "advanced")
        rows.append({
            "question": case["question"],
            "basic_grounding": calculate_confidence(texts, basic),
            "advanced_grounding": calculate_confidence(texts, advanced),
            "basic_similarity": calculate_answer_similarity(case["expected_answer"], basic),
            "advanced_similarity": calculate_answer_similarity(case["expected_answer"], advanced),
            "basic_answer": basic,
            "advanced_answer": advanced,
        })

    output = Path("evaluation_results/prompt_strategy_comparison.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    basic_avg = sum(row["basic_grounding"] + row["basic_similarity"] for row in rows) / (2 * len(rows))
    advanced_avg = sum(row["advanced_grounding"] + row["advanced_similarity"] for row in rows) / (2 * len(rows))
    report = Path("evaluation_results/prompt_strategy_comparison.md")
    winner = "Advanced" if advanced_avg >= basic_avg else "Basic"
    report.write_text(
        "# Prompt Strategy Comparison\n\n"
        f"- Questions compared: {len(rows)}\n"
        f"- Basic average score: {basic_avg:.2f}%\n"
        f"- Advanced average score: {advanced_avg:.2f}%\n"
        f"- Recommended strategy: **{winner}**\n\n"
        "The advanced strategy is designed to improve grounding by explicitly requiring every factual statement to be supported by retrieved context.\n",
        encoding="utf-8",
    )
    print(f"CSV: {output}\nMarkdown: {report}")
    return output


if __name__ == "__main__":
    run()
