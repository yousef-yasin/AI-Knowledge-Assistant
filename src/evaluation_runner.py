"""Run the 25-question automatic end-to-end evaluation suite."""
from __future__ import annotations

from src.context_builder import ContextBuilder
from src.embeddings import EmbeddingGenerator
from src.evaluation import evaluate_answer, generate_evaluation_reports, load_evaluation_questions
from src.generator import generate_answer
from src.retriever import RetrievalEngine
from src.vector_store import VectorDB


def run() -> None:
    questions = load_evaluation_questions("evaluation_questions.json")
    vector_db = VectorDB("./chroma_db", "knowledge_base")
    retriever = RetrievalEngine(vector_db, EmbeddingGenerator())
    builder = ContextBuilder(max_tokens=3000)
    results = []

    for index, case in enumerate(questions, 1):
        chunks = retriever.retrieve(case["question"], top_k=3,
                                    use_hybrid=True, use_multi_query=True)
        context = builder.build_context(chunks, min_score=0.20)
        answer = generate_answer(case["question"], context.context, [], "advanced")
        result = evaluate_answer(
            case["question"], case["expected_answer"], answer,
            [chunk.text for chunk in chunks],
        )
        results.append(result)
        print(f"[{index:02d}/{len(questions)}] score={result['overall_score']}% — {case['question']}")

    reports = generate_evaluation_reports(results)
    print("\nEvaluation completed:")
    for kind, path in reports.items():
        print(f"{kind.upper()}: {path}")


if __name__ == "__main__":
    run()
