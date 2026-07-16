"""
evaluation.py

Evaluate AI answers, calculate confidence scores,
and generate CSV and Markdown evaluation reports.
"""

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List


EVALUATION_RESULTS_DIR = Path("evaluation_results")


def normalize_text(text: str) -> str:
    """
    Normalize text before comparison.

    This function:
    - Converts text to lowercase.
    - Removes punctuation.
    - Removes extra spaces.
    """

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def calculate_confidence(
    retrieved_chunks: List[str],
    answer: str,
) -> int:
    """
    Estimate a confidence score from 0 to 100 based on
    how much the answer overlaps with the retrieved context.
    """

    if not retrieved_chunks or not answer.strip():
        return 0

    context = normalize_text(
        " ".join(retrieved_chunks)
    )

    normalized_answer = normalize_text(answer)

    answer_words = normalized_answer.split()

    if not answer_words:
        return 0

    matched_words = sum(
        1
        for word in answer_words
        if word in context
    )

    confidence = int(
        (matched_words / len(answer_words)) * 100
    )

    return max(0, min(confidence, 100))


def calculate_answer_similarity(
    expected_answer: str,
    actual_answer: str,
) -> int:
    """
    Compare the expected answer with the actual answer.

    Returns a similarity score from 0 to 100.
    """

    if not expected_answer.strip() or not actual_answer.strip():
        return 0

    expected = normalize_text(expected_answer)
    actual = normalize_text(actual_answer)

    similarity = SequenceMatcher(
        None,
        expected,
        actual,
    ).ratio()

    return int(similarity * 100)


def check_retrieval_correctness(
    expected_answer: str,
    retrieved_chunks: List[str],
) -> bool:
    """
    Check whether the retrieved context contains information
    related to the expected answer.
    """

    if not expected_answer.strip() or not retrieved_chunks:
        return False

    expected_words = set(
        normalize_text(expected_answer).split()
    )

    context_words = set(
        normalize_text(
            " ".join(retrieved_chunks)
        ).split()
    )

    if not expected_words:
        return False

    matched_words = expected_words.intersection(
        context_words
    )

    overlap_ratio = len(matched_words) / len(expected_words)

    return overlap_ratio >= 0.50


def check_answer_grounded(
    answer: str,
    retrieved_chunks: List[str],
) -> bool:
    """
    Check whether the generated answer is sufficiently
    supported by the retrieved context.
    """

    confidence = calculate_confidence(
        retrieved_chunks,
        answer,
    )

    return confidence >= 60


def calculate_overall_score(
    retrieval_correct: bool,
    answer_grounded: bool,
    confidence: int,
    answer_similarity: int,
) -> int:
    """
    Calculate the overall evaluation score.

    Weights:
    - Retrieval correctness: 25%
    - Answer groundedness: 25%
    - Confidence score: 25%
    - Answer similarity: 25%
    """

    retrieval_score = 100 if retrieval_correct else 0
    grounded_score = 100 if answer_grounded else 0

    overall_score = int(
        (
            retrieval_score
            + grounded_score
            + confidence
            + answer_similarity
        )
        / 4
    )

    return max(0, min(overall_score, 100))


def evaluate_answer(
    question: str,
    expected_answer: str,
    actual_answer: str,
    retrieved_chunks: List[str],
) -> Dict[str, Any]:
    """
    Evaluate one AI answer and return all required metrics.
    """

    confidence = calculate_confidence(
        retrieved_chunks,
        actual_answer,
    )

    answer_similarity = calculate_answer_similarity(
        expected_answer,
        actual_answer,
    )

    retrieval_correct = check_retrieval_correctness(
        expected_answer,
        retrieved_chunks,
    )

    answer_grounded = check_answer_grounded(
        actual_answer,
        retrieved_chunks,
    )

    overall_score = calculate_overall_score(
        retrieval_correct,
        answer_grounded,
        confidence,
        answer_similarity,
    )

    return {
        "question": question,
        "expected_answer": expected_answer,
        "actual_answer": actual_answer,
        "retrieval_correct": retrieval_correct,
        "answer_grounded": answer_grounded,
        "confidence": confidence,
        "answer_similarity": answer_similarity,
        "overall_score": overall_score,
    }


def ensure_evaluation_directory() -> None:
    """
    Create the evaluation_results directory if it does not exist.
    """

    EVALUATION_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_evaluation_questions(
    file_path: str = "evaluation_questions.json",
) -> List[Dict[str, Any]]:
    """
    Load evaluation questions from a JSON file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file was not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        data = json.load(json_file)

    if not isinstance(data, list):
        raise ValueError(
            "The evaluation questions file must contain a JSON list."
        )

    return data


def export_report_to_csv(
    results: List[Dict[str, Any]],
    file_name: str = "evaluation_report.csv",
) -> Path:
    """
    Export evaluation results to a CSV file.
    """

    ensure_evaluation_directory()

    output_path = EVALUATION_RESULTS_DIR / file_name

    fieldnames = [
        "question",
        "expected_answer",
        "actual_answer",
        "retrieval_correct",
        "answer_grounded",
        "confidence",
        "answer_similarity",
        "overall_score",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    return output_path


def export_report_to_markdown(
    results: List[Dict[str, Any]],
    file_name: str = "evaluation_report.md",
) -> Path:
    """
    Export evaluation results to a Markdown report.
    """

    ensure_evaluation_directory()

    output_path = EVALUATION_RESULTS_DIR / file_name

    if results:
        average_score = int(
            sum(
                result["overall_score"]
                for result in results
            )
            / len(results)
        )
    else:
        average_score = 0

    lines = [
        "# AI Knowledge Assistant Evaluation Report",
        "",
        f"- Total evaluation questions: {len(results)}",
        f"- Average overall score: {average_score}%",
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"## Test Case {index}",
                "",
                f"**Question:** {result['question']}",
                "",
                f"**Expected Answer:** "
                f"{result['expected_answer']}",
                "",
                f"**Actual Answer:** "
                f"{result['actual_answer']}",
                "",
                f"**Retrieval Correct:** "
                f"{result['retrieval_correct']}",
                "",
                f"**Answer Grounded:** "
                f"{result['answer_grounded']}",
                "",
                f"**Confidence:** "
                f"{result['confidence']}%",
                "",
                f"**Answer Similarity:** "
                f"{result['answer_similarity']}%",
                "",
                f"**Overall Score:** "
                f"{result['overall_score']}%",
                "",
                "---",
                "",
            ]
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_path


def generate_evaluation_reports(
    results: List[Dict[str, Any]],
) -> Dict[str, Path]:
    """
    Generate both CSV and Markdown evaluation reports.
    """

    csv_path = export_report_to_csv(results)
    markdown_path = export_report_to_markdown(results)

    return {
        "csv": csv_path,
        "markdown": markdown_path,
    }


if __name__ == "__main__":
    retrieved_chunks = [
        "Employees receive 21 paid annual leave days per year.",
        "Emergency leave requires approval from the direct manager.",
    ]

    question = (
        "How many annual leave days do employees receive?"
    )

    expected_answer = (
        "Employees receive 21 paid annual leave days per year."
    )

    actual_answer = (
        "Employees receive 21 paid annual leave days per year."
    )

    result = evaluate_answer(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
        retrieved_chunks=retrieved_chunks,
    )

    reports = generate_evaluation_reports([result])

    print("\nEvaluation Result")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nReports created successfully:")
    print(f"CSV: {reports['csv']}")
    print(f"Markdown: {reports['markdown']}")