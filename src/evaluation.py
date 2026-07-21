"""
evaluation.py

Evaluate AI answers, calculate confidence scores,
and generate CSV and Markdown evaluation reports.
"""

# Import the CSV module to create CSV evaluation reports.
import csv

# Import the JSON module to read evaluation questions from JSON files.
import json

# Import regular expressions for cleaning and normalizing text.
import re

# Import SequenceMatcher to calculate similarity between two strings.
from difflib import SequenceMatcher

# Import Path for easier file and directory handling.
from pathlib import Path

# Import type hints to make function inputs and outputs clearer.
from typing import Any, Dict, List


# Define the directory where evaluation reports will be stored.
EVALUATION_RESULTS_DIR = Path("evaluation_results")


def normalize_text(text: str) -> str:
    """
    Normalize text before comparison.

    This function:
    - Converts text to lowercase.
    - Removes punctuation.
    - Removes extra spaces.
    """

    # Convert all characters to lowercase so comparison is case-insensitive.
    text = text.lower()

    # Replace punctuation and special characters with spaces.
    text = re.sub(r"[^\w\s]", " ", text)

    # Replace multiple spaces with one single space.
    text = re.sub(r"\s+", " ", text)

    # Remove spaces from the beginning and end of the text.
    return text.strip()


def calculate_confidence(
    retrieved_chunks: List[str],
    answer: str,
) -> int:
    """
    Estimate a confidence score from 0 to 100 based on
    how much the answer overlaps with the retrieved context.
    """

    # Return zero if no context was retrieved or the answer is empty.
    if not retrieved_chunks or not answer.strip():
        return 0

    # Combine all retrieved chunks into one string and normalize it.
    context = normalize_text(
        " ".join(retrieved_chunks)
    )

    # Normalize the generated answer before comparing it with the context.
    normalized_answer = normalize_text(answer)

    # Split the normalized answer into individual words.
    answer_words = normalized_answer.split()

    # Return zero if the normalized answer contains no words.
    if not answer_words:
        return 0

    # Count how many words from the answer appear in the retrieved context.
    matched_words = sum(
        1
        for word in answer_words
        if word in context
    )

    # Calculate the percentage of answer words supported by the context.
    confidence = int(
        (matched_words / len(answer_words)) * 100
    )

    # Ensure the confidence score remains between 0 and 100.
    return max(0, min(confidence, 100))


def calculate_answer_similarity(
    expected_answer: str,
    actual_answer: str,
) -> int:
    """
    Compare the expected answer with the actual answer.

    Returns a similarity score from 0 to 100.
    """

    # Return zero if either answer is empty.
    if not expected_answer.strip() or not actual_answer.strip():
        return 0

    # Normalize the expected answer before comparison.
    expected = normalize_text(expected_answer)

    # Normalize the actual AI-generated answer before comparison.
    actual = normalize_text(actual_answer)

    # Calculate the character-sequence similarity between both answers.
    similarity = SequenceMatcher(
        None,
        expected,
        actual,
    ).ratio()

    # Convert the similarity value from 0–1 into a percentage from 0–100.
    return int(similarity * 100)


def check_retrieval_correctness(
    expected_answer: str,
    retrieved_chunks: List[str],
) -> bool:
    """
    Check whether the retrieved context contains information
    related to the expected answer.
    """

    # Return False if the expected answer is empty or no chunks were retrieved.
    if not expected_answer.strip() or not retrieved_chunks:
        return False

    # Normalize the expected answer and store its unique words in a set.
    expected_words = set(
        normalize_text(expected_answer).split()
    )

    # Combine and normalize the retrieved chunks,
    # then store their unique words in a set.
    context_words = set(
        normalize_text(
            " ".join(retrieved_chunks)
        ).split()
    )

    # Return False if the expected answer contains no valid words.
    if not expected_words:
        return False

    # Find the words shared by the expected answer and retrieved context.
    matched_words = expected_words.intersection(
        context_words
    )

    # Calculate how much of the expected answer appears in the context.
    overlap_ratio = len(matched_words) / len(expected_words)

    # Consider retrieval correct if at least 50% of the words overlap.
    return overlap_ratio >= 0.50


def check_answer_grounded(
    answer: str,
    retrieved_chunks: List[str],
) -> bool:
    """
    Check whether the generated answer is sufficiently
    supported by the retrieved context.
    """

    # Calculate how much of the generated answer appears in the context.
    confidence = calculate_confidence(
        retrieved_chunks,
        answer,
    )

    # Consider the answer grounded if confidence is at least 60%.
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

    # Convert retrieval correctness from Boolean to a numeric score.
    retrieval_score = 100 if retrieval_correct else 0

    # Convert answer groundedness from Boolean to a numeric score.
    grounded_score = 100 if answer_grounded else 0

    # Calculate the average of the four evaluation metrics.
    overall_score = int(
        (
            retrieval_score
            + grounded_score
            + confidence
            + answer_similarity
        )
        / 4
    )

    # Ensure the final score remains between 0 and 100.
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

    # Calculate the confidence of the generated answer.
    confidence = calculate_confidence(
        retrieved_chunks,
        actual_answer,
    )

    # Compare the actual answer with the expected answer.
    answer_similarity = calculate_answer_similarity(
        expected_answer,
        actual_answer,
    )

    # Check whether the correct information was retrieved.
    retrieval_correct = check_retrieval_correctness(
        expected_answer,
        retrieved_chunks,
    )

    # Check whether the generated answer is supported by the context.
    answer_grounded = check_answer_grounded(
        actual_answer,
        retrieved_chunks,
    )

    # Calculate the final overall evaluation score.
    overall_score = calculate_overall_score(
        retrieval_correct,
        answer_grounded,
        confidence,
        answer_similarity,
    )

    # Return all evaluation details in a dictionary.
    return {
        # The original question asked by the user.
        "question": question,

        # The correct or expected answer.
        "expected_answer": expected_answer,

        # The answer generated by the AI system.
        "actual_answer": actual_answer,

        # Whether the retrieval system found relevant information.
        "retrieval_correct": retrieval_correct,

        # Whether the answer is supported by the retrieved information.
        "answer_grounded": answer_grounded,

        # The context-overlap confidence score.
        "confidence": confidence,

        # The similarity score between expected and actual answers.
        "answer_similarity": answer_similarity,

        # The final combined evaluation score.
        "overall_score": overall_score,
    }


def ensure_evaluation_directory() -> None:
    """
    Create the evaluation_results directory if it does not exist.
    """

    # Create the output directory and any missing parent directories.
    # No error is raised if the directory already exists.
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

    # Convert the provided file path into a Path object.
    path = Path(file_path)

    # Check whether the evaluation questions file exists.
    if not path.exists():
        # Raise a clear error if the file cannot be found.
        raise FileNotFoundError(
            f"Evaluation file was not found: {path}"
        )

    # Open the JSON file in read mode using UTF-8 encoding.
    with path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        # Convert the JSON content into a Python object.
        data = json.load(json_file)

    # Verify that the JSON file contains a list.
    if not isinstance(data, list):
        # Raise an error if the JSON structure is incorrect.
        raise ValueError(
            "The evaluation questions file must contain a JSON list."
        )

    # Return the loaded evaluation questions.
    return data


def export_report_to_csv(
    results: List[Dict[str, Any]],
    file_name: str = "evaluation_report.csv",
) -> Path:
    """
    Export evaluation results to a CSV file.
    """

    # Ensure that the evaluation output directory exists.
    ensure_evaluation_directory()

    # Create the complete output path for the CSV file.
    output_path = EVALUATION_RESULTS_DIR / file_name

    # Define the columns that will appear in the CSV report.
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

    # Open the CSV file in write mode.
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        # Create a CSV writer that writes dictionaries as rows.
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        # Write the column names as the first row.
        writer.writeheader()

        # Write all evaluation result dictionaries to the CSV file.
        writer.writerows(results)

    # Return the path of the generated CSV report.
    return output_path


def export_report_to_markdown(
    results: List[Dict[str, Any]],
    file_name: str = "evaluation_report.md",
) -> Path:
    """
    Export evaluation results to a Markdown report.
    """

    # Ensure that the evaluation output directory exists.
    ensure_evaluation_directory()

    # Create the complete output path for the Markdown file.
    output_path = EVALUATION_RESULTS_DIR / file_name

    # Calculate the average overall score if results are available.
    if results:
        average_score = int(
            sum(
                result["overall_score"]
                for result in results
            )
            / len(results)
        )
    else:
        # Use zero when there are no evaluation results.
        average_score = 0

    # Create the report title and summary section.
    lines = [
        "# AI Knowledge Assistant Evaluation Report",
        "",
        f"- Total evaluation questions: {len(results)}",
        f"- Average overall score: {average_score}%",
        "",
    ]

    # Add a detailed report section for every evaluation result.
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                # Display the test case number.
                f"## Test Case {index}",
                "",

                # Display the evaluated question.
                f"**Question:** {result['question']}",
                "",

                # Display the correct expected answer.
                f"**Expected Answer:** "
                f"{result['expected_answer']}",
                "",

                # Display the answer generated by the AI.
                f"**Actual Answer:** "
                f"{result['actual_answer']}",
                "",

                # Display whether retrieval was correct.
                f"**Retrieval Correct:** "
                f"{result['retrieval_correct']}",
                "",

                # Display whether the answer was grounded.
                f"**Answer Grounded:** "
                f"{result['answer_grounded']}",
                "",

                # Display the confidence percentage.
                f"**Confidence:** "
                f"{result['confidence']}%",
                "",

                # Display the answer similarity percentage.
                f"**Answer Similarity:** "
                f"{result['answer_similarity']}%",
                "",

                # Display the final overall score.
                f"**Overall Score:** "
                f"{result['overall_score']}%",
                "",

                # Add a separator between test cases.
                "---",
                "",
            ]
        )

    # Join all report lines and write them to the Markdown file.
    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # Return the path of the generated Markdown report.
    return output_path


def generate_evaluation_reports(
    results: List[Dict[str, Any]],
) -> Dict[str, Path]:
    """
    Generate both CSV and Markdown evaluation reports.
    """

    # Generate the CSV evaluation report.
    csv_path = export_report_to_csv(results)

    # Generate the Markdown evaluation report.
    markdown_path = export_report_to_markdown(results)

    # Return the paths of both generated reports.
    return {
        "csv": csv_path,
        "markdown": markdown_path,
    }


# Run the following test code only when this file is executed directly.
if __name__ == "__main__":
    # Example chunks that simulate information retrieved
    # from the knowledge base.
    retrieved_chunks = [
        "Employees receive 21 paid annual leave days per year.",
        "Emergency leave requires approval from the direct manager.",
    ]

    # Example question to evaluate.
    question = (
        "How many annual leave days do employees receive?"
    )

    # The correct answer expected from the AI system.
    expected_answer = (
        "Employees receive 21 paid annual leave days per year."
    )

    # The actual answer generated by the AI system.
    actual_answer = (
        "Employees receive 21 paid annual leave days per year."
    )

    # Evaluate the example answer using all evaluation metrics.
    result = evaluate_answer(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
        retrieved_chunks=retrieved_chunks,
    )

    # Generate CSV and Markdown reports for the example result.
    reports = generate_evaluation_reports([result])

    # Print a title for the evaluation result in the terminal.
    print("\nEvaluation Result")

    # Print a separator line.
    print("-" * 40)

    # Print every evaluation metric and its value.
    for key, value in result.items():
        print(f"{key}: {value}")

    # Print a success message after generating the reports.
    print("\nReports created successfully:")

    # Print the location of the generated CSV file.
    print(f"CSV: {reports['csv']}")

    # Print the location of the generated Markdown file.
    print(f"Markdown: {reports['markdown']}")