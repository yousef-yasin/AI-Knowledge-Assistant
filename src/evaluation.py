"""
evaluation.py

Evaluate AI answers and calculate a confidence score.
"""

from typing import List #to support type hinting for lists


def calculate_confidence( #to calculate a confidence score based on the overlap between the answer and the retrieved context
    retrieved_chunks: List[str], #meaning a list of strings representing the retrieved context chunks
    answer: str #GEMINI ANSWER
) -> int:
    """
    Estimate a confidence score (0-100) based on
    how much the answer overlaps with the retrieved context.
    """

    if not retrieved_chunks:
        return 0

    context = " ".join(retrieved_chunks).lower()
    answer = answer.lower()

    answer_words = answer.split() #to make it like list of words

    if not answer_words:
        return 0

    matched_words = 0

    for word in answer_words:
        if word in context:
            matched_words += 1

    confidence = int((matched_words / len(answer_words)) * 100)

    confidence = max(0, min(confidence, 100))

    return confidence


def evaluate_answer( #to evaluate the answer and return useful evaluation information
    question: str,
    answer: str,
    retrieved_chunks: List[str]
) -> dict:
    """
    Evaluate the generated answer and return
    useful evaluation information.
    """

    confidence = calculate_confidence(
        retrieved_chunks,
        answer
    )

    return {
        "question": question,
        "answer": answer,
        "confidence": confidence
    }


if __name__ == "__main__":

    retrieved_chunks = [
        "Employees receive 21 annual leave days.",
        "Emergency leave requires manager approval."
    ]

    question = "How many annual leave days do employees receive?"

    answer = "Employees receive 21 annual leave days."

    result = evaluate_answer(
        question,
        answer,
        retrieved_chunks
    )

    print(result)