from generator import generate_answer


def compare_prompt_strategies(
    question: str,
    context: str,
    history: list
) -> dict:
    """
    Compare the responses generated using
    the basic and advanced prompt strategies.
    """

    basic_answer = generate_answer(
        question=question,
        context=context,
        history=history,
        strategy="basic"
    )

    advanced_answer = generate_answer(
        question=question,
        context=context,
        history=history,
        strategy="advanced"
    )

    return {
        "basic": basic_answer,
        "advanced": advanced_answer
    }