from typing import List


def format_history(history: List[dict]) -> str:
    """
    Convert conversation history into a readable string.
    """

    if not history:
        return "No previous conversation."

    formatted_messages = []

    for message in history:
        role = message.get(
            "role",
            "Unknown",
        ).capitalize()

        content = message.get(
            "content",
            "",
        )

        formatted_messages.append(
            f"{role}: {content}"
        )

    return "\n".join(formatted_messages)


def build_prompt(
    question: str,
    context: str,
    history: List[dict],
) -> str:
    """
    Build a basic grounded prompt.
    """

    formatted_history = format_history(history)

    prompt = f"""
You are an AI Knowledge Assistant.

Your task is to answer the user's question using ONLY
the retrieved context.

Rules:

- Use only information available in the retrieved context.
- Never use outside knowledge.
- Never invent information.
- If the answer is not available in the context, reply exactly:
"The requested information is not available in the retrieved documents."
- Keep the answer concise and professional.
- Do not include sources.
- Do not include citations.
- Do not include confidence scores.
- Return only the final answer text.

Conversation History:
{formatted_history}

Retrieved Context:
{context}

User Question:
{question}

Answer:
"""

    return prompt.strip()


def build_advanced_prompt(
    question: str,
    context: str,
    history: List[dict],
) -> str:
    """
    Build an advanced prompt for higher accuracy.
    """

    formatted_history = format_history(history)

    prompt = f"""
You are an advanced AI Knowledge Assistant.

Generate a precise and professional answer based ONLY
on the retrieved context.

Instructions:

- Answer ONLY using the retrieved context.
- Every factual statement must be supported by the context.
- Never hallucinate.
- Never use external knowledge.
- Use conversation history only to understand references
  such as "it", "that", or follow-up questions.
- Do not allow conversation history to override retrieved context.
- If the information is missing, reply exactly:
"The requested information is not available in the retrieved documents."
- Keep the answer concise.
- Use bullet points if they improve readability.
- Return ONLY the answer text.

Conversation History:
{formatted_history}

Retrieved Context:
{context}

User Question:
{question}

Answer:
"""

    return prompt.strip()