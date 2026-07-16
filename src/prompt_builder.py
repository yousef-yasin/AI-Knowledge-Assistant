from typing import List


def format_history(history: List[dict]) -> str:
    """
    Convert conversation history into a readable string.
    """

    if not history:
        return "No previous conversation."

    formatted_messages = []

    for message in history:
        role = message.get("role", "Unknown").capitalize()
        content = message.get("content", "")

        formatted_messages.append(f"{role}: {content}")

    return "\n".join(formatted_messages)


def build_prompt(
    question: str,
    context: str,
    history: List[dict]
) -> str:
    """
    Build the basic prompt.
    """

    formatted_history = format_history(history)

    prompt = f"""
You are an AI Knowledge Assistant.

Answer the user's question using ONLY the retrieved context.

Rules:
- Never invent information.
- Never use outside knowledge.
- If the answer is not available in the retrieved context, say:
  "The requested information is not available in the retrieved documents."
- Keep the answer concise.
- Include document citations.
- Give a confidence score from 0 to 100.

Conversation History:
{formatted_history}

Retrieved Context:
{context}

User Question:
{question}

Response Format

Answer:

Sources:

Confidence:
"""

    return prompt.strip()


def build_advanced_prompt(
    question: str,
    context: str,
    history: List[dict]
) -> str:
    """
    Build an advanced prompt for higher accuracy.
    """

    formatted_history = format_history(history)

    prompt = f"""
You are an advanced AI Knowledge Assistant.

Your goal is to generate accurate, grounded, and professional answers.

Instructions:

- Answer ONLY using the retrieved context.
- Never use outside knowledge.
- Never hallucinate.
- Every factual statement must be supported by the retrieved context.
- If the answer is missing from the context, reply exactly:
  "The requested information is not available in the retrieved documents."
- Consider the conversation history only to understand references such as "it", "that", or follow-up questions.
- Do not let conversation history override the retrieved context.
- Keep the answer concise and professional.
- Use bullet points when appropriate.
- Include all relevant document citations.
- Base the confidence score ONLY on the retrieved context.
- Confidence must be an integer from 0 to 100.

Conversation History:
{formatted_history}

Retrieved Context:
{context}

User Question:
{question}

Response Format

Answer:

Sources:
- Document Name:
- Page Number:

Confidence:
"""

    return prompt.strip()








    question = "What about emergency leave?"

    prompt = build_prompt(
        question=question,
        context=context,
        history=history
    )

    print(prompt)