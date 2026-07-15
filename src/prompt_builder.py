from typing import List


def format_history(history: List[dict]) -> str:
    """
    Convert conversation history into a readable string.
    """

    if not history:
        return "No previous conversation."

    formatted_messages = []

    for message in history:
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")

        formatted_messages.append(f"{role}: {content}")

    return "\n".join(formatted_messages)


def build_prompt(
    question: str,
    context: str,
    history: List[dict]
) -> str:
    """
    Build the final prompt that will be sent to Gemini.
    """

    formatted_history = format_history(history)

    prompt = f"""
You are an AI Knowledge Assistant.

Answer the user's question using ONLY the retrieved context below.
Do not use your own knowledge.

Rules:

- Never invent information.
- Never use outside knowledge.
- If the answer does not exist in the context, reply:
  "The requested information is not available in the provided documents."
- Be concise.
- Use bullet points when appropriate.
- Include document citations at the end.
- Base the confidence score only on how clearly the answer appears in the context.
- Confidence must be a number from 0 to 100.

----------------------------
Conversation History
----------------------------

{formatted_history}

----------------------------
Retrieved Context
----------------------------

{context}

----------------------------
User Question
----------------------------

{question}

----------------------------
Answer Format
----------------------------

Answer:
Sources:
Confidence:
"""

    return prompt.strip()


if __name__ == "__main__":
    history = [
        {
            "role": "user",
            "content": "Tell me about annual leave."
        },
        {
            "role": "assistant",
            "content": "Employees receive 21 annual leave days."
        }
    ]

    context = """
Source: Employee Handbook.pdf
Page: 15

Employees receive 21 annual leave days.
Emergency leave requires manager approval.
"""

    question = "What about emergency leave?"

    prompt = build_prompt(
        question=question,
        context=context,
        history=history
    )

    print(prompt)