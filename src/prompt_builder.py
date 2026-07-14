from typing import List


def format_history(history: List[dict]) -> str:
    """
    Converts conversation history into a readable string.
    """

    if not history:
        return "No previous conversation."

    formatted = ""

    for message in history:
        role = message["role"].capitalize()
        content = message["content"]
        formatted += f"{role}: {content}\n"

    return formatted


def build_prompt(question: str, context: str, history: List[dict]) -> str:
    """
    Builds the final prompt for Gemini.
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

    return prompt

