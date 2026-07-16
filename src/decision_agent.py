import os
from dotenv import load_dotenv
from google import genai

from prompt_builder import format_history

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def needs_retrieval(question: str, history: list) -> bool:
    """
    Decide whether document retrieval is required
    or if conversation memory is sufficient.
    """

    history_text = format_history(history)

    prompt = f"""
You are an AI Retrieval Decision Agent.

Conversation History:
{history_text}

Current Question:
{question}

Task:

Decide whether the system should retrieve information
from the knowledge base.

Return YES if:
- The question requires information from documents.
- The answer is not completely available from conversation history.
- The user asks about company policies, rules, documents, or facts.

Return NO if:
- The question is only a follow-up.
- The answer can be inferred from conversation history.
- The user refers to previous answers using words like:
  "it", "that", "those", "previous answer", etc.

Return ONLY one word:

YES

or

NO
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        decision = response.text.strip().upper()

        if "YES" in decision:
            return True

        if "NO" in decision:
            return False

        # Default: retrieve documents
        return True

    except Exception:

        # Safer fallback
        return True