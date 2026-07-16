import os
import json
# we want to convert the text returned by Gemini into a Python dictionary.

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def validate_answer(question: str, context: str, answer: str) -> dict:
    """
    Validates whether the generated answer is fully supported
    by the retrieved context.

    Returns:
        {
            "status": "PASS" or "FAIL",
            "reason": "...",
            "supported": True/False
        }
    """
#we build a dedicated prompt for verification.
    prompt = f"""
You are an AI response validator.

Your task is to validate whether the generated answer is completely supported by the retrieved context.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Validation Rules:

1. Every statement must be supported by the context.
2. Do not allow hallucinated information.
3. The answer must be relevant to the question.
4. The answer must not contain outside knowledge.
5. If citations exist, they must match the context.

Return ONLY valid JSON in this format:

{{
    "status": "PASS",
    "reason": "Short explanation",
    "supported": true
}}

or

{{
    "status": "FAIL",
    "reason": "Short explanation",
    "supported": false
}}

Do not return markdown.
Do not return code blocks.
Return JSON only.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        result = response.text.strip()

        if result.startswith("```json"):
            result = result[7:]
        elif result.startswith("```"):
            result = result[3:]

        if result.endswith("```"):
            result = result[:-3]

        return json.loads(result.strip())

    except Exception as e:

        return {
            "status": "FAIL",
            "reason": f"Validation error: {str(e)}",
            "supported": False
        }