import os
# to reach the Environment Variables
from dotenv import load_dotenv
# to read the env. file and load it's values
from google import genai
# import Google GenAI library to use Gemini

from src.prompt_builder import build_prompt
# Load environment variables
load_dotenv()

# Create Gemini client (to contact with Gemini)
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def generate_answer(question: str, context: str, history: list) -> str:
    """
    Generates an answer using Gemini based on the retrieved context.
    """

    # Build the prompt
    prompt = build_prompt(
        question=question,
        context=context,
        history=history
    )

    try:
        # Send the prompt to Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Error generating response: {e}"

def generate_text(prompt: str) -> str:
    """
    Generates text directly from a provided prompt.

    This helper function can be used by other modules,
    such as the document summarizer.
    """

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")

    cleaned_prompt = prompt.strip()

    if not cleaned_prompt:
        raise ValueError("prompt cannot be empty.")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=cleaned_prompt,
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        return response.text.strip()

    except Exception as error:
        raise RuntimeError(
            f"Error generating text: {error}"
        ) from error