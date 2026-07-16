import os

from dotenv import load_dotenv
from google import genai

from src.prompt_builder import(
    build_prompt,
    build_advanced_prompt
)

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def generate_text(prompt: str) -> str:
    """
    Send any prompt to Gemini and return the generated text.

    This function is intended for components such as:
    - Document Summarizer
    - Query Rewriter
    - Future AI utilities
    """

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Error generating text: {e}"


def generate_answer(
    question: str,
    context: str,
    history: list,
    strategy: str = "basic"
) -> str:
    """
    Generate an answer using Gemini.

    Parameters:
        question : User question
        context  : Retrieved context
        history  : Conversation history
        strategy : basic | advanced
    """

    if strategy.lower() == "advanced":

        prompt = build_advanced_prompt(
            question=question,
            context=context,
            history=history
        )

    else:

        prompt = build_prompt(
            question=question,
            context=context,
            history=history
        )

    return generate_text(prompt)


def stream_generate_answer(
    question: str,
    context: str,
    history: list,
    strategy: str = "basic"
):
    """
    Generate a streaming response from Gemini.

    Yields pieces of the response as they are generated.
    """

    if strategy.lower() == "advanced":

        prompt = build_advanced_prompt(
            question=question,
            context=context,
            history=history
        )

    else:

        prompt = build_prompt(
            question=question,
            context=context,
            history=history
        )

    try:

        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt
        )

        for chunk in response:

            if chunk.text:
                yield chunk.text

    except Exception as e:

        yield f"Error generating response: {e}"




