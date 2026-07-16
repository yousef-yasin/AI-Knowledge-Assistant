import os
from dotenv import load_dotenv
from google import genai

from prompt_builder import (
    build_prompt,
    build_advanced_prompt
)

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def generate_answer(
    question: str,
    context: str,
    history: list,
    strategy: str = "basic",
    stream: bool = False
):
    """
    Generate an answer using Gemini.

    Parameters:
        question: User question
        context: Retrieved context
        history: Conversation history
        strategy: "basic" or "advanced"
        stream: Enable streaming response

    Returns:
        str (normal mode)
        generator (streaming mode)
    """

    # Choose prompt strategy
    if strategy.lower() == "advanced":
        prompt = build_advanced_prompt(
            question,
            context,
            history
        )
    else:
        prompt = build_prompt(
            question,
            context,
            history
        )

    try:

        # Streaming Response
        if stream:

            response = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=prompt
            )

            for chunk in response:

                if chunk.text:
                    yield chunk.text

        # Normal Response
        else:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            return response.text

    except Exception as e:

        if stream:
            yield f"Error generating response: {e}"
        else:
            return f"Error generating response: {e}"








#old
'''

import os
# to reach the Environment Variables
from dotenv import load_dotenv
# to read the env. file and load it's values
from google import genai
# import Google GenAI library to use Gemini

from prompt_builder import build_prompt

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

'''