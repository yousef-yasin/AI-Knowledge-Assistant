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