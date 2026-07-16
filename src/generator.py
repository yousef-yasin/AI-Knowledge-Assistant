from __future__ import annotations

import os
from collections.abc import Iterator

from dotenv import load_dotenv
from google import genai

from src.prompt_builder import build_advanced_prompt, build_prompt, format_history

load_dotenv()


def _client() -> genai.Client:
    """Create a configured Gemini client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing from the .env file.")
    return genai.Client(api_key=api_key)


def generate_text(prompt: str) -> str:
    """Generate one complete Gemini response while keeping the client alive."""
    client: genai.Client | None = None
    try:
        client = _client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = (response.text or "").strip()
        return text or "Error generating text: Gemini returned an empty response."
    except Exception as error:
        return f"Error generating text: {error}"
    finally:
        if client is not None:
            client.close()


def generate_answer(
    question: str,
    context: str,
    history: list,
    strategy: str = "advanced",
) -> str:
    prompt_builder = (
        build_advanced_prompt
        if strategy.lower() == "advanced"
        else build_prompt
    )
    prompt = prompt_builder(
        question=question,
        context=context,
        history=history,
    )
    return generate_text(prompt)


def generate_memory_answer(question: str, history: list) -> str:
    prompt = f"""You are an AI conversation-memory assistant.
Answer the current question ONLY from the conversation history below.
Do not use outside knowledge. If history is insufficient, reply exactly:
The requested information is not available in the conversation memory.

Conversation History:
{format_history(history)}

Current Question:
{question}

Answer:"""
    return generate_text(prompt)


def stream_generate_answer(
    question: str,
    context: str,
    history: list,
    strategy: str = "advanced",
) -> Iterator[str]:
    """Stream a Gemini answer while retaining the client for the full stream."""
    prompt_builder = (
        build_advanced_prompt
        if strategy.lower() == "advanced"
        else build_prompt
    )
    prompt = prompt_builder(
        question=question,
        context=context,
        history=history,
    )

    client: genai.Client | None = None
    try:
        # Keep this reference alive until every streamed chunk is consumed.
        client = _client()
        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        received_text = False
        for chunk in response:
            if chunk.text:
                received_text = True
                yield chunk.text

        if not received_text:
            yield "Error generating response: Gemini returned an empty response."
    except Exception as error:
        yield f"Error generating response: {error}"
    finally:
        if client is not None:
            client.close()
