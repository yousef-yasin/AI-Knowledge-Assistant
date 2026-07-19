from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from dotenv import load_dotenv
from google import genai

from src.prompt_builder import (
    build_advanced_prompt,
    build_prompt,
    format_history,
)

load_dotenv()


def _create_client() -> genai.Client:
    """Create a configured Gemini client only when needed."""

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing from the .env file."
        )

    return genai.Client(api_key=api_key)


class _ModelsProxy:
    """
    Lazy proxy for Gemini models.

    This keeps generator.client.models available so the
    existing tests can monkeypatch generate_content.
    """

    def generate_content(
        self,
        **kwargs: Any,
    ):
        real_client = _create_client()

        try:
            return real_client.models.generate_content(
                **kwargs
            )
        finally:
            real_client.close()

    def generate_content_stream(
        self,
        **kwargs: Any,
    ):
        real_client = _create_client()

        try:
            for chunk in (
                real_client.models
                .generate_content_stream(**kwargs)
            ):
                yield chunk
        finally:
            real_client.close()


class _ClientProxy:
    def __init__(self) -> None:
        self.models = _ModelsProxy()


# Keep this public because the existing tests patch:
# generator.client.models.generate_content
client = _ClientProxy()


def generate_text(prompt: str) -> str:
    """Generate one complete Gemini response."""

    try:
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = (
            response.text or ""
        ).strip()

        return (
            text
            or (
                "Error generating text: "
                "Gemini returned an empty response."
            )
        )

    except Exception as error:
        return f"Error generating text: {error}"
    


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


def generate_memory_answer(
    question: str,
    history: list,
) -> str:
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


    try:
        received_text = False

        response = (
            client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=prompt,
            )
        )

        for chunk in response:
            if chunk.text:
                received_text = True
                yield chunk.text

        if not received_text:
            yield (
                "Error generating response: "
                "Gemini returned an empty response."
            )

    except Exception as error:
        yield f"Error generating response: {error}"
