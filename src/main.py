"""
main.py

Main integration file for the AI Knowledge Assistant.

This file connects:

1. Query rewriting
2. Semantic retrieval
3. Context building
4. Prompt building
5. Gemini answer generation
6. Answer validation
7. Confidence evaluation
8. Conversation export
"""


from typing import Any


# Import the query rewriting function
from query_rewriter import rewrite_query


# Import the retrieval function
from retriever import retrieve_relevant_chunks


# Import the context building function
from context_builder import build_context


# Import the prompt building function
from prompt_builder import build_prompt


# Import the Gemini answer generation function
from generator import generate_answer


# Import the response validation function
from validator import validate_answer


# Import the evaluation function
from evaluation import evaluate_answer


# Import the conversation export function
from export import export_conversation


def get_chunk_text(chunk: Any) -> str:
    """
    Extract the text from a retrieved chunk.

    The retrieved chunk may be:
    - A string
    - A dictionary containing text or content
    """

    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, dict):
        return str(
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("chunk_text")
            or ""
        )

    return str(chunk)


def extract_sources(
    retrieved_chunks: list[Any]
) -> list[dict[str, Any]]:
    """
    Extract document source information from retrieved chunks.

    Each source may contain:
    - document_name
    - page_number
    - chunk_number
    """

    sources: list[dict[str, Any]] = []

    for chunk in retrieved_chunks:

        if not isinstance(chunk, dict):
            continue

        metadata = chunk.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        document_name = (
            chunk.get("document_name")
            or metadata.get("document_name")
            or metadata.get("source")
            or "Unknown document"
        )

        page_number = (
            chunk.get("page_number")
            or metadata.get("page_number")
            or metadata.get("page")
        )

        chunk_number = (
            chunk.get("chunk_number")
            or metadata.get("chunk_number")
            or metadata.get("chunk_id")
        )

        source = {
            "document_name": document_name,
            "page_number": page_number,
            "chunk_number": chunk_number
        }

        # Prevent duplicate sources
        if source not in sources:
            sources.append(source)

    return sources


def process_validation(
    validation_result: Any,
    original_answer: str
) -> tuple[bool, str]:
    """
    Process different validator return formats.

    The validator may return:
    - True or False
    - A dictionary
    - A validated answer string
    """

    if isinstance(validation_result, bool):
        return validation_result, original_answer

    if isinstance(validation_result, dict):

        is_valid = validation_result.get(
            "is_valid",
            validation_result.get("valid", True)
        )

        validated_answer = validation_result.get(
            "answer",
            validation_result.get(
                "validated_answer",
                original_answer
            )
        )

        return bool(is_valid), str(validated_answer)

    if isinstance(validation_result, str):
        return True, validation_result

    return True, original_answer


def run_assistant() -> None:
    """
    Run the complete AI Knowledge Assistant pipeline.
    """

    print("=" * 60)
    print("AI Knowledge Assistant")
    print("=" * 60)

    print("Ask questions using the documents in the knowledge base.")
    print("Type 'exit' to stop the program.")
    print()

    # History used by the prompt builder
    conversation_history: list[dict[str, str]] = []

    # Full conversation used by export.py
    export_history: list[dict[str, Any]] = []

    while True:

        question = input("You: ").strip()

        if not question:
            print("Please enter a question.\n")
            continue

        if question.lower() in {
            "exit",
            "quit",
            "stop"
        }:
            break

        try:
            # STEP 1:
            # Rewrite the user's question to improve retrieval
            rewritten_question = rewrite_query(
                question,
                conversation_history
            )

            print(
                f"\nRewritten query: {rewritten_question}"
            )

            # STEP 2:
            # Retrieve the most relevant document chunks
            retrieved_chunks = retrieve_relevant_chunks(
                rewritten_question,
                top_k=5
            )

            if not retrieved_chunks:
                unavailable_answer = (
                    "The requested information is not available "
                    "in the provided documents."
                )

                print(f"\nAssistant: {unavailable_answer}")
                print("Confidence: 0%\n")

                conversation_history.append(
                    {
                        "role": "user",
                        "content": question
                    }
                )

                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": unavailable_answer
                    }
                )

                export_history.append(
                    {
                        "question": question,
                        "answer": unavailable_answer,
                        "sources": [],
                        "confidence": 0
                    }
                )

                continue

            # STEP 3:
            # Build a clean context from retrieved chunks
            context = build_context(retrieved_chunks)

            # STEP 4:
            # Build the final prompt for Gemini
            prompt = build_prompt(
                question=question,
                context=context,
                history=conversation_history
            )

            # STEP 5:
            # Send the prompt to Gemini
            generated_answer = generate_answer(prompt)

            # STEP 6:
            # Validate the generated answer
            validation_result = validate_answer(
                answer=generated_answer,
                context=context
            )

            is_valid, final_answer = process_validation(
                validation_result,
                generated_answer
            )

            if not is_valid:
                final_answer = (
                    "The generated answer could not be validated "
                    "using the provided documents."
                )

            # STEP 7:
            # Prepare chunk text for evaluation
            chunk_texts = [
                get_chunk_text(chunk)
                for chunk in retrieved_chunks
            ]

            evaluation_result = evaluate_answer(
                question=question,
                answer=final_answer,
                retrieved_chunks=chunk_texts
            )

            confidence = evaluation_result.get(
                "confidence",
                0
            )

            # STEP 8:
            # Extract source metadata
            sources = extract_sources(
                retrieved_chunks
            )

            # Display the final result
            print("\nAssistant:")
            print(final_answer)

            print("\nSources:")

            if sources:
                for source in sources:

                    source_text = source["document_name"]

                    if source["page_number"] is not None:
                        source_text += (
                            f" - Page "
                            f"{source['page_number']}"
                        )

                    if source["chunk_number"] is not None:
                        source_text += (
                            f" - Chunk "
                            f"{source['chunk_number']}"
                        )

                    print(f"- {source_text}")

            else:
                print("- No sources available")

            print(f"\nConfidence: {confidence}%")
            print("-" * 60)

            # Save the question in conversation memory
            conversation_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            # Save the answer in conversation memory
            conversation_history.append(
                {
                    "role": "assistant",
                    "content": final_answer
                }
            )

            # Save the result for future export
            export_history.append(
                {
                    "question": question,
                    "answer": final_answer,
                    "sources": sources,
                    "confidence": confidence
                }
            )

        except Exception as error:
            print("\nAn error occurred while processing the question:")
            print(f"{type(error).__name__}: {error}\n")

    # Export the conversation after the user exits
    if export_history:

        print("\nWould you like to export the conversation?")

        export_choice = input(
            "Enter txt, md, or no: "
        ).strip().lower()

        if export_choice in {
            "txt",
            "md",
            "markdown"
        }:
            try:
                exported_path = export_conversation(
                    conversation=export_history,
                    export_format=export_choice
                )

                print(
                    f"Conversation exported successfully: "
                    f"{exported_path}"
                )

            except Exception as error:
                print(
                    f"Conversation export failed: {error}"
                )

        else:
            print("Conversation was not exported.")

    print("\nAI Knowledge Assistant stopped.")


if __name__ == "__main__":
    run_assistant()