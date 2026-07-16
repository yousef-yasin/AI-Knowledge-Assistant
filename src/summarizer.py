from __future__ import annotations
# This code isn't linked to Gemini .
from dataclasses import dataclass  # Creates a simple class for storing the document summary.
from typing import Any, Callable, Iterable  # Defines accepted data and function types.


# Defines the expected format of the text generation function.
# It receives a prompt as text and returns the generated response as text.
TextGenerator = Callable[[str], str]


@dataclass
class SummarizedDocument:
    """Stores the original document text, its summary, and metadata."""

    text: str  # Stores the original document text.
    summary: str  # Stores the generated document summary.
    metadata: dict[str, Any]  # Stores the document information.


class DocumentSummarizer:
    """Generates concise summaries for documents during indexing."""

    def __init__(
        self,
        text_generator: TextGenerator,
        max_input_characters: int = 12000,
        minimum_text_length: int = 50,
    ) -> None:
        """
        Initialize the document summarizer.

        Args:
            text_generator:
                A function that sends a prompt to an AI model
                and returns the generated response.

            max_input_characters:
                Maximum number of document characters sent to the AI model.

            minimum_text_length:
                Minimum text length required before generating a summary.
        """

        if not callable(text_generator):
            raise TypeError("text_generator must be a callable function.")

        if max_input_characters < 1:
            raise ValueError(
                "max_input_characters must be greater than 0."
            )

        if minimum_text_length < 1:
            raise ValueError(
                "minimum_text_length must be greater than 0."
            )

        self.text_generator = text_generator
        # Stores the function that will generate the summary.

        self.max_input_characters = max_input_characters
        # Limits the amount of text sent to the AI model.

        self.minimum_text_length = minimum_text_length
        # Prevents unnecessary summarization of very short texts.

    def build_summary_prompt(
        self,
        document_text: str,
    ) -> str:
        """
        Build the prompt used to summarize a document.

        Args:
            document_text: The document text that will be summarized.

        Returns:
            A formatted prompt ready to be sent to the AI model.
        """

        return (
            "You are a document summarization assistant.\n\n"
            "Create a concise and accurate summary of the following "
            "document.\n\n"
            "Rules:\n"
            "- Use only information found in the document.\n"
            "- Do not add unsupported information.\n"
            "- Include the main topic and the most important points.\n"
            "- Keep the summary clear and professional.\n"
            "- Return only the summary.\n\n"
            f"Document:\n{document_text}"
        )

    def summarize_text(
        self,
        text: str,
    ) -> str:
        """
        Generate a summary for one text.

        Args:
            text: The document text.

        Returns:
            The generated summary.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        cleaned_text = text.strip()
        # Removes unnecessary spaces from the beginning and end.

        if not cleaned_text:
            raise ValueError("Cannot summarize empty text.")

        if len(cleaned_text) < self.minimum_text_length:
            return cleaned_text
            # Very short text is returned directly because it does not
            # need an AI-generated summary.

        limited_text = cleaned_text[: self.max_input_characters]
        # Limits long documents before sending them to the AI model.

        prompt = self.build_summary_prompt(limited_text)
        # Creates the summarization prompt.

        generated_summary = self.text_generator(prompt)
        # Sends the prompt to the provided AI generation function.

        if not isinstance(generated_summary, str):
            raise TypeError(
                "text_generator must return the generated text as a string."
            )

        summary = generated_summary.strip()
        # Removes unnecessary spaces from the generated response.

        if not summary:
            raise ValueError(
                "The text generator returned an empty summary."
            )

        return summary

    def summarize_document(
        self,
        text: str,
        metadata: dict[str, Any],
    ) -> SummarizedDocument:
        """
        Generate a summary for one document while preserving its metadata.

        Args:
            text: The original document text.
            metadata: The document metadata.

        Returns:
            A SummarizedDocument object.
        """

        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        summary = self.summarize_text(text)
        # Generates the summary from the original document text.

        updated_metadata = metadata.copy()
        # Creates a copy to avoid modifying the original metadata.

        updated_metadata["summary"] = summary
        # Adds the generated summary to the document metadata.

        updated_metadata["is_summarized"] = True
        # Indicates that the document was summarized successfully.

        return SummarizedDocument(
            text=text,
            summary=summary,
            metadata=updated_metadata,
        )

    def summarize_documents(
        self,
        documents: Iterable[Any],
    ) -> list[SummarizedDocument]:
        """
        Generate summaries for multiple loaded documents.

        Each document must contain:
        - document.text
        - document.metadata

        Args:
            documents: Loaded documents returned by the document loader.

        Returns:
            A list of summarized documents.
        """

        summarized_documents: list[SummarizedDocument] = []

        for document in documents:
            if not hasattr(document, "text"):
                raise AttributeError(
                    "Each document must contain a text attribute."
                )

            if not hasattr(document, "metadata"):
                raise AttributeError(
                    "Each document must contain a metadata attribute."
                )

            summarized_document = self.summarize_document(
                text=document.text,
                metadata=document.metadata,
            )
            # Generates a summary for the current loaded document.

            summarized_documents.append(summarized_document)
            # Adds the result to the final list.

        return summarized_documents


def summarize_documents(
    documents: Iterable[Any],
    text_generator: TextGenerator,
    max_input_characters: int = 12000,
    minimum_text_length: int = 50,
) -> list[SummarizedDocument]:
    """
    Helper function for summarizing multiple documents in one line.

    Example:
        summarized_documents = summarize_documents(
            documents=loaded_documents,
            text_generator=generate_text,
        )
    """

    summarizer = DocumentSummarizer(
        text_generator=text_generator,
        max_input_characters=max_input_characters,
        minimum_text_length=minimum_text_length,
    )
    # Creates the document summarizer.

    return summarizer.summarize_documents(documents)
    # Summarizes all received documents.