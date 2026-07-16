"""
main.py

Final integration file for the AI Knowledge Assistant.
"""

from pathlib import Path
from typing import Any

from src.citations import Citation, CitationManager
from src.context_builder import ContextBuilder
from src.embeddings import EmbeddingGenerator
from src.export import export_conversation
from src.generator import generate_answer
from src.knowledge_base.chunker import DocumentChunker
from src.knowledge_base.loader import DocumentLoader
from src.memory import ConversationMemory
from src.query_rewriter import QueryRewriter
from src.retriever import RetrievalEngine
from src.validator import validate_answer
from src.vector_store import VectorDB


DATA_DIRECTORY = Path("data")
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
}


def index_knowledge_base(
    vector_db: VectorDB,
    force_reindex: bool = False,
) -> int:
    """
    Load, chunk, embed, and store all supported files from data/.

    The current implementation avoids duplicate indexing by skipping
    indexing when the vector database already contains chunks.

    Args:
        vector_db:
            Vector database instance.

        force_reindex:
            Reserved for future full re-index support.

    Returns:
        Number of indexed chunks.
    """

    existing_count = vector_db.count()

    if existing_count > 0 and not force_reindex:
        print(
            f"Knowledge base already contains "
            f"{existing_count} indexed chunks."
        )
        return existing_count

    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Knowledge base directory was not found: "
            f"{DATA_DIRECTORY.resolve()}"
        )

    document_paths = sorted(
        path
        for path in DATA_DIRECTORY.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not document_paths:
        raise FileNotFoundError(
            "No supported documents were found inside data/."
        )

    loader = DocumentLoader()
    chunker = DocumentChunker(
        chunk_size=800,
        chunk_overlap=100,
    )
    embedding_generator = EmbeddingGenerator()

    all_documents = []

    print("\nIndexing knowledge base...")
    print("-" * 60)

    for document_path in document_paths:
        try:
            loaded_documents = loader.load(document_path)
            all_documents.extend(loaded_documents)

            print(
                f"Loaded: {document_path.name} "
                f"({len(loaded_documents)} document part(s))"
            )

        except Exception as error:
            print(
                f"Skipped {document_path.name}: {error}"
            )

    if not all_documents:
        raise RuntimeError(
            "No readable documents could be loaded."
        )

    chunks = chunker.chunk_documents(all_documents)

    if not chunks:
        raise RuntimeError(
            "No chunks were generated from the documents."
        )

    print(f"Generated chunks: {len(chunks)}")
    print("Generating embeddings...")

    embedded_chunks = embedding_generator.embed_chunks(chunks)

    if not embedded_chunks:
        raise RuntimeError(
            "No embeddings were generated."
        )

    vector_db.add_embedded_chunks(embedded_chunks)

    indexed_count = vector_db.count()

    print(
        f"Knowledge base indexed successfully. "
        f"Total stored chunks: {indexed_count}"
    )

    return indexed_count


def citation_to_source(
    citation: Citation,
) -> dict[str, Any]:
    """
    Convert a Citation object into the dictionary format
    required by export.py.
    """

    return {
        "document_name": citation.document_name,
        "page_number": citation.page_number,
        "chunk_number": citation.chunk_number,
    }


def calculate_overall_confidence(
    citations: list[Citation],
) -> float:
    """
    Calculate the average citation confidence.
    """

    return CitationManager.calculate_overall_confidence(
        citations
    )


def validation_passed(
    validation_result: dict[str, Any],
) -> bool:
    """
    Check whether validator.py approved the answer.
    """

    status = str(
        validation_result.get("status", "")
    ).upper()

    supported = bool(
        validation_result.get("supported", False)
    )

    return status == "PASS" and supported


def print_sources(
    citations: list[Citation],
) -> None:
    """
    Print formatted source citations.
    """

    print("\nSources:")

    if not citations:
        print("- No supporting sources were found.")
        return

    citation_manager = CitationManager()

    for index, citation in enumerate(
        citations,
        start=1,
    ):
        formatted = citation_manager.format_citation(
            citation
        )

        print(f"{index}. {formatted}")


def run_assistant() -> None:
    """
    Run the complete AI Knowledge Assistant pipeline.
    """

    print("=" * 60)
    print("AI Knowledge Assistant")
    print("=" * 60)

    vector_db = VectorDB(
        persist_path="./chroma_db",
        collection_name="knowledge_base",
    )

    try:
        index_knowledge_base(vector_db)

    except Exception as error:
        print(f"\nIndexing failed: {error}")
        return

    embedding_generator = EmbeddingGenerator()
    query_rewriter = QueryRewriter()
    citation_manager = CitationManager()

    retrieval_engine = RetrievalEngine(
        vector_db=vector_db,
        embedding_generator=embedding_generator,
        citation_manager=citation_manager,
        query_rewriter=query_rewriter,
    )

    context_builder = ContextBuilder(
        max_tokens=3000,
        citation_manager=citation_manager,
    )

    memory = ConversationMemory()

    export_history: list[dict[str, Any]] = []

    print("\nAsk questions using the indexed documents.")
    print("Commands:")
    print("- exit: stop the program")
    print("- export: export the conversation")
    print("- clear: clear conversation memory")
    print()

    while True:
        question = input("You: ").strip()

        if not question:
            print("Please enter a question.\n")
            continue

        command = question.lower()

        if command in {
            "exit",
            "quit",
            "stop",
        }:
            break

        if command == "clear":
            memory.clear()
            export_history.clear()

            print(
                "\nConversation memory cleared.\n"
            )
            continue

        if command == "export":
            if not export_history:
                print(
                    "\nThere is no conversation to export.\n"
                )
                continue

            try:
                exported_files = export_conversation(
                    export_history
                )

                print("\nConversation exported successfully.")

                if isinstance(exported_files, dict):
                    for file_type, file_path in (
                        exported_files.items()
                    ):
                        print(
                            f"{file_type.upper()}: "
                            f"{file_path}"
                        )
                else:
                    print(exported_files)

                print()

            except Exception as error:
                print(
                    f"\nConversation export failed: "
                    f"{error}\n"
                )

            continue

        try:
            rewritten_result = query_rewriter.rewrite(
                question
            )

            rewritten_question = (
                rewritten_result.rewritten
            )

            print(
                f"\nRewritten query: "
                f"{rewritten_question}"
            )

            retrieved_chunks = (
                retrieval_engine.retrieve(
                rewritten_question,
                top_k=3,
                rewrite_query=False,
                )
            )

            if not retrieved_chunks:
                unavailable_answer = (
                    "The requested information is not "
                    "available in the retrieved documents."
                )

                print(
                    f"\nAssistant: "
                    f"{unavailable_answer}"
                )
                print("Confidence: 0%\n")

                memory.add_user_message(question)
                memory.add_assistant_message(
                    unavailable_answer
                )

                export_history.append(
                    {
                        "question": question,
                        "answer": unavailable_answer,
                        "sources": [],
                        "confidence": 0,
                    }
                )

                continue

            context_result = (
                context_builder.build_context(
                    retrieved_chunks,
                    min_score=0.20,
                )
            )

            if not context_result.context.strip():
                unavailable_answer = (
                    "The requested information is not "
                    "available in the retrieved documents."
                )

                print(
                    f"\nAssistant: "
                    f"{unavailable_answer}"
                )
                print("Confidence: 0%\n")

                memory.add_user_message(question)
                memory.add_assistant_message(
                    unavailable_answer
                )

                export_history.append(
                    {
                        "question": question,
                        "answer": unavailable_answer,
                        "sources": [],
                        "confidence": 0,
                    }
                )

                continue

            history = memory.get_history()

            generated_answer = generate_answer(
                question=question,
                context=context_result.context,
                history=history,
                strategy="advanced",
            )

            if generated_answer.startswith(
                "Error generating"
            ):
                print(
                    f"\nAssistant: "
                    f"{generated_answer}\n"
                )
                continue

            validation_result = validate_answer(
                question=question,
                context=context_result.context,
                answer=generated_answer,
            )

            if validation_passed(
                validation_result
            ):
                final_answer = generated_answer.strip()

            else:
                validation_reason = (
                    validation_result.get(
                        "reason",
                        "The answer could not be validated.",
                    )
                )

                final_answer = (
                    "The generated answer could not be "
                    "validated using the retrieved documents. "
                    "Therefore, the answer was rejected.\n\n"
                    f"Reason: {validation_reason}"
                )

            confidence = (
                calculate_overall_confidence(
                    context_result.citations
                )
            )

            print(
                f"\nAssistant: {final_answer}"
            )

            print_sources(
                context_result.citations
            )

            print(
                f"\nOverall Confidence: "
                f"{confidence:.2f}%"
            )

            print(
                f"Chunks used: "
                f"{context_result.chunks_used}"
            )

            print(
                f"Chunks dropped: "
                f"{context_result.chunks_dropped}\n"
            )

            memory.add_user_message(question)
            memory.add_assistant_message(
                final_answer
            )

            export_history.append(
                {
                    "question": question,
                    "answer": final_answer,
                    "sources": [
                        citation_to_source(citation)
                        for citation
                        in context_result.citations
                    ],
                    "confidence": confidence,
                }
            )

        except Exception as error:
            print(
                f"\nAn error occurred: {error}\n"
            )

    if export_history:
        export_choice = input(
            "\nExport conversation before exiting? "
            "(y/n): "
        ).strip().lower()

        if export_choice in {
            "y",
            "yes",
        }:
            try:
                exported_files = export_conversation(
                    export_history
                )

                print(
                    "\nConversation exported successfully."
                )

                if isinstance(exported_files, dict):
                    for file_type, file_path in (
                        exported_files.items()
                    ):
                        print(
                            f"{file_type.upper()}: "
                            f"{file_path}"
                        )
                else:
                    print(exported_files)

            except Exception as error:
                print(
                    f"Conversation export failed: "
                    f"{error}"
                )

    print("\nGoodbye.")


if __name__ == "__main__":
    run_assistant()