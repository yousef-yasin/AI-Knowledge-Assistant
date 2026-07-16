"""Final integration for the AI Knowledge Assistant and all bonus features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.citations import Citation, CitationManager
from src.context_builder import ContextBuilder
from src.decision_agent import RetrievalDecisionAgent
from src.embeddings import EmbeddingGenerator
from src.export import export_conversation
from src.generator import generate_answer, generate_memory_answer, stream_generate_answer
from src.knowledge_base.chunker import DocumentChunk, DocumentChunker
from src.knowledge_base.loader import DocumentLoader
from src.memory import ConversationMemory
from src.query_rewriter import QueryRewriter
from src.retriever import RetrievalEngine
from src.summarizer import DocumentSummarizer
from src.validator import validate_answer
from src.vector_store import VectorDB

DATA_DIRECTORY = Path("data")
TOP_K = 3
MIN_RETRIEVAL_SCORE = 0.20
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def index_knowledge_base(vector_db: VectorDB, force_reindex: bool = False) -> int:
    existing_count = vector_db.count()
    if existing_count > 0 and not force_reindex:
        print(f"Knowledge base already contains {existing_count} indexed chunks.")
        return existing_count
    if force_reindex and existing_count:
        vector_db.reset()
    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError(f"Knowledge base directory was not found: {DATA_DIRECTORY.resolve()}")

    paths = sorted(path for path in DATA_DIRECTORY.iterdir()
                   if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise FileNotFoundError("No supported documents were found inside data/.")

    loader = DocumentLoader(enable_ocr=True)
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
    summarizer = DocumentSummarizer(max_sentences=4, max_characters=1200)
    embedding_generator = EmbeddingGenerator()
    loaded_documents = []

    print("\nIndexing knowledge base...")
    print("-" * 60)
    for path in paths:
        try:
            documents = loader.load(path)
            loaded_documents.extend(documents)
            methods = sorted({str(doc.metadata.get("extraction_method", "standard")) for doc in documents})
            print(f"Loaded: {path.name} ({len(documents)} part(s), extraction: {', '.join(methods)})")
        except Exception as error:
            print(f"Skipped {path.name}: {error}")

    if not loaded_documents:
        raise RuntimeError("No readable documents could be loaded.")

    chunks = chunker.chunk_documents(loaded_documents)
    summary_chunks: list[DocumentChunk] = []
    for summary in summarizer.summarize_documents(loaded_documents):
        summary_chunks.append(DocumentChunk(text=summary.summary, metadata=summary.metadata))
    all_chunks = chunks + summary_chunks
    print(f"Generated chunks: {len(chunks)}")
    print(f"Generated document summaries: {len(summary_chunks)}")
    print("Generating embeddings...")
    vector_db.add_embedded_chunks(embedding_generator.embed_chunks(all_chunks))
    print(f"Knowledge base indexed successfully. Total stored chunks: {vector_db.count()}")
    return vector_db.count()


def citation_to_source(citation: Citation) -> dict[str, Any]:
    return {
        "document_name": citation.document_name,
        "page_number": citation.page_number,
        "chunk_number": citation.chunk_number,
    }


def validation_passed(result: dict[str, Any]) -> bool:
    return str(result.get("status", "")).upper() == "PASS" and bool(result.get("supported", False))


def calculate_overall_confidence(citations: list[Citation], validation: dict[str, Any]) -> float:
    return CitationManager.calculate_answer_confidence(
        citations=citations,
        validation_supported=validation_passed(validation),
    )


def print_sources(citations: list[Citation]) -> None:
    print("\nSources:")
    if not citations:
        print("- Answered from conversation memory; document retrieval was not required.")
        return
    manager = CitationManager()
    for index, citation in enumerate(citations, 1):
        print(f"{index}. {manager.format_citation(citation)}")


def run_assistant() -> None:
    print("=" * 60)
    print("AI Knowledge Assistant — Full Bonus Edition")
    print("=" * 60)

    vector_db = VectorDB(persist_path="./chroma_db", collection_name="knowledge_base")
    try:
        index_knowledge_base(vector_db)
    except Exception as error:
        print(f"\nIndexing failed: {error}")
        return

    embeddings = EmbeddingGenerator()
    rewriter = QueryRewriter()
    citations = CitationManager()
    retriever = RetrievalEngine(vector_db, embeddings, citations, rewriter)
    context_builder = ContextBuilder(max_tokens=3000, citation_manager=citations)
    agent = RetrievalDecisionAgent()
    memory = ConversationMemory()
    export_history: list[dict[str, Any]] = []
    streaming_enabled = True

    print("\nAsk questions using the indexed documents.")
    print("Bonus features: Hybrid Search, Multi-query Retrieval, OCR, Summaries, Agent, Streaming")
    print("Commands:")
    print("- exit: stop the program")
    print("- export: export the conversation")
    print("- clear: clear conversation memory")
    print("- reindex: rebuild the index, including OCR and summaries")
    print("- stream: toggle streaming responses")
    print()

    while True:
        question = input("You: ").strip()
        if not question:
            print("Please enter a question.\n")
            continue
        command = question.lower()
        if command in {"exit", "quit", "stop"}:
            break
        if command == "clear":
            memory.clear(); export_history.clear()
            print("\nConversation memory cleared.\n")
            continue
        if command == "stream":
            streaming_enabled = not streaming_enabled
            print(f"\nStreaming responses: {'ON' if streaming_enabled else 'OFF'}\n")
            continue
        if command == "reindex":
            try:
                index_knowledge_base(vector_db, force_reindex=True)
                retriever.refresh_keyword_index()
                print("\nRe-indexing completed. Hybrid keyword index refreshed.\n")
            except Exception as error:
                print(f"\nRe-indexing failed: {error}\n")
            continue
        if command == "export":
            if not export_history:
                print("\nThere is no conversation to export.\n")
                continue
            files = export_conversation(export_history)
            print("\nConversation exported successfully.")
            if isinstance(files, dict):
                for kind, path in files.items():
                    print(f"{kind.upper()}: {path}")
            print()
            continue

        try:
            history = memory.get_history()
            route = agent.decide(question, history)
            print(f"\nAgent route: {'Knowledge Base Retrieval' if route == 'retrieval' else 'Conversation Memory'}")

            if route == "memory":
                final_answer = generate_memory_answer(question, history)
                current_citations: list[Citation] = []
                confidence = 90.0 if not final_answer.startswith("Error") else 0.0
                chunks_used = 0
                chunks_dropped = 0
            else:
                rewritten = rewriter.rewrite(question).rewritten
                variants = rewriter.generate_multi_queries(rewritten)
                print(f"Rewritten query: {rewritten}")
                print(f"Multi-query variants: {len(variants)}")
                retrieved = retriever.retrieve(
                    rewritten, top_k=TOP_K, rewrite_query=False,
                    use_hybrid=True, use_multi_query=True,
                )
                context = context_builder.build_context(retrieved, min_score=MIN_RETRIEVAL_SCORE)
                if not context.context.strip():
                    final_answer = "The requested information is not available in the retrieved documents."
                    current_citations = []
                    confidence = 0.0
                    chunks_used = 0
                    chunks_dropped = len(retrieved)
                else:
                    if streaming_enabled:
                        print("\nAssistant: ", end="", flush=True)
                        parts: list[str] = []
                        for part in stream_generate_answer(question, context.context, history, "advanced"):
                            print(part, end="", flush=True)
                            parts.append(part)
                        print()
                        generated = "".join(parts).strip()
                    else:
                        generated = generate_answer(question, context.context, history, "advanced")
                        print(f"\nAssistant: {generated}")
                    if generated.startswith(("Error generating response:", "Error generating text:")):
                        final_answer = generated
                        validation = {
                            "status": "FAIL",
                            "supported": False,
                            "reason": "Answer generation failed before validation.",
                        }
                        current_citations = context.citations
                        confidence = 0.0
                    else:
                        validation = validate_answer(
                            question,
                            context.context,
                            generated,
                        )
                        if validation_passed(validation):
                            final_answer = generated
                        else:
                            final_answer = (
                                "The generated answer could not be validated using the retrieved documents. "
                                f"Reason: {validation.get('reason', 'Validation failed.')}"
                            )
                        current_citations = context.citations
                        confidence = calculate_overall_confidence(
                            current_citations,
                            validation,
                        )
                    chunks_used = context.chunks_used
                    chunks_dropped = context.chunks_dropped

            if route == "memory" or not streaming_enabled:
                if route == "memory":
                    print(f"\nAssistant: {final_answer}")
            elif route == "retrieval" and 'generated' in locals() and final_answer != generated:
                print(f"Assistant validation result: {final_answer}")

            print_sources(current_citations)
            print(f"\nOverall Confidence: {confidence:.2f}%")
            print(f"Chunks used: {chunks_used}")
            print(f"Chunks dropped: {chunks_dropped}\n")

            memory.add_user_message(question)
            memory.add_assistant_message(final_answer)
            export_history.append({
                "question": question,
                "answer": final_answer,
                "sources": [citation_to_source(c) for c in current_citations],
                "confidence": confidence,
                "route": route,
            })
        except Exception as error:
            print(f"\nAn error occurred: {error}\n")

    print("\nGoodbye.")


if __name__ == "__main__":
    run_assistant()
