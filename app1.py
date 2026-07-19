from __future__ import annotations

from pathlib import Path

import streamlit as st


# -----------------------------------------
# Page configuration
# -----------------------------------------

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Knowledge Assistant")

st.write(
    "Upload your documents, process them, "
    "then ask questions about their content."
)

DATA_DIRECTORY = Path("data")

SUPPORTED_EXTENSIONS = [
    "pdf",
    "docx",
    "txt",
    "md",
    "csv",
]


# -----------------------------------------
# Session state
# -----------------------------------------

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "context_builder" not in st.session_state:
    st.session_state.context_builder = None

if "memory" not in st.session_state:
    st.session_state.memory = None


# -----------------------------------------
# Save uploaded files
# -----------------------------------------

def save_uploaded_files(uploaded_files) -> list[Path]:
    DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_files: list[Path] = []

    for uploaded_file in uploaded_files:
        file_name = Path(uploaded_file.name).name
        file_path = DATA_DIRECTORY / file_name

        file_path.write_bytes(
            uploaded_file.getbuffer()
        )

        saved_files.append(file_path)

    return saved_files


# -----------------------------------------
# Initialize project only when requested
# -----------------------------------------

def initialize_project(force_reindex: bool = True) -> int:
    # Imports are placed here intentionally.
    # They will not run while the page is opening.

    from src.citations import CitationManager
    from src.context_builder import ContextBuilder
    from src.embeddings import EmbeddingGenerator
    from src.main import index_knowledge_base
    from src.memory import ConversationMemory
    from src.query_rewriter import QueryRewriter
    from src.retriever import RetrievalEngine
    from src.vector_store import VectorDB

    vector_db = VectorDB(
        persist_path="./chroma_db",
        collection_name="knowledge_base",
    )

    embedding_generator = EmbeddingGenerator()
    citation_manager = CitationManager()
    query_rewriter = QueryRewriter()

    indexed_count = index_knowledge_base(
        vector_db=vector_db,
        force_reindex=force_reindex,
    )

    retriever = RetrievalEngine(
        vector_db=vector_db,
        embedding_generator=embedding_generator,
        citation_manager=citation_manager,
        query_rewriter=query_rewriter,
    )

    context_builder = ContextBuilder(
        max_tokens=3000,
        citation_manager=citation_manager,
    )

    st.session_state.vector_db = vector_db
    st.session_state.retriever = retriever
    st.session_state.context_builder = context_builder
    st.session_state.memory = ConversationMemory()
    st.session_state.system_ready = True

    return indexed_count


# -----------------------------------------
# Sidebar
# -----------------------------------------

with st.sidebar:
    st.header("📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
    )

    force_reindex = st.checkbox(
        "Rebuild knowledge base",
        value=True,
    )

    process_button = st.button(
        "Process Documents",
        type="primary",
        use_container_width=True,
    )

    if process_button:
        if not uploaded_files:
            st.warning(
                "Please upload at least one document."
            )

        else:
            try:
                saved_files = save_uploaded_files(
                    uploaded_files
                )

                st.info(
                    f"Saved {len(saved_files)} file(s)."
                )

                with st.spinner(
                    "Loading documents and generating embeddings..."
                ):
                    indexed_count = initialize_project(
                        force_reindex=force_reindex
                    )

                st.success(
                    "Documents processed successfully."
                )

                st.metric(
                    "Indexed chunks",
                    indexed_count,
                )

            except Exception as error:
                st.error(
                    f"Processing failed: {error}"
                )

    st.divider()

    if st.session_state.system_ready:
        st.success("Knowledge base is ready.")
    else:
        st.warning("Upload and process documents first.")


# -----------------------------------------
# Chat history
# -----------------------------------------

st.subheader("💬 Ask Your Documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------------------
# Question processing
# -----------------------------------------

question = st.chat_input(
    "Ask a question about the uploaded documents..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not st.session_state.system_ready:
            answer = (
                "Please upload and process documents "
                "before asking a question."
            )

            st.warning(answer)

        else:
            try:
                with st.spinner(
                    "Searching the documents..."
                ):
                    from src.generator import generate_answer
                    from src.validator import validate_answer

                    retrieved_chunks = (
                        st.session_state.retriever.retrieve(
                            query=question,
                            top_k=3,
                            rewrite_query=True,
                            use_hybrid=True,
                            use_multi_query=True,
                        )
                    )

                    context_result = (
                        st.session_state.context_builder.build_context(
                            retrieved_chunks,
                            min_score=0.20,
                        )
                    )

                    if not context_result.context.strip():
                        answer = (
                            "The requested information was not "
                            "found in the uploaded documents."
                        )

                        st.warning(answer)

                    else:
                        history = (
                            st.session_state.memory.get_history()
                        )

                        generated_answer = generate_answer(
                            question=question,
                            context=context_result.context,
                            history=history,
                            strategy="advanced",
                        )

                        validation = validate_answer(
                            question=question,
                            context=context_result.context,
                            answer=generated_answer,
                        )

                        validation_status = str(
                            validation.get("status", "")
                        ).upper()

                        supported = bool(
                            validation.get(
                                "supported",
                                False,
                            )
                        )

                        if (
                            validation_status == "PASS"
                            and supported
                        ):
                            answer = generated_answer
                            st.markdown(answer)

                        else:
                            reason = validation.get(
                                "reason",
                                "The answer could not be validated.",
                            )

                            answer = (
                                "The answer could not be validated "
                                "using the uploaded documents.\n\n"
                                f"Reason: {reason}"
                            )

                            st.warning(answer)

                        if context_result.citations:
                            with st.expander("📚 Sources"):
                                for index, citation in enumerate(
                                    context_result.citations,
                                    start=1,
                                ):
                                    st.write(
                                        f"{index}. "
                                        f"{citation.document_name} "
                                        f"— Page: "
                                        f"{citation.page_number or 'N/A'}"
                                    )

                        st.session_state.memory.add_user_message(
                            question
                        )

                        st.session_state.memory.add_assistant_message(
                            answer
                        )

            except Exception as error:
                answer = f"Error: {error}"
                st.error(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )