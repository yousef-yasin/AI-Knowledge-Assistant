from src.decision_agent import RetrievalDecisionAgent
from src.query_rewriter import QueryRewriter
from src.summarizer import DocumentSummarizer
from src.keyword_search import BM25Index


def test_multi_query_generates_unique_variants():
    queries = QueryRewriter().generate_multi_queries(
        "How many annual leave days do employees receive?"
    )
    assert 1 <= len(queries) <= 3
    assert len(queries) == len(set(queries))


def test_decision_agent_uses_memory_for_pronoun_follow_up():
    history = [{"role": "assistant", "content": "Employees receive 21 days."}]
    assert RetrievalDecisionAgent().decide("What about that?", history) == "memory"


def test_decision_agent_retrieves_company_policy():
    history = [{"role": "assistant", "content": "Hello."}]
    assert RetrievalDecisionAgent().decide("What is the annual leave policy?", history) == "retrieval"


def test_extractive_summary_is_shorter():
    text = " ".join([
        "Employees receive annual leave.",
        "The company operates in Amman.",
        "Employees receive annual leave every year.",
        "Security rules must be followed.",
        "Annual leave requires approval.",
        "Equipment must be returned.",
    ] * 10)
    summary = DocumentSummarizer(max_sentences=3, max_characters=300).summarize_text(text)
    assert summary
    assert len(summary) <= 300
    assert len(summary) < len(text)


def test_bm25_keyword_search():
    index = BM25Index()
    index.build(
        ["1", "2"],
        ["annual leave policy twenty one days", "information security passwords"],
        [{"document_name": "leave.md"}, {"document_name": "security.md"}],
    )
    result = index.search("annual leave", top_k=1)
    assert result
    assert result[0].chunk_id == "1"
