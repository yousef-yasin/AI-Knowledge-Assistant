from src.retriever import RetrievalEngine


class FakeVectorDB:
    def query(self, query_embedding, n_results, where=None):
        assert query_embedding == [0.1, 0.2]
        assert n_results == 2

        return {
            "documents": [["First result", "Second result"]],
            "metadatas": [[
                {"document_name": "first.txt", "chunk_number": 1},
                {"document_name": "second.txt", "chunk_number": 2},
            ]],
            "distances": [[0.10, 0.25]],
        }


class FakeEmbeddingGenerator:
    def generate_embedding(self, text: str):
        assert text == "annual leave days"
        return [0.1, 0.2]


class FakeRewriteResult:
    rewritten = "annual leave days"


class FakeQueryRewriter:
    def rewrite(self, query: str):
        assert query == "How many leave days?"
        return FakeRewriteResult()


def test_retrieve_returns_ranked_chunks() -> None:
    retriever = RetrievalEngine(
        vector_db=FakeVectorDB(),
        embedding_generator=FakeEmbeddingGenerator(),
        query_rewriter=FakeQueryRewriter(),
    )

    results = retriever.retrieve("How many leave days?", top_k=2)

    assert len(results) == 2
    assert results[0].text == "First result"
    assert results[0].score == 0.90
    assert results[1].metadata["document_name"] == "second.txt"


def test_retrieve_can_skip_second_rewrite() -> None:
    retriever = RetrievalEngine(
        vector_db=FakeVectorDB(),
        embedding_generator=FakeEmbeddingGenerator(),
        query_rewriter=FakeQueryRewriter(),
    )

    results = retriever.retrieve(
        "annual leave days",
        top_k=2,
        rewrite_query=False,
    )

    assert len(results) == 2
