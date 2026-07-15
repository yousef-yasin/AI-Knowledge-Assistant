from src.vector_store import VectorDB
from src.retriever import RetrievalEngine

# Connect to your existing persisted DB (same path used when you built it)
db = VectorDB(persist_path="./chroma_db", collection_name="knowledge_base")

print(f"Documents in collection: {db.count()}")

retriever = RetrievalEngine(vector_db=db)

# <- replace with something relevant to your actual docs
query = "your test question here"
results = retriever.retrieve(query, top_k=3)

print(f"\nQuery: {query}\n")
for i, chunk in enumerate(results, start=1):
    print(f"--- Result {i} (score: {chunk.score:.4f}) ---")
    print(chunk.text[:200])
    print(f"metadata: {chunk.metadata}")
    print()
