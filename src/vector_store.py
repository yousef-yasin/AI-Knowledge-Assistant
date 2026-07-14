
# vector_db.py

import chromadb
from chromadb.config import Settings

class VectorDB:
    def __init__(self, persist_path: str = "./chroma_db", collection_name: str = "knowledge_base"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine similarity, matches your TASK1 approach
        )

    def add_documents(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def count(self) -> int:
        return self.collection.count()