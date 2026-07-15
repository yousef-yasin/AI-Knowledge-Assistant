
# vector_db.py

import chromadb  # to import the DB


class VectorDB:
    def __init__(self, persist_path: str = "./chroma_db",  # initialize the database, persist_path is the path where the database will be stored
                 # collection_name to specify the name of the collection in the database
                 collection_name: str = "knowledge_base"):
        # actual connection to the database, using PersistentClient to ensure data is saved to disk
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(  # if the collection with the specified name does not exist, it will be created
            name=collection_name,
            # metadata to specify the distance metric used for similarity search, in this case, cosine similarity
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def count(self) -> int:  # returns the number of documents in the collection
        return self.collection.count()


def query(
    self,
    query_embedding: list[float],
    n_results: int = 5,
    where: dict | None = None,
) -> dict:
    """
    Search the collection for the most similar embeddings.

    Args:
        query_embedding: The embedding vector to search against.
        n_results: How many results to return.
        where: Optional metadata filter, e.g. {"source": "file.pdf"}.

    Returns:
        Chroma's raw query response (documents, metadatas, distances).
    """
    return self.collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )
