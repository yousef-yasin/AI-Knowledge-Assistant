
# vector_db.py

import chromadb # to import the DB

class VectorDB:
    def __init__(self, persist_path: str = "./chroma_db",   #initialize the database, persist_path is the path where the database will be stored 
    collection_name: str = "knowledge_base"):               #collection_name to specify the name of the collection in the database
        self.client = chromadb.PersistentClient(path=persist_path) #actual connection to the database, using PersistentClient to ensure data is saved to disk
        self.collection = self.client.get_or_create_collection( #if the collection with the specified name does not exist, it will be created
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # metadata to specify the distance metric used for similarity search, in this case, cosine similarity
        )

    def add_documents(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def count(self) -> int: # returns the number of documents in the collection
        return self.collection.count()




