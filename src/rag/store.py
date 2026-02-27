from langchain_community.vectorstores import Chroma
from rag.embedder import get_embedder
from rag.config import CHROMA_PATH, CHROMA_COLLECTION

class VectorStore:
    def __init__(self):
        self.embedder = get_embedder()
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embedder,
            collection_name=CHROMA_COLLECTION
        )

    def add_documents(self, texts, metadatas):
        self.vectorstore.add_texts(texts, metadatas=metadatas)
        self.vectorstore.persist()

    def search(self, query, k=5):
        return self.vectorstore.similarity_search(query, k=k)
