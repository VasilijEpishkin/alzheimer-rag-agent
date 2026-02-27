from rag.store import VectorStore

class PubMedRetriever:
    def __init__(self):
        self.store = VectorStore()

    def retrieve(self, query: str, k: int = 5):
        return self.store.search(query, k)
