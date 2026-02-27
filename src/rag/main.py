from rag.retriever import PubMedRetriever
from rag.generator import AnswerGenerator
from rag.logger import logger

class AlzheimerRAG:
    def __init__(self):
        self.retriever = PubMedRetriever()
        self.generator = AnswerGenerator()

    def query(self, text: str):
        logger.info(f"Processing query: {text}")
        docs = self.retriever.retrieve(text)
        context = [doc.page_content for doc in docs]
        answer = self.generator.generate(text, context)
        
        return {
            "answer": answer,
            "source_documents": docs
        }
