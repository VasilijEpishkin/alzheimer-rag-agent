from typing import List
from langchain_community.llms import HuggingFaceHub
from rag.config import LLM_MODEL
import os

class AnswerGenerator:
    def __init__(self):
        self.llm = HuggingFaceHub(
            repo_id=LLM_MODEL,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

    def generate(self, query: str, contexts: List[str]) -> str:
        prompt = self._build_prompt(query, contexts)
        return self.llm(prompt)

    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        context_str = "\n\n".join(contexts)
        return f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
