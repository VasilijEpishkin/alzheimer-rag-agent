"""Generation module: OpenRouter LLM with citation verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import requests

from evidence_rag.models import Document
from evidence_rag.config import Settings


class OpenRouterClient:
    """Simple client for OpenRouter API."""

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/your-repo/evidence-rag",  # Optional, but recommended.
                "X-Title": "Evidence RAG",
            }
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate text from OpenRouter API."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        response = self.session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def close(self) -> None:
        self.session.close()


def verify_citations(
    answer: str, retrieved_documents: List[Document]
) -> List[str]:
    """Verify that every [PMID] citation in the answer is in the retrieved documents.
    Returns a list of unverified PMIDs (empty if all verified).
    """
    import re

    # Extract PMIDs from answer in the format [PMID]
    pmid_pattern = re.compile(r'\[(\d+)\]')
    cited_pmids = set(pmid_pattern.findall(answer))

    retrieved_pmids = {doc.doc_id for doc in retrieved_documents}

    unverified = [pmid for pmid in cited_pmids if pmid not in retrieved_pmids]
    return unverified


def build_generation_prompt(
    query: str,
    retrieved_documents: List[Document],
    stance_outputs: List[Any],
    answerability_prob: float,
) -> str:
    """Build a prompt for the LLM that includes context and instructions for citation."""
    # We'll create a simple prompt that includes the retrieved abstracts and instructions.
    # In a real system, we might format the context more structured.
    context_parts = []
    for i, doc in enumerate(retrieved_documents):
        context_parts.append(
            f"[PMID:{doc.doc_id}] Title: {doc.title}\nAbstract: {doc.text}\n"
        )
    context = "\n---\n".join(context_parts)

    # We can also include stance and answerability info if desired, but for simplicity we omit.
    # The instructions: answer the question, cite every claim with [PMID], and only use the provided context.
    prompt = f"""Answer the following question based on the provided context.
    Do not use any external knowledge. If the context does not contain enough information to answer, say so.
    Every factual claim in your answer must be accompanied by a citation in the format [PMID] where the PMID is from the context.
    Do not cite any PMID that is not in the context.

    Question: {query}

    Context:
    {context}

    Answer:"""
    return prompt


def generate_answer(
    query: str,
    retrieved_documents: List[Document],
    stance_outputs: List[Any],
    answerability_prob: float,
    settings: Settings,
) -> str:
    """Generate an answer using OpenRouter and verify citations.
    If answerability_prob is below a threshold, we abstain.
    """
    # Abstention threshold: we can make this configurable.
    abstention_threshold = 0.5
    if answerability_prob < abstention_threshold:
        return "I do not have enough evidence to answer this question."

    # If no OpenRouter API key, we cannot generate. In production, we should error.
    # For now, we return a placeholder to avoid fabricating data.
    if not settings.openrouter_api_key:
        # In a real system, we might raise an error or return a message.
        # We choose to return a message indicating the missing key.
        return "OpenRouter API key not configured. Cannot generate answer."

    client = OpenRouterClient(api_key=settings.openrouter_api_key)
    try:
        prompt = build_generation_prompt(
            query, retrieved_documents, stance_outputs, answerability_prob
        )
        answer = client.generate(prompt)
        # Verify citations
        unverified = verify_citations(answer, retrieved_documents)
        if unverified:
            # In a real system, we might want to flag or regenerate.
            # For now, we append a warning.
            answer += f"\n\nWarning: The following PMIDs were cited but not found in the retrieved context: {', '.join(unverified)}"
        return answer
    finally:
        client.close()


__all__ = [
    "OpenRouterClient",
    "verify_citations",
    "build_generation_prompt",
    "generate_answer",
]