"""Generation module: OpenRouter LLM with citation verifier."""

from __future__ import annotations

from typing import Any

import requests

from evidence_rag.config import Settings
from evidence_rag.models import Document


class OpenRouterClient:
    """Simple client for OpenRouter API."""

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/VasilijEpishkin/alzheimer-rag-agent",
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
    answer: str, retrieved_documents: list[Document]
) -> list[str]:
    """Verify that each [PMID:123] citation is in the retrieved documents.
    Returns a list of unverified PMIDs (empty if all verified).
    """
    import re

    # Extract PMIDs from answer in the format [PMID]
    pmid_pattern = re.compile(r"\[(?:PMID:)?(\d+)\]")
    cited_pmids = set(pmid_pattern.findall(answer))

    retrieved_pmids = {doc.doc_id for doc in retrieved_documents}

    unverified = [pmid for pmid in cited_pmids if pmid not in retrieved_pmids]
    return unverified


def build_generation_prompt(
    query: str,
    retrieved_documents: list[Document],
    stance_outputs: list[Any],
    answerability_prob: float,
) -> str:
    """Build a prompt for the LLM that includes context and instructions for citation."""
    context_parts = []
    for index, doc in enumerate(retrieved_documents):
        stance = stance_outputs[index] if index < len(stance_outputs) else {}
        stance_text = (
            ""
            if not stance
            else (
                "\nStance probabilities (support, contradict, neutral): "
                f"{stance['supports']:.3f}, {stance['contradicts']:.3f}, "
                f"{stance['neutral']:.3f}"
            )
        )
        context_parts.append(
            f"[PMID:{doc.doc_id}] Title: {doc.title}\nAbstract: {doc.text}"
            f"{stance_text}\n"
        )
    context = "\n---\n".join(context_parts)

    prompt = (
        "Answer the question using only the supplied evidence. If evidence is insufficient, "
        "say so. Cite every factual claim with a source in the form [PMID:123], using only "
        "PMIDs present in the context.\n\n"
        f"Question: {query}\n\nContext:\n{context}\nAnswer:"
    )
    return prompt


def generate_answer(
    query: str,
    retrieved_documents: list[Document],
    stance_outputs: list[Any],
    answerability_prob: float,
    settings: Settings,
    abstention_threshold: float = 0.5,
) -> str:
    """Generate an answer using OpenRouter and verify citations.
    If answerability_prob is below a threshold, we abstain.
    """
    if answerability_prob < abstention_threshold:
        return "I do not have enough evidence to answer this question."

    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to generate an answer")

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
            answer += (
                "\n\nWarning: Unverified PMIDs: "
                f"{', '.join(unverified)}"
            )
        return answer
    finally:
        client.close()


__all__ = [
    "OpenRouterClient",
    "verify_citations",
    "build_generation_prompt",
    "generate_answer",
]
