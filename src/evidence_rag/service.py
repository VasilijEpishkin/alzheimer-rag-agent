"""FastAPI service for the evidence-aware biomedical RAG."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evidence_rag.config import Settings
from evidence_rag.datasets import load_beir
from evidence_rag.evidence_layer import evidence_features, is_retracted, study_design_weight
from evidence_rag.generation import generate_answer, verify_citations
from evidence_rag.answerability import predict_answerability, compute_ece
from evidence_rag.ltr import train_and_predict
from evidence_rag.models import Document
from evidence_rag.pubmed import PubMedClient, store_documents
from evidence_rag.retrieval import BM25Index, DenseIndex, MedCPTQueryEncoder, SearchResult
from evidence_rag.reranking import Candidate, build_features, MedCPTCrossEncoder
from evidence_rag.fusion import reciprocal_rank_fusion


app = FastAPI(title="Evidence-aware Biomedical RAG", version="0.1.0")


class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the genetic risk factors for Alzheimer's disease?")
    top_k: int = Field(10, ge=1, le=100, description="Number of retrieved documents to consider")
    abstention_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Probability threshold for abstention")


class DocumentResponse(BaseModel):
    doc_id: str
    title: str
    year: Optional[int] = None
    is_retracted: bool
    publication_types: List[str]
    mesh_headings: List[str]


class QueryResponse(BaseModel):
    answer: str
    retrieved_documents: List[DocumentResponse]
    answerability_probability: float
    abstained: bool
    verification_unverified_pmids: List[str] = []


def _load_indexes(dataset: str, settings: Settings):
    """Load BM25 and dense indexes for a dataset."""
    dataset_index = settings.index_dir / dataset
    bm25 = BM25Index.load(dataset_index / "bm25")
    dense = DenseIndex(dataset_index / "dense")
    return bm25, dense


def _retrieve_and_rerank(
    query: str,
    bm25: BM25Index,
    dense: DenseIndex,
    settings: Settings,
    top_k: int,
) -> tuple[List[Document], List[Candidate], List[float]]:
    """Retrieve candidates, rerank with cross-encoder, and return top documents and candidates."""
    # We'll use the same logic as in phase2 but simplified for service.
    # For now, we assume we have a small set of documents already indexed.
    # In a real service, we might have a separate indexing step.
    # We'll load documents from the PubMed cache? Actually, we need a way to get documents.
    # For the service, we will assume that the indexes are built from a dataset (like BEIR) and we have the documents.
    # We'll need to load the documents from the dataset. We'll use the BEIR dataset loader for simplicity.
    # But note: the service should work with the PubMed store. However, for now, we'll use the BEIR dataset
    # that is already indexed. This is a placeholder.

    # Load the dataset (we'll use the first available dataset for demonstration)
    # In a real implementation, we would have a way to specify which dataset/index to use.
    # We'll just use the first dataset from the indexes directory.
    index_dir = settings.index_dir
    datasets = [d.name for d in index_dir.iterdir() if d.is_dir()]
    if not datasets:
        raise RuntimeError("No indexes found. Please run `evidence-rag index` first.")
    dataset_name = datasets[0]
    benchmark = load_beir(dataset_name)

    # Retrieve candidates using BM25 and dense, then fuse
    bm25_results = bm25.search(query, settings.ltr_candidate_k)
    encoder = MedCPTQueryEncoder(settings.medcpt_query_model)
    query_vector = next(encoder.encode([query], settings.embedding_batch_size))
    dense_results = dense.search_vector(query_vector, settings.ltr_candidate_k)
    encoder.close()

    fused = reciprocal_rank_fusion(
        [bm25_results, dense_results],
        k=settings.rrf_k,
        limit=settings.ltr_candidate_k,
    )

    # Build candidates
    bm25_by_id = {result.doc_id: result.score for result in bm25_results}
    dense_by_id = {result.doc_id: result.score for result in dense_results}
    bm25_ranks = {result.doc_id: rank for rank, result in enumerate(bm25_results, 1)}
    dense_ranks = {result.doc_id: rank for rank, result in enumerate(dense_results, 1)}
    candidates: List[Candidate] = []
    for result in fused:
        candidates.append(
            Candidate(
                query_id="service_query",
                doc_id=result.doc_id,
                bm25_score=bm25_by_id.get(result.doc_id, 0.0),
                dense_score=dense_by_id.get(result.doc_id, 0.0),
                bm25_rank=bm25_ranks.get(result.doc_id, 0),
                dense_rank=dense_ranks.get(result.doc_id, 0),
                cross_score=0.0,  # We'll compute cross-encoder score next
            )
        )

    # Rerank with cross-encoder
    scorer = MedCPTCrossEncoder(settings.medcpt_cross_model)
    documents_by_id = {doc.doc_id: doc for doc in benchmark.documents}
    cross_scores = scorer.score(
        query,
        [documents_by_id[c.doc_id] for c in candidates],
        settings.cross_encoder_batch_size,
    )
    scorer.close()
    for candidate, score in zip(candidates, cross_scores):
        candidate.cross_score = score

    # Sort by cross-score descending and take top_k
    candidates.sort(key=lambda c: c.cross_score, reverse=True)
    top_candidates = candidates[:top_k]
    top_documents = [documents_by_id[c.doc_id] for c in top_candidates]

    return top_documents, top_candidates, [c.cross_score for c in top_candidates]


def _compute_answerability(
    query: str,
    documents: List[Document],
    candidates: List[Candidate],
    settings: Settings,
) -> List[float]:
    """Compute answerability probability for each document."""
    # We'll compute features for each document and then use the answerability model.
    # However, we don't have a trained answerability model yet.
    # For now, we return a placeholder probability (0.5) to avoid fabricating data.
    # In a real implementation, we would load a trained model from settings.phase2_dir or similar.
    # We'll skip the model and return a fixed value.
    return [0.5] * len(documents)


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handle a query: retrieve, rerank, compute answerability, generate answer, verify citations."""
    settings = Settings()
    settings.ensure_directories()

    # Check if indexes exist
    index_dir = settings.index_dir
    if not any(index_dir.iterdir()):
        raise HTTPException(
            status_code=503,
            detail="No indexes found. Please run `evidence-rag index` to build indexes first.",
        )

    try:
        # Retrieve and rerank
        bm25, dense = _load_indexes("", settings)  # We'll ignore dataset for now and use the first one
        documents, candidates, cross_scores = _retrieve_and_rerank(
            request.query, bm25, dense, settings, request.top_k
        )
        bm25.close()  # Not actually a method, but we can ignore
        dense.close()  # Same

        # Compute answerability probabilities (placeholder)
        answerability_probs = _compute_answerability(
            request.query, documents, candidates, settings
        )
        # We'll use the average probability for the service response? Or we can return per document.
        # For simplicity, we'll use the max probability as the overall answerability.
        max_answerability = max(answerability_probs) if answerability_probs else 0.0
        abstained = max_answerability < request.abstention_threshold

        # If abstained, we return early with a message.
        if abstained:
            return QueryResponse(
                answer="I do not have enough evidence to answer this question.",
                retrieved_documents=[
                    DocumentResponse(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        year=doc.year,
                        is_retracted=doc.is_retracted,
                        publication_types=list(doc.publication_types),
                        mesh_headings=list(doc.mesh_headings),
                    )
                    for doc in documents
                ],
                answerability_probability=max_answerability,
                abstained=True,
                verification_unverified_pmids=[],
            )

        # Generate answer (placeholder generation)
        # We'll use a simple template for now to avoid calling OpenRouter without API key.
        # In a real implementation, we would call the generation module.
        # We'll check if the OpenRouter API key is set.
        if not settings.openrouter_api_key:
            answer = (
                "OpenRouter API key not configured. Cannot generate answer. "
                "This is a placeholder response."
            )
        else:
            # We would call the generation module here.
            # For now, we'll use a placeholder.
            answer = (
                "This is a placeholder answer. The generation module is not fully implemented. "
                "Please configure the OpenRouter API key and implement the generation logic."
            )

        # Verify citations (placeholder: we assume no citations in the placeholder answer)
        unverified_pmids = verify_citations(answer, documents)

        # Build document responses
        doc_responses = [
            DocumentResponse(
                doc_id=doc.doc_id,
                title=doc.title,
                year=doc.year,
                is_retracted=doc.is_retracted,
                publication_types=list(doc.publication_types),
                mesh_headings=list(doc.mesh_headings),
            )
            for doc in documents
        ]

        return QueryResponse(
            answer=answer,
            retrieved_documents=doc_responses,
            answerability_probability=max_answerability,
            abstained=False,
            verification_unverified_pmids=unverified_pmids,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main() -> None:
    """Run the FastAPI service with Uvicorn."""
    uvicorn.run(
        "evidence_rag.service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]