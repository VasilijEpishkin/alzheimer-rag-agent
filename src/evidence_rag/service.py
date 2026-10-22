"""FastAPI service for the evidence-aware biomedical RAG."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evidence_rag.answerability import aggregate_answerability_features, predict_answerability
from evidence_rag.claim_evidence_ledger import record_answer
from evidence_rag.config import Settings
from evidence_rag.datasets import load_beir
from evidence_rag.evidence_layer import answerability_features
from evidence_rag.fusion import reciprocal_rank_fusion
from evidence_rag.generation import generate_answer, verify_citations
from evidence_rag.models import Benchmark, Document
from evidence_rag.pubmed import enrich_benchmark_with_pubmed, load_pubmed_benchmark
from evidence_rag.reranking import Candidate, MedCPTCrossEncoder
from evidence_rag.retrieval import BM25Index, DenseIndex, MedCPTQueryEncoder

app = FastAPI(title="Evidence-aware Biomedical RAG", version="0.1.0")


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        json_schema_extra={
            "example": "What are the genetic risk factors for Alzheimer's disease?"
        },
    )
    dataset: Literal["scifact", "nfcorpus", "pubmed"] = "pubmed"
    top_k: int = Field(10, ge=1, le=100, description="Number of retrieved documents to consider")
    abstention_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Probability threshold for abstention"
    )


class DocumentResponse(BaseModel):
    doc_id: str
    title: str
    year: int | None = None
    is_retracted: bool
    publication_types: list[str]
    mesh_headings: list[str]


class QueryResponse(BaseModel):
    answer_id: str
    answer: str
    retrieved_documents: list[DocumentResponse]
    answerability_probability: float
    abstained: bool
    verification_unverified_pmids: list[str] = Field(default_factory=list)


def _load_indexes(dataset: str, settings: Settings):
    """Load BM25 and dense indexes for a dataset."""
    dataset_index = settings.index_dir / dataset
    bm25 = BM25Index.load(dataset_index / "bm25")
    dense = DenseIndex(dataset_index / "dense")
    return bm25, dense


def _retrieve_and_rerank(
    query: str,
    benchmark: Benchmark,
    bm25: BM25Index,
    dense: DenseIndex,
    settings: Settings,
    top_k: int,
) -> tuple[list[Document], list[Candidate], list[float], list[tuple[Document, Candidate]]]:
    """Retrieve candidates, rerank with cross-encoder, and return top documents and candidates."""
    # Retrieve candidates using BM25 and dense, then fuse.
    bm25_results = bm25.search(query, settings.ltr_candidate_k)
    encoder = MedCPTQueryEncoder(settings.medcpt_query_model)
    try:
        query_vector = next(encoder.encode([query], settings.embedding_batch_size))
        dense_results = dense.search_vector(query_vector, settings.ltr_candidate_k)
    finally:
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
    candidates: list[Candidate] = []
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
    try:
        cross_scores = scorer.score(
            query,
            [documents_by_id[c.doc_id] for c in candidates],
            settings.cross_encoder_batch_size,
        )
    finally:
        scorer.close()
    candidates = [
        replace(candidate, cross_score=score)
        for candidate, score in zip(candidates, cross_scores, strict=True)
    ]

    # Sort by cross-score descending and take top_k
    candidates.sort(key=lambda c: c.cross_score, reverse=True)
    excluded_retracted = [
        (documents_by_id[candidate.doc_id], candidate)
        for candidate in candidates
        if documents_by_id[candidate.doc_id].is_retracted
    ]
    candidates = [
        candidate
        for candidate in candidates
        if not documents_by_id[candidate.doc_id].is_retracted
    ]
    top_candidates = candidates[:top_k]
    top_documents = [documents_by_id[c.doc_id] for c in top_candidates]

    return (
        top_documents,
        top_candidates,
        [c.cross_score for c in top_candidates],
        excluded_retracted,
    )


def _compute_answerability(
    query: str,
    documents: list[Document],
    candidates: list[Candidate],
    settings: Settings,
) -> tuple[list[float], dict[str, dict[str, object]]]:
    """Compute answerability scores with the trained model artifact."""
    if not documents:
        return [], {}
    document_features = np.stack(
        [
            answerability_features(query, document, candidate)
            for document, candidate in zip(documents, candidates, strict=True)
        ]
    )
    model_path = settings.phase2_dir / "answerability.cbm"
    features = aggregate_answerability_features(document_features)[None, :]
    evidence_by_doc = {
        document.doc_id: {
            "stance": {
                "supports": float(values[-4]),
                "contradicts": float(values[-3]),
                "neutral": float(values[-2]),
            },
            "study_design_weight": float(values[-1]),
        }
        for document, values in zip(documents, document_features, strict=True)
    }
    return predict_answerability(features, model_path).tolist(), evidence_by_doc


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Retrieve, assess answerability, and generate a citation-checked answer."""
    settings = Settings()
    settings.ensure_directories()

    index_dir = settings.index_dir
    dataset_index = index_dir / request.dataset
    if not dataset_index.is_dir():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No indexes found for {request.dataset}. "
                f"Run `evidence-rag index {request.dataset}` first."
            ),
        )

    try:
        if request.dataset == "pubmed":
            benchmark = load_pubmed_benchmark(settings)
            if not benchmark.documents:
                raise HTTPException(
                    status_code=503,
                    detail="Local PubMed corpus is empty. Run `evidence-rag ingest` first.",
                )
        else:
            benchmark = load_beir(request.dataset)
            enrich_benchmark_with_pubmed(benchmark, settings, fetch_missing=False)
        bm25, dense = _load_indexes(request.dataset, settings)
        documents, candidates, _, excluded_retracted = _retrieve_and_rerank(
            request.query, benchmark, bm25, dense, settings, request.top_k
        )

        answerability_probs, evidence_by_doc = _compute_answerability(
            request.query, documents, candidates, settings
        )
        max_answerability = max(answerability_probs) if answerability_probs else 0.0
        abstained = max_answerability < request.abstention_threshold

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
        if abstained:
            answer = "I do not have enough evidence to answer this question."
            answer_id = record_answer(
                settings.data_dir / "claim_evidence.sqlite3",
                query=request.query,
                dataset=request.dataset,
                answer=answer,
                answerability_probability=max_answerability,
                abstention_threshold=request.abstention_threshold,
                documents=documents,
                candidates=candidates,
                evidence_by_doc=evidence_by_doc,
                excluded_retracted=excluded_retracted,
            )
            return QueryResponse(
                answer_id=answer_id,
                answer=answer,
                retrieved_documents=doc_responses,
                answerability_probability=max_answerability,
                abstained=True,
                verification_unverified_pmids=[],
            )

        answer = generate_answer(
            request.query,
            documents,
            [evidence_by_doc.get(doc.doc_id, {}).get("stance", {}) for doc in documents],
            max_answerability,
            settings,
            request.abstention_threshold,
        )
        unverified_pmids = verify_citations(answer, documents)
        answer_id = record_answer(
            settings.data_dir / "claim_evidence.sqlite3",
            query=request.query,
            dataset=request.dataset,
            answer=answer,
            answerability_probability=max_answerability,
            abstention_threshold=request.abstention_threshold,
            documents=documents,
            candidates=candidates,
            evidence_by_doc=evidence_by_doc,
            excluded_retracted=excluded_retracted,
            unverified_pmids=unverified_pmids,
        )
        return QueryResponse(
            answer_id=answer_id,
            answer=answer,
            retrieved_documents=doc_responses,
            answerability_probability=max_answerability,
            abstained=False,
            verification_unverified_pmids=unverified_pmids,
        )
    except HTTPException:
        raise
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail="External generation service failed") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Internal query error") from error


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
