"""Persist auditable answer and claim-to-evidence records in SQLite."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidence_rag.models import Document
from evidence_rag.reranking import Candidate

_POLICY_VERSION = "pubmed-retraction-v1"
_CITATION = re.compile(r"\[(?:PMID:)?(\d+)\]")
_TOKEN = re.compile(r"[a-z0-9]{3,}")


def _best_passage(claim: str, document: Document) -> str | None:
    claim_terms = set(_TOKEN.findall(claim.casefold()))
    passages = re.split(r"(?<=[.!?])\s+", document.text.strip())
    if not claim_terms or not passages or not document.text.strip():
        return None
    passage = max(
        passages, key=lambda text: len(claim_terms & set(_TOKEN.findall(text.casefold())))
    )
    return passage if claim_terms & set(_TOKEN.findall(passage.casefold())) else None


def record_answer(
    path: Path,
    *,
    query: str,
    dataset: str,
    answer: str,
    answerability_probability: float,
    abstention_threshold: float,
    documents: Sequence[Document],
    candidates: Sequence[Candidate],
    evidence_by_doc: Mapping[str, Mapping[str, Any]],
    excluded_retracted: Sequence[tuple[Document, Candidate]],
    unverified_pmids: Sequence[str] = (),
) -> str:
    """Write the answer, cited passages, evidence context and retracted exclusions."""
    answer_id = str(uuid.uuid4())
    abstained = answerability_probability < abstention_threshold
    decision = "abstain" if abstained else "answer"
    documents_by_id = {document.doc_id: document for document in documents}
    candidates_by_id = {candidate.doc_id: candidate for candidate in candidates}
    entries: list[tuple[Any, ...]] = []

    if not abstained:
        for claim in filter(None, re.split(r"(?<=[.!?])\s+", answer.strip())):
            cited_pmids = _CITATION.findall(claim)
            for pmid in cited_pmids or [None]:
                document = documents_by_id.get(pmid or "")
                candidate = candidates_by_id.get(pmid or "")
                evidence = evidence_by_doc.get(pmid or "", {})
                stance = evidence.get("stance", {})
                final_decision = (
                    "uncited"
                    if pmid is None
                    else "unverified"
                    if document is None
                    else "qualified"
                    if stance.get("contradicts", 0.0) > stance.get("supports", 0.0)
                    else "answer"
                )
                entries.append(
                    (
                        answer_id,
                        claim,
                        pmid,
                        _best_passage(claim, document) if document else None,
                        candidate.bm25_score if candidate else None,
                        candidate.dense_score if candidate else None,
                        candidate.cross_score if candidate else None,
                        json.dumps(document.publication_types) if document else None,
                        int(document.is_retracted) if document else None,
                        document.metadata_fetched_at if document else None,
                        evidence.get("study_design_weight"),
                        stance.get("supports"),
                        stance.get("contradicts"),
                        stance.get("neutral"),
                        final_decision,
                        None,
                    )
                )

    for document, candidate in excluded_retracted:
        entries.append(
            (
                answer_id,
                None,
                document.pubmed_id or document.doc_id,
                None,
                candidate.bm25_score,
                candidate.dense_score,
                candidate.cross_score,
                json.dumps(document.publication_types),
                1,
                document.metadata_fetched_at,
                None,
                None,
                None,
                None,
                "excluded",
                "retracted",
            )
        )

    if not abstained and (
        unverified_pmids
        or any(entry[14] in {"qualified", "uncited", "unverified"} for entry in entries)
    ):
        decision = "qualified"

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS answers ("
            "answer_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, query TEXT NOT NULL, "
            "dataset TEXT NOT NULL, answer TEXT NOT NULL, answerability_probability REAL NOT NULL, "
            "abstention_threshold REAL NOT NULL, final_decision TEXT NOT NULL, "
            "retraction_policy_version TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS ledger_entries ("
            "entry_id INTEGER PRIMARY KEY, answer_id TEXT NOT NULL, claim_text TEXT, "
            "cited_pmid TEXT, cited_passage TEXT, bm25_score REAL, dense_score REAL, "
            "cross_score REAL, publication_types TEXT, is_retracted INTEGER, "
            "metadata_fetched_at TEXT, evidence_policy_weight REAL, stance_supports REAL, "
            "stance_contradicts REAL, stance_neutral REAL, final_decision TEXT NOT NULL, "
            "excluded_reason TEXT, FOREIGN KEY(answer_id) REFERENCES answers(answer_id))"
        )
        connection.execute(
            "INSERT INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                answer_id,
                datetime.now(UTC).isoformat(),
                query,
                dataset,
                answer,
                answerability_probability,
                abstention_threshold,
                decision,
                _POLICY_VERSION,
            ),
        )
        connection.executemany(
            "INSERT INTO ledger_entries (answer_id, claim_text, cited_pmid, cited_passage, "
            "bm25_score, dense_score, cross_score, publication_types, is_retracted, "
            "metadata_fetched_at, evidence_policy_weight, stance_supports, stance_contradicts, "
            "stance_neutral, final_decision, excluded_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            entries,
        )
    return answer_id
