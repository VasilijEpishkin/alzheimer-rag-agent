import sqlite3

from evidence_rag.claim_evidence_ledger import record_answer
from evidence_rag.models import Document
from evidence_rag.reranking import Candidate


def test_record_answer_persists_claim_evidence_and_retracted_exclusions(tmp_path):
    document = Document(
        "123",
        "A trial",
        "The drug reduced amyloid burden in a randomized trial.",
        2024,
        ("Alzheimer Disease",),
        ("Randomized Controlled Trial",),
        False,
        "123",
        "2026-10-22T10:00:00+00:00",
    )
    retracted = Document(
        "456",
        "A retracted trial",
        "Earlier result.",
        is_retracted=True,
        pubmed_id="456",
        metadata_fetched_at="2026-10-22T10:00:00+00:00",
    )
    candidate = Candidate("q", "123", 2.0, 0.8, 1, 2, 0.9)
    excluded_candidate = Candidate("q", "456", 1.0, 0.7, 2, 3, 0.8)

    answer_id = record_answer(
        tmp_path / "ledger.sqlite3",
        query="Does the drug reduce amyloid?",
        dataset="pubmed",
        answer="The drug reduced amyloid [PMID:123].",
        answerability_probability=0.9,
        abstention_threshold=0.5,
        documents=[document],
        candidates=[candidate],
        evidence_by_doc={
            "123": {
                "stance": {"supports": 0.8, "contradicts": 0.1, "neutral": 0.1},
                "study_design_weight": 4.0,
            }
        },
        excluded_retracted=[(retracted, excluded_candidate)],
    )

    with sqlite3.connect(tmp_path / "ledger.sqlite3") as connection:
        answer = connection.execute(
            "SELECT retraction_policy_version, final_decision FROM answers WHERE answer_id=?",
            (answer_id,),
        ).fetchone()
        entries = connection.execute(
            "SELECT cited_pmid, cited_passage, cross_score, evidence_policy_weight, "
            "metadata_fetched_at, excluded_reason FROM ledger_entries "
            "WHERE answer_id=? ORDER BY entry_id",
            (answer_id,),
        ).fetchall()

    assert answer == ("pubmed-retraction-v1", "answer")
    assert entries[0] == (
        "123",
        "The drug reduced amyloid burden in a randomized trial.",
        0.9,
        4.0,
        "2026-10-22T10:00:00+00:00",
        None,
    )
    assert entries[1][0] == "456"
    assert entries[1][-1] == "retracted"
