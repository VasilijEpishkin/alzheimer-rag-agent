from pathlib import Path

from evidence_rag.fusion import reciprocal_rank_fusion
from evidence_rag.models import Document
from evidence_rag.retrieval import BM25Index, SearchResult


def test_bm25_round_trip(tmp_path: Path) -> None:
    documents = [
        Document("a", "Alzheimer trial", "amyloid outcome"),
        Document("b", "Parkinson cohort", "motor outcome"),
    ]
    index = BM25Index.build(documents)
    index.save(tmp_path)
    results = BM25Index.load(tmp_path).search("amyloid Alzheimer", 2)
    assert results[0].doc_id == "a"


def test_rrf_rewards_consensus() -> None:
    first = [SearchResult("a", 10), SearchResult("b", 9)]
    second = [SearchResult("b", 1), SearchResult("c", 0)]
    assert reciprocal_rank_fusion([first, second], k=60)[0].doc_id == "b"
