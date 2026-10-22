from evidence_rag.config import Settings
from evidence_rag.models import Benchmark, Document
from evidence_rag.phase2 import _cross_score_candidates, _load_candidates, _save_candidates
from evidence_rag.reranking import Candidate


def test_cross_scores_resume_and_invalidate_on_query_change(tmp_path, monkeypatch) -> None:
    calls = []
    initializations = []

    class Scorer:
        def __init__(self, _model):
            initializations.append(_model)

        def score(self, query, documents, _batch_size):
            calls.append(query)
            return [float(len(calls))] * len(documents)

        def close(self):
            pass

    monkeypatch.setattr("evidence_rag.phase2.MedCPTCrossEncoder", Scorer)
    settings = Settings(data_dir=tmp_path)
    rows = [Candidate("q1", "d1", 1.0, 0.5, 1, 1)]
    benchmark = Benchmark("SciFact", [Document("d1", "title", "text")], {"q1": "query"}, {})

    assert _cross_score_candidates(benchmark, rows, settings)[0].cross_score == 1.0
    assert _cross_score_candidates(benchmark, rows, settings)[0].cross_score == 1.0
    assert calls == ["query"]
    assert len(initializations) == 1

    benchmark.queries["q1"] = "changed query"
    assert _cross_score_candidates(benchmark, rows, settings)[0].cross_score == 2.0
    assert calls == ["query", "changed query"]
    assert len(initializations) == 2

    checkpoint = next((tmp_path / "phase2" / "cross-scores" / "SciFact" / "test").glob("*.npz"))
    checkpoint.write_bytes(b"corrupt checkpoint")
    assert _cross_score_candidates(benchmark, rows, settings)[0].cross_score == 3.0
    assert calls == ["query", "changed query", "changed query"]


def test_candidate_cache_is_atomic_compatible_and_recovers_from_corruption(tmp_path) -> None:
    path = tmp_path / "candidates.npz"
    rows = [Candidate("q1", "d1", 1.0, 0.5, 1, 1, 0.25)]

    _save_candidates(path, rows, "fingerprint")
    assert _load_candidates(path, "fingerprint") == rows
    assert _load_candidates(path, "stale fingerprint") is None

    path.write_bytes(b"interrupted cache write")
    assert _load_candidates(path, "fingerprint") is None
