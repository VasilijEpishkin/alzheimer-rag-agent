from pathlib import Path

import mlflow
import numpy as np

from evidence_rag import evaluation
from evidence_rag.config import Settings
from evidence_rag.models import Benchmark, Document
from evidence_rag.retrieval import SearchResult


class _FakeBM25:
    def search(self, query: str, k: int) -> list[SearchResult]:
        rankings = {
            "query one": [SearchResult("d1", 2.0), SearchResult("d2", 1.0)],
            "query two": [SearchResult("d2", 2.0), SearchResult("d1", 1.0)],
        }
        return rankings[query][:k]


class _FakeDense:
    def __init__(self, path: Path) -> None:
        self.path = path

    def search_vector(self, vector: np.ndarray, k: int) -> list[SearchResult]:
        ranking = ["d1", "d2"] if vector[0] == 1 else ["d2", "d1"]
        return [SearchResult(doc_id, 1.0 / rank) for rank, doc_id in enumerate(ranking, 1)][:k]


class _FakeEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 16):
        yield from (np.array([index], dtype=np.float32) for index in range(1, len(texts) + 1))

    def close(self) -> None:
        pass


def test_evaluate_dataset_writes_statistics_and_mlflow_runs(tmp_path: Path, monkeypatch) -> None:
    benchmark = Benchmark(
        name="fixture",
        documents=[Document("d1", "one", "text"), Document("d2", "two", "text")],
        queries={"q1": "query one", "q2": "query two"},
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    )
    monkeypatch.setattr(evaluation, "load_beir", lambda name: benchmark)
    monkeypatch.setattr(evaluation.BM25Index, "load", lambda path: _FakeBM25())
    monkeypatch.setattr(evaluation, "DenseIndex", _FakeDense)
    monkeypatch.setattr(evaluation, "MedCPTQueryEncoder", _FakeEncoder)
    settings = Settings(
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        index_dir=tmp_path / "indexes",
        results_dir=tmp_path / "results",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        mlflow_experiment="test-phase-1",
        bootstrap_samples=100,
        permutation_samples=100,
    )

    output = evaluation.evaluate_dataset("fixture", settings)

    assert output.is_file()
    result = evaluation.json.loads(output.read_text())
    assert set(result["methods"]) == {"bm25", "dense", "hybrid"}
    assert all(
        {"metrics", "ci95", "per_query"} <= values.keys()
        for values in result["methods"].values()
    )
    assert result["paired_permutation_p_values"]
    experiment = mlflow.get_experiment_by_name("test-phase-1")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 3
    assert "metrics.ndcg_at_10" in runs.columns