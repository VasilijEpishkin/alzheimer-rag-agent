import json
from pathlib import Path

from evidence_rag import phase2
from evidence_rag.config import Settings
from evidence_rag.models import Benchmark, Document
from evidence_rag.reranking import Candidate


def test_phase2_uses_train_labels_and_test_qrels_only_for_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    documents = [
        Document("d1", "alpha", "text"),
        Document("d2", "beta", "text"),
        Document("d3", "gamma", "text"),
        Document("d4", "delta", "text"),
    ]
    train = Benchmark(
        "fixture",
        documents,
        {"train-1": "alpha", "train-2": "gamma"},
        {"train-1": {"d1": 1}, "train-2": {"d3": 1}},
        split="train",
    )
    test = Benchmark(
        "fixture",
        documents,
        {"test-1": "alpha", "test-2": "gamma"},
        {"test-1": {"d2": 1}, "test-2": {"d4": 1}},
        split="test",
    )
    candidates = {
        "train": [
            Candidate("train-1", "d1", 2.0, 0.8, 1, 1, 2.0),
            Candidate("train-1", "d2", 1.0, 0.2, 2, 2, 1.0),
            Candidate("train-2", "d3", 2.0, 0.8, 1, 1, 2.0),
            Candidate("train-2", "d4", 1.0, 0.2, 2, 2, 1.0),
        ],
        "test": [
            Candidate("test-1", "d1", 2.0, 0.8, 1, 1, 2.0),
            Candidate("test-1", "d2", 1.0, 0.2, 2, 2, 1.0),
            Candidate("test-2", "d3", 2.0, 0.8, 1, 1, 2.0),
            Candidate("test-2", "d4", 1.0, 0.2, 2, 2, 1.0),
        ],
    }
    observed_labels: list[list[float]] = []

    def load_fixture(name, split):
        if split == "dev":
            raise ValueError("Unknown 'fixture' split 'dev'; choose from train, test")
        return train if split == "train" else test

    monkeypatch.setattr(phase2, "load_beir", load_fixture)
    monkeypatch.setattr(
        phase2,
        "_get_candidates",
        lambda benchmark, settings: candidates[benchmark.split],
    )
    monkeypatch.setattr(phase2, "_log_runs", lambda *args: None)

    def fake_train(
        kind,
        train_features,
        train_labels,
        train_group_sizes,
        validation_features,
        validation_labels,
        validation_group_sizes,
        test_features,
        feature_names,
        model_path,
        **kwargs,
    ):
        observed_labels.append(train_labels.tolist())
        assert set(validation_labels.tolist()) == {0.0, 1.0}
        assert len(validation_group_sizes) == 1
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("fixture")
        return test_features[:, 0], 10

    monkeypatch.setattr(phase2, "train_and_predict", fake_train)
    settings = Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        index_dir=tmp_path / "indexes",
        results_dir=tmp_path / "results",
        ltr_candidate_k=10,
        bootstrap_samples=100,
        permutation_samples=100,
    )

    output = phase2.evaluate_phase2("fixture", settings)

    result = json.loads(output.read_text())
    assert observed_labels == [[1.0, 0.0]] * 4
    assert result["train_split"] == "train"
    assert result["test_split"] == "test"
    assert result["training_queries"] == 1
    assert result["validation_queries"] == 1
    assert result["validation_split"] == "stratified train query holdout"
    assert set(result["methods"]) == {
        "cross_encoder",
        "catboost_relevance",
        "catboost_evidence",
        "xgboost_relevance",
        "xgboost_evidence",
    }
    assert result["methods"]["cross_encoder"]["metrics"]["mrr@10"] == 0.5
