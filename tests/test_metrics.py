import pytest

from evidence_rag.metrics import ndcg_at_k, per_query_metrics, recall_at_k, reciprocal_rank_at_k
from evidence_rag.statistics import bootstrap_confidence_intervals, paired_permutation_test


def test_metrics_with_graded_relevance() -> None:
    ranking = ["d2", "d1", "x"]
    qrels = {"d1": 2, "d2": 1, "d3": 1}
    assert recall_at_k(ranking, qrels, 2) == pytest.approx(2 / 3)
    assert reciprocal_rank_at_k(ranking, qrels) == 1.0
    assert ndcg_at_k(["d1", "d2"], qrels, 10) > ndcg_at_k(["d2", "d1"], qrels, 10)


def test_statistics_are_deterministic_and_paired() -> None:
    per_query = {
        "q1": {"mrr@10": 1.0},
        "q2": {"mrr@10": 0.0},
        "q3": {"mrr@10": 0.5},
    }
    first = bootstrap_confidence_intervals(per_query, samples=200, seed=7)
    second = bootstrap_confidence_intervals(per_query, samples=200, seed=7)
    assert first == second
    assert paired_permutation_test(
        {"q1": 1.0, "q2": 1.0}, {"q1": 0.0, "q2": 0.0}, samples=200, seed=7
    ) < 0.6
    assert paired_permutation_test(
        {"q1": 1.0}, {"q1": 1.0}, samples=200, seed=7
    ) == 1.0


def test_per_query_ignores_queries_without_qrels() -> None:
    output = per_query_metrics({"q1": ["d1"], "q2": ["d2"]}, {"q1": {"d1": 1}, "q2": {}})
    assert set(output) == {"q1"}
