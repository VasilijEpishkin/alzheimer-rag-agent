import math
from collections.abc import Mapping, Sequence

import numpy as np


def recall_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, relevance in qrels.items() if relevance > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant)


def reciprocal_rank_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    gains = [qrels.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    dcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sorted((relevance for relevance in qrels.values() if relevance > 0), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def per_query_metrics(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    recall_ks: Sequence[int] = (10, 100),
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for query_id, relevance in qrels.items():
        if not relevance or query_id not in rankings:
            continue
        ranked_ids = rankings[query_id]
        values = {f"recall@{k}": recall_at_k(ranked_ids, relevance, k) for k in recall_ks}
        values["mrr@10"] = reciprocal_rank_at_k(ranked_ids, relevance)
        values["ndcg@10"] = ndcg_at_k(ranked_ids, relevance)
        output[query_id] = values
    return output


def aggregate(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    metrics = next(iter(per_query.values())).keys()
    return {
        metric: float(np.mean([query[metric] for query in per_query.values()]))
        for metric in metrics
    }
