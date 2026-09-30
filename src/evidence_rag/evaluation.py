import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow

from evidence_rag.config import Settings
from evidence_rag.datasets import load_beir
from evidence_rag.fusion import reciprocal_rank_fusion
from evidence_rag.metrics import aggregate, per_query_metrics
from evidence_rag.retrieval import BM25Index, DenseIndex, MedCPTQueryEncoder, SearchResult
from evidence_rag.statistics import bootstrap_confidence_intervals, paired_permutation_test

_METHODS = ("bm25", "dense", "hybrid")


def _ids(results: Sequence[SearchResult]) -> list[str]:
    return [result.doc_id for result in results]


def evaluate_dataset(name: str, settings: Settings) -> Path:
    settings.ensure_directories()
    benchmark = load_beir(name)
    dataset_index = settings.index_dir / name
    bm25 = BM25Index.load(dataset_index / "bm25")
    dense = DenseIndex(dataset_index / "dense")
    rankings: dict[str, dict[str, list[str]]] = {method: {} for method in _METHODS}

    for query_id, query in benchmark.queries.items():
        rankings["bm25"][query_id] = _ids(bm25.search(query, settings.retrieval_k))

    encoder = MedCPTQueryEncoder(settings.medcpt_query_model)
    query_ids = list(benchmark.queries)
    query_texts = [benchmark.queries[query_id] for query_id in query_ids]
    try:
        for query_id, vector in zip(
            query_ids,
            encoder.encode(query_texts, settings.embedding_batch_size),
            strict=True,
        ):
            rankings["dense"][query_id] = _ids(dense.search_vector(vector, settings.retrieval_k))
    finally:
        encoder.close()

    for query_id in query_ids:
        bm25_results = [SearchResult(doc_id, 0.0) for doc_id in rankings["bm25"][query_id]]
        dense_results = [SearchResult(doc_id, 0.0) for doc_id in rankings["dense"][query_id]]
        rankings["hybrid"][query_id] = _ids(
            reciprocal_rank_fusion(
                [bm25_results, dense_results], k=settings.rrf_k, limit=settings.retrieval_k
            )
        )

    methods: dict[str, Any] = {}
    per_method: dict[str, dict[str, dict[str, float]]] = {}
    for method in _METHODS:
        query_metrics = per_query_metrics(rankings[method], benchmark.qrels)
        per_method[method] = query_metrics
        methods[method] = {
            "metrics": aggregate(query_metrics),
            "ci95": {
                metric: list(bounds)
                for metric, bounds in bootstrap_confidence_intervals(
                    query_metrics,
                    samples=settings.bootstrap_samples,
                    seed=settings.random_seed,
                ).items()
            },
            "per_query": query_metrics,
        }

    comparisons: dict[str, dict[str, float]] = {}
    for first, second in (("hybrid", "bm25"), ("hybrid", "dense"), ("dense", "bm25")):
        key = f"{first}_vs_{second}"
        comparisons[key] = {}
        for metric in methods[first]["metrics"]:
            comparisons[key][metric] = paired_permutation_test(
                {query_id: values[metric] for query_id, values in per_method[first].items()},
                {query_id: values[metric] for query_id, values in per_method[second].items()},
                samples=settings.permutation_samples,
                seed=settings.random_seed,
            )

    result = {
        "dataset": name,
        "documents": len(benchmark.documents),
        "queries": len(benchmark.queries),
        "retrieval_k": settings.retrieval_k,
        "methods": methods,
        "paired_permutation_p_values": comparisons,
    }
    output = settings.results_dir / f"{name}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    _log_mlflow(result, output, settings)
    return output


def _log_mlflow(result: Mapping[str, Any], artifact: Path, settings: Settings) -> None:
    tracking_uri = settings.mlflow_tracking_uri
    if "://" not in tracking_uri:
        tracking_uri = Path(tracking_uri).resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    for method, values in result["methods"].items():
        with mlflow.start_run(run_name=f"{result['dataset']}-{method}"):
            mlflow.log_params(
                {
                    "dataset": result["dataset"],
                    "method": method,
                    "documents": result["documents"],
                    "queries": result["queries"],
                    "retrieval_k": result["retrieval_k"],
                }
            )
            mlflow.log_metrics(values["metrics"])
            mlflow.log_artifact(artifact)
