import gc
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from evidence_rag.config import Settings
from evidence_rag.datasets import load_beir
from evidence_rag.fusion import reciprocal_rank_fusion
from evidence_rag.ltr import train_and_predict
from evidence_rag.metrics import aggregate, per_query_metrics
from evidence_rag.models import Benchmark
from evidence_rag.reranking import (
    FEATURE_NAMES,
    RELEVANCE_FEATURES,
    Candidate,
    MedCPTCrossEncoder,
    build_features,
)
from evidence_rag.retrieval import BM25Index, DenseIndex, MedCPTQueryEncoder
from evidence_rag.statistics import bootstrap_confidence_intervals, paired_permutation_test

_VARIANTS = {
    "relevance": RELEVANCE_FEATURES,
    "evidence": FEATURE_NAMES,
}
_LOGGER = logging.getLogger(__name__)


def _tracking_uri(uri: str) -> str:
    if "://" not in uri:
        uri = Path(uri).resolve().as_uri()
    if uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return uri


def _candidate_cache_path(settings: Settings, dataset: str, split: str) -> Path:
    return settings.phase2_dir / f"{dataset}-{split}-k{settings.ltr_candidate_k}.npz"


def _save_candidates(path: Path, rows: Sequence[Candidate], model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                query_ids=np.asarray([row.query_id for row in rows]),
                doc_ids=np.asarray([row.doc_id for row in rows]),
                bm25_scores=np.asarray([row.bm25_score for row in rows], dtype=np.float32),
                dense_scores=np.asarray([row.dense_score for row in rows], dtype=np.float32),
                bm25_ranks=np.asarray([row.bm25_rank for row in rows], dtype=np.int32),
                dense_ranks=np.asarray([row.dense_rank for row in rows], dtype=np.int32),
                cross_scores=np.asarray([row.cross_score for row in rows], dtype=np.float32),
                cross_model=np.asarray(model_name),
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_candidates(path: Path, model_name: str) -> list[Candidate] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["cross_model"]) != model_name:
                return None
            return [
                Candidate(
                    query_id=str(query_id),
                    doc_id=str(doc_id),
                    bm25_score=float(bm25_score),
                    dense_score=float(dense_score),
                    bm25_rank=int(bm25_rank),
                    dense_rank=int(dense_rank),
                    cross_score=float(cross_score),
                )
                for (
                    query_id,
                    doc_id,
                    bm25_score,
                    dense_score,
                    bm25_rank,
                    dense_rank,
                    cross_score,
                ) in zip(
                    data["query_ids"],
                    data["doc_ids"],
                    data["bm25_scores"],
                    data["dense_scores"],
                    data["bm25_ranks"],
                    data["dense_ranks"],
                    data["cross_scores"],
                    strict=True,
                )
            ]
    except (OSError, ValueError, KeyError):
        return None


def _retrieve_candidates(
    benchmark: Benchmark, settings: Settings, bm25: BM25Index, dense: DenseIndex
) -> list[Candidate]:
    rows: list[Candidate] = []
    encoder = MedCPTQueryEncoder(settings.medcpt_query_model)
    query_ids = list(benchmark.queries)
    query_texts = [benchmark.queries[query_id] for query_id in query_ids]
    try:
        vectors = encoder.encode(query_texts, settings.embedding_batch_size)
        for query_id, query, vector in zip(query_ids, query_texts, vectors, strict=True):
            bm25_results = bm25.search(query, settings.ltr_candidate_k)
            dense_results = dense.search_vector(vector, settings.ltr_candidate_k)
            fused = reciprocal_rank_fusion(
                [bm25_results, dense_results],
                k=settings.rrf_k,
                limit=settings.ltr_candidate_k,
            )
            bm25_by_id = {result.doc_id: result.score for result in bm25_results}
            dense_by_id = {result.doc_id: result.score for result in dense_results}
            bm25_ranks = {result.doc_id: rank for rank, result in enumerate(bm25_results, 1)}
            dense_ranks = {result.doc_id: rank for rank, result in enumerate(dense_results, 1)}
            rows.extend(
                Candidate(
                    query_id=query_id,
                    doc_id=result.doc_id,
                    bm25_score=bm25_by_id.get(result.doc_id, 0.0),
                    dense_score=dense_by_id.get(result.doc_id, 0.0),
                    bm25_rank=bm25_ranks.get(result.doc_id, 0),
                    dense_rank=dense_ranks.get(result.doc_id, 0),
                )
                for result in fused
            )
    finally:
        encoder.close()
    return rows


def _cross_score_candidates(
    benchmark: Benchmark, rows: Sequence[Candidate], settings: Settings
) -> list[Candidate]:
    scorer = None
    documents = benchmark.documents_by_id
    scored: list[Candidate] = []
    checkpoint_dir = settings.phase2_dir / "cross-scores" / benchmark.name / benchmark.split
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    groups: list[tuple[str, Sequence[Candidate]]] = []
    start = 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and rows[end].query_id == rows[start].query_id:
            end += 1
        groups.append((rows[start].query_id, rows[start:end]))
        start = end
    try:
        for index, (query_id, group) in enumerate(groups, 1):
            input_hash = hashlib.sha256(
                json.dumps(
                    [
                        settings.medcpt_cross_model,
                        benchmark.queries[query_id],
                        [
                            [row.doc_id, documents[row.doc_id].retrieval_text]
                            for row in group
                        ],
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            checkpoint = checkpoint_dir / f"{hashlib.sha256(query_id.encode()).hexdigest()}.npz"
            scores = None
            if checkpoint.exists():
                try:
                    with np.load(checkpoint, allow_pickle=False) as data:
                        if str(data["input_hash"]) == input_hash:
                            scores = data["scores"].tolist()
                except (OSError, ValueError, KeyError):
                    pass
                if scores is not None and len(scores) != len(group):
                    scores = None
            if scores is None:
                if scorer is None:
                    scorer = MedCPTCrossEncoder(settings.medcpt_cross_model)
                scores = scorer.score(
                    benchmark.queries[query_id],
                    [documents[row.doc_id] for row in group],
                    settings.cross_encoder_batch_size,
                )
                fd, temporary = tempfile.mkstemp(dir=checkpoint_dir, suffix=".npz")
                try:
                    with os.fdopen(fd, "wb") as stream:
                        np.savez_compressed(
                            stream, input_hash=np.asarray(input_hash), scores=scores
                        )
                    os.replace(temporary, checkpoint)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            scored.extend(
                replace(row, cross_score=score)
                for row, score in zip(group, scores, strict=True)
            )
            if index % 10 == 0 or index == len(groups):
                _LOGGER.info("Cross-encoder scored %d/%d queries", index, len(groups))
    finally:
        if scorer is not None:
            scorer.close()
    return scored


def _get_candidates(benchmark: Benchmark, settings: Settings) -> list[Candidate]:
    path = _candidate_cache_path(settings, benchmark.name, benchmark.split)
    cache_model = "|".join(
        (
            settings.medcpt_cross_model,
            settings.medcpt_query_model,
            str(settings.ltr_candidate_k),
            str(settings.rrf_k),
            benchmark.name,
            benchmark.split,
            hashlib.sha256(
                json.dumps(benchmark.queries, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            hashlib.sha256(
                json.dumps(
                    [
                        [document.doc_id, document.retrieval_text]
                        for document in benchmark.documents
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )
    )
    cached = _load_candidates(path, cache_model)
    if cached is not None:
        return cached
    dataset_index = settings.index_dir / benchmark.name
    rows = _retrieve_candidates(
        benchmark,
        settings,
        BM25Index.load(dataset_index / "bm25"),
        DenseIndex(dataset_index / "dense"),
    )
    rows = _cross_score_candidates(benchmark, rows, settings)
    _save_candidates(path, rows, cache_model)
    return rows


def _matrix(
    benchmark: Benchmark, rows: Sequence[Candidate]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    features = np.stack(
        [
            build_features(
                benchmark.queries[row.query_id], benchmark.documents_by_id[row.doc_id], row
            )
            for row in rows
        ]
    )
    labels = np.asarray(
        [benchmark.qrels.get(row.query_id, {}).get(row.doc_id, 0) for row in rows],
        dtype=np.float32,
    )
    return features, labels, [row.query_id for row in rows], [row.doc_id for row in rows]


def _training_rows(
    features: np.ndarray, labels: np.ndarray, query_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    selected: list[int] = []
    group_sizes: list[int] = []
    skipped = 0
    start = 0
    while start < len(query_ids):
        end = start + 1
        while end < len(query_ids) and query_ids[end] == query_ids[start]:
            end += 1
        group_labels = labels[start:end]
        if np.any(group_labels > 0) and np.any(group_labels == 0):
            selected.extend(range(start, end))
            group_sizes.append(end - start)
        else:
            skipped += 1
        start = end
    if not selected:
        raise ValueError("No training query has both a judged positive and a candidate negative")
    indices = np.asarray(selected, dtype=np.int64)
    return features[indices], labels[indices], group_sizes, skipped


def _rankings(
    query_ids: Sequence[str], doc_ids: Sequence[str], scores: Sequence[float]
) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[str, float]]] = {}
    for query_id, doc_id, score in zip(query_ids, doc_ids, scores, strict=True):
        grouped.setdefault(query_id, []).append((doc_id, float(score)))
    return {
        query_id: [doc_id for doc_id, _ in sorted(values, key=lambda item: (-item[1], item[0]))]
        for query_id, values in grouped.items()
    }


def _method_result(
    rankings: Mapping[str, Sequence[str]], benchmark: Benchmark, settings: Settings
) -> dict[str, Any]:
    query_metrics = per_query_metrics(rankings, benchmark.qrels)
    known = {
        document.doc_id
        for document in benchmark.documents
        if document.publication_types or document.year is not None or document.mesh_headings
    }
    top_documents = [doc_id for ranking in rankings.values() for doc_id in ranking[:10]]
    known_top = [doc_id for doc_id in top_documents if doc_id in known]
    retracted_known = sum(benchmark.documents_by_id[doc_id].is_retracted for doc_id in known_top)
    return {
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
        "safety": {
            "evidence_metadata_coverage_at_10": len(known_top) / len(top_documents)
            if top_documents
            else 0.0,
            "retracted_fraction_among_known_at_10": retracted_known / len(known_top)
            if known_top
            else None,
        },
    }


def _comparisons(
    methods: Mapping[str, Mapping[str, Any]], settings: Settings
) -> dict[str, dict[str, float]]:
    pairs = [(method, "cross_encoder") for method in methods if method != "cross_encoder"]
    pairs.extend(
        [
            ("catboost_evidence", "catboost_relevance"),
            ("xgboost_evidence", "xgboost_relevance"),
            ("catboost_evidence", "xgboost_evidence"),
        ]
    )
    output: dict[str, dict[str, float]] = {}
    for first, second in pairs:
        output[f"{first}_vs_{second}"] = {
            metric: paired_permutation_test(
                {
                    query_id: values[metric]
                    for query_id, values in methods[first]["per_query"].items()
                },
                {
                    query_id: values[metric]
                    for query_id, values in methods[second]["per_query"].items()
                },
                samples=settings.permutation_samples,
                seed=settings.random_seed,
            )
            for metric in methods[first]["metrics"]
        }
    return output


def _log_runs(
    result: Mapping[str, Any],
    artifact: Path,
    model_paths: Mapping[str, Path],
    settings: Settings,
) -> None:
    mlflow.set_tracking_uri(_tracking_uri(settings.mlflow_tracking_uri))
    mlflow.set_experiment(settings.phase2_mlflow_experiment)
    for method, values in result["methods"].items():
        with mlflow.start_run(run_name=f"{result['dataset']}-{method}"):
            mlflow.log_params(
                {
                    "dataset": result["dataset"],
                    "train_split": result["train_split"],
                    "test_split": "test",
                    "method": method,
                    "candidate_k": result["candidate_k"],
                    "cross_model": settings.medcpt_cross_model,
                    "iterations": settings.ltr_iterations if method in model_paths else 0,
                    "training_queries": result["training_queries"],
                    "skipped_training_queries": result["skipped_training_queries"],
                }
            )
            mlflow.log_metrics(
                {name.replace("@", "_at_"): value for name, value in values["metrics"].items()}
            )
            mlflow.log_metric(
                "evidence_metadata_coverage_at_10",
                values["safety"]["evidence_metadata_coverage_at_10"],
            )
            mlflow.log_artifact(str(artifact))
            if method in model_paths:
                mlflow.log_artifact(str(model_paths[method]), artifact_path="models")


def evaluate_phase2(name: str, settings: Settings) -> Path:
    settings.ensure_directories()
    train = load_beir(name, split="train")
    test = load_beir(name, split="test")
    train_rows = _get_candidates(train, settings)
    test_rows = _get_candidates(test, settings)
    train_features, train_labels, train_query_ids, _ = _matrix(train, train_rows)
    test_features, _, test_query_ids, test_doc_ids = _matrix(test, test_rows)
    train_features, train_labels, group_sizes, skipped = _training_rows(
        train_features, train_labels, train_query_ids
    )

    methods: dict[str, dict[str, Any]] = {}
    cross_rankings = _rankings(
        test_query_ids,
        test_doc_ids,
        [row.cross_score for row in test_rows],
    )
    methods["cross_encoder"] = _method_result(cross_rankings, test, settings)
    model_paths: dict[str, Path] = {}
    for kind in ("catboost", "xgboost"):
        for variant, feature_names in _VARIANTS.items():
            indices = [FEATURE_NAMES.index(feature) for feature in feature_names]
            method = f"{kind}_{variant}"
            suffix = "cbm" if kind == "catboost" else "json"
            model_path = settings.phase2_dir / "models" / f"{name}-{method}.{suffix}"
            predictions = train_and_predict(
                kind,  # type: ignore[arg-type]
                train_features[:, indices],
                train_labels,
                group_sizes,
                test_features[:, indices],
                feature_names,
                model_path,
                seed=settings.random_seed,
                iterations=settings.ltr_iterations,
            )
            methods[method] = _method_result(
                _rankings(test_query_ids, test_doc_ids, predictions.tolist()), test, settings
            )
            model_paths[method] = model_path
            del predictions
            gc.collect()

    result: dict[str, Any] = {
        "dataset": name,
        "train_split": "train",
        "test_split": "test",
        "candidate_k": settings.ltr_candidate_k,
        "training_queries": len(group_sizes),
        "skipped_training_queries": skipped,
        "train_candidates": int(sum(group_sizes)),
        "test_queries": len(test.queries),
        "test_candidates": len(test_rows),
        "feature_names": list(FEATURE_NAMES),
        "methods": methods,
    }
    result["paired_permutation_p_values"] = _comparisons(methods, settings)
    output = settings.results_dir / f"{name}-phase2.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    _log_runs(result, output, model_paths, settings)
    return output
