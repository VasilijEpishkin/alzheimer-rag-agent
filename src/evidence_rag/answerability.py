"""Answerability model: calibrated CatBoost for abstention decision."""

from __future__ import annotations

import gc
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evidence_rag.config import Settings
from evidence_rag.reranking import Candidate

try:
    from catboost import CatBoostClassifier, Pool
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore
    IsotonicRegression = None  # type: ignore


_ANSWERABILITY_EVIDENCE_K = 5
_ANSWERABILITY_LABEL = (
    "positive when at least one of the top five cross-encoder-ranked candidates "
    "has a positive BEIR qrel; otherwise negative"
)


def aggregate_answerability_features(features: np.ndarray) -> np.ndarray:
    """Summarize the top five reranked documents with per-feature mean and maximum."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array")
    features = features[:_ANSWERABILITY_EVIDENCE_K]
    return np.concatenate((features.mean(axis=0), features.max(axis=0))).astype(np.float32)


def _validate_training_data(features: np.ndarray, labels: np.ndarray, name: str) -> None:
    if features.ndim != 2 or features.shape[0] != labels.shape[0] or not features.shape[0]:
        raise ValueError(f"{name} features and labels must have matching non-empty rows")
    if not np.isfinite(features).all():
        raise ValueError(f"{name} features must be finite")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{name} labels must contain both binary classes 0 and 1")


def _train_answerability_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    model_path: Path,
    *,
    seed: int,
    iterations: int,
) -> Any:
    """Fit CatBoost on train rows and isotonic calibration on separate dev rows."""
    if CatBoostClassifier is None:  # pragma: no cover
        raise RuntimeError("CatBoost is not installed. Install catboost>=1.2,<2")
    if IsotonicRegression is None:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for answerability calibration")
    _validate_training_data(train_features, train_labels, "training")
    _validate_training_data(calibration_features, calibration_labels, "calibration")
    if train_features.shape[1] != calibration_features.shape[1]:
        raise ValueError("training and calibration feature counts must match")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    train_pool = Pool(
        train_features,
        label=train_labels,
        feature_names=[f"feat_{i}" for i in range(train_features.shape[1])],
    )
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        random_seed=seed,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool)
    calibration_probs = model.predict_proba(calibration_features)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        calibration_probs, calibration_labels
    )
    calibration_path = model_path.with_suffix(".calibration.json")
    descriptor, temporary_path = tempfile.mkstemp(dir=model_path.parent, suffix=".cbm")
    os.close(descriptor)
    try:
        model.save_model(temporary_path)
        os.replace(temporary_path, model_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    descriptor, temporary_path = tempfile.mkstemp(dir=model_path.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                {
                    "method": "isotonic",
                    "x_thresholds": calibrator.X_thresholds_.tolist(),
                    "y_thresholds": calibrator.y_thresholds_.tolist(),
                },
                stream,
                allow_nan=False,
            )
        os.replace(temporary_path, calibration_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return model


def _calibrate_probabilities(probs: np.ndarray, calibration_path: Path) -> np.ndarray:
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("method") != "isotonic":
        raise ValueError(f"Unsupported calibration method in {calibration_path}")
    x_thresholds = np.asarray(calibration.get("x_thresholds"), dtype=np.float64)
    y_thresholds = np.asarray(calibration.get("y_thresholds"), dtype=np.float64)
    if (
        x_thresholds.ndim != 1
        or y_thresholds.ndim != 1
        or not len(x_thresholds)
        or len(x_thresholds) != len(y_thresholds)
        or not np.isfinite(x_thresholds).all()
        or not np.isfinite(y_thresholds).all()
        or np.any(np.diff(x_thresholds) < 0)
        or np.any(np.diff(y_thresholds) < 0)
    ):
        raise ValueError(f"Invalid isotonic calibration thresholds in {calibration_path}")
    return np.interp(probs, x_thresholds, y_thresholds).clip(0.0, 1.0)


def predict_answerability(
    features: np.ndarray,
    model_path: Path,
) -> np.ndarray:
    """Predict probability of having enough evidence to answer.
    Returns an array of probabilities for the positive class.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    calibration_path = model_path.with_suffix(".calibration.json")
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    if CatBoostClassifier is None:  # pragma: no cover
        raise RuntimeError("CatBoost is not installed. Install catboost>=1.2,<2")

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    # Predict probabilities for the positive class
    probs = _calibrate_probabilities(
        model.predict_proba(features)[:, 1], calibration_path
    )
    del model
    gc.collect()
    return probs


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).
    Returns the ECE as a float.
    """
    if len(probs) == 0:
        return 0.0
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_bounds[:-1]
    bin_uppers = bin_bounds[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=True):
        # Determine which samples are in the bin
        in_bin = (probs >= bin_lower if bin_lower == 0 else probs > bin_lower) & (
            probs <= bin_upper
        )
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability diagram: bin accuracies and average confidences.
    Returns (bin_centers, bin_accuracies, bin_confidences).
    """
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_bounds[:-1]
    bin_uppers = bin_bounds[1:]

    bin_centers = (bin_lowers + bin_uppers) / 2
    bin_accuracies = []
    bin_confidences = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=True):
        in_bin = (probs >= bin_lower if bin_lower == 0 else probs > bin_lower) & (
            probs <= bin_upper
        )
        if in_bin.any():
            bin_accuracies.append(labels[in_bin].mean())
            bin_confidences.append(probs[in_bin].mean())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
    return bin_centers, np.array(bin_accuracies), np.array(bin_confidences)


def _answerability_rows(
    benchmark: Any, rows: Sequence[Candidate]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from evidence_rag.evidence_layer import answerability_features

    by_query: dict[str, list[Candidate]] = {}
    for candidate in rows:
        by_query.setdefault(candidate.query_id, []).append(candidate)
    query_features: list[np.ndarray] = []
    labels: list[int] = []
    query_ids: list[str] = []
    for query_id, query in benchmark.queries.items():
        top_candidates = sorted(
            by_query.get(query_id, ()), key=lambda candidate: candidate.cross_score, reverse=True
        )[:_ANSWERABILITY_EVIDENCE_K]
        if not top_candidates:
            continue
        features = np.stack(
            [
                answerability_features(
                    query, benchmark.documents_by_id[candidate.doc_id], candidate
                )
                for candidate in top_candidates
            ]
        )
        query_features.append(aggregate_answerability_features(features))
        query_ids.append(query_id)
        labels.append(
            int(
                any(
                    benchmark.qrels.get(query_id, {}).get(candidate.doc_id, 0) > 0
                    for candidate in top_candidates
                )
            )
        )
    if not query_features:
        raise ValueError(
            f"No retrieved candidates available for {benchmark.name}/{benchmark.split}"
        )
    return np.stack(query_features), np.asarray(labels, dtype=np.int8), query_ids


def _split_calibration_rows(
    features: np.ndarray, labels: np.ndarray, seed: int, fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratify a deterministic calibration holdout from a split without dev data."""
    rng = np.random.default_rng(seed)
    calibration_indices: list[int] = []
    train_indices: list[int] = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("A train split needs at least two rows from each proxy-label class")
        rng.shuffle(indices)
        calibration_count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        calibration_indices.extend(indices[:calibration_count].tolist())
        train_indices.extend(indices[calibration_count:].tolist())
    rng.shuffle(calibration_indices)
    rng.shuffle(train_indices)
    return (
        features[train_indices],
        labels[train_indices],
        features[calibration_indices],
        labels[calibration_indices],
    )


def _write_reliability_svg(probs: np.ndarray, labels: np.ndarray, path: Path) -> None:
    centers, accuracies, confidences = reliability_diagram(probs, labels)
    counts, _ = np.histogram(probs, bins=np.linspace(0, 1, len(centers) + 1))
    points = "\n".join(
        f'<circle cx="{80 + confidence * 500:.1f}" cy="{330 - accuracy * 300:.1f}" '
        'r="5" fill="#1769aa" />'
        for count, accuracy, confidence in zip(counts, accuracies, confidences, strict=True)
        if count
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" '
        'viewBox="0 0 640 420">'
        '<rect width="640" height="420" fill="white" />'
        '<path d="M80 330 L580 30" stroke="#777" stroke-dasharray="6 5" />'
        '<path d="M80 30 V330 H580" fill="none" stroke="#222" />'
        f'{points}'
        '<text x="330" y="390" text-anchor="middle">Mean predicted probability</text>'
        '<text x="20" y="180" transform="rotate(-90 20 180)" '
        'text-anchor="middle">Observed positive fraction</text>'
        '<text x="320" y="20" text-anchor="middle">Answerability reliability (test split)</text>'
        '</svg>\n'
    )


def evaluate_answerability(name: str, settings: Settings) -> Path:
    """Train on BEIR train, calibrate on dev, and report metrics on test."""
    from evidence_rag.datasets import load_beir
    from evidence_rag.phase2 import _get_candidates
    from evidence_rag.pubmed import enrich_benchmark_with_pubmed

    settings.ensure_directories()
    split_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    split_queries: dict[str, int] = {}
    for split in ("train", "test"):
        benchmark = load_beir(name, split=split)
        enrich_benchmark_with_pubmed(benchmark, settings)
        rows = _get_candidates(benchmark, settings)
        features, labels, _ = _answerability_rows(benchmark, rows)
        _validate_training_data(features, labels, split)
        split_data[split] = features, labels
        split_queries[split] = len(labels)
        del rows, benchmark
        gc.collect()

    try:
        calibration_benchmark = load_beir(name, split="dev")
    except ValueError as error:
        if "Unknown" not in str(error) or "split" not in str(error):
            raise
        train_features, train_labels, calibration_features, calibration_labels = (
            _split_calibration_rows(*split_data["train"], settings.random_seed)
        )
        split_data["train"] = train_features, train_labels
        split_data["dev"] = calibration_features, calibration_labels
        split_queries["train"] = len(train_labels)
        calibration_source = "stratified 20% holdout from official train split"
        calibration_split_name = "train_holdout"
    else:
        enrich_benchmark_with_pubmed(calibration_benchmark, settings)
        calibration_rows = _get_candidates(calibration_benchmark, settings)
        calibration_features, calibration_labels, _ = _answerability_rows(
            calibration_benchmark, calibration_rows
        )
        _validate_training_data(calibration_features, calibration_labels, "dev")
        split_data["dev"] = calibration_features, calibration_labels
        split_queries["dev"] = len(calibration_labels)
        calibration_source = "official dev split"
        calibration_split_name = "dev"
        del calibration_rows, calibration_benchmark
        gc.collect()
    if "dev" not in split_queries:
        split_queries["dev"] = len(split_data["dev"][1])

    model_path = settings.phase2_dir / "answerability.cbm"
    model = _train_answerability_model(
        *split_data["train"],
        *split_data["dev"],
        model_path,
        seed=settings.random_seed,
        iterations=settings.ltr_iterations,
    )
    test_features, test_labels = split_data["test"]
    raw_probs = np.asarray(model.predict_proba(test_features)[:, 1], dtype=np.float64)
    calibrated_probs = _calibrate_probabilities(
        raw_probs, model_path.with_suffix(".calibration.json")
    )
    bins, accuracies, confidences = reliability_diagram(calibrated_probs, test_labels)
    plot_path = settings.results_dir / f"{name}-answerability-reliability.svg"
    _write_reliability_svg(calibrated_probs, test_labels, plot_path)
    result = {
        "dataset": name,
        "label_proxy": _ANSWERABILITY_LABEL,
        "evidence_k": _ANSWERABILITY_EVIDENCE_K,
        "feature_aggregation": "per-feature mean and maximum over top candidates",
        "train_split": "train",
        "calibration_split": calibration_split_name,
        "calibration_source": calibration_source,
        "test_split": "test",
        "queries": {
            output_name: {
                "count": split_queries[split],
                "positive_count": int(split_data[split][1].sum()),
                "positive_rate": float(split_data[split][1].mean()),
            }
            for output_name, split in (
                ("train", "train"),
                ("calibration", "dev"),
                ("test", "test"),
            )
        },
        "test_metrics": {
            "raw_ece": compute_ece(raw_probs, test_labels),
            "calibrated_ece": compute_ece(calibrated_probs, test_labels),
        },
        "reliability": {
            "bin_centers": bins.tolist(),
            "observed_positive_rate": accuracies.tolist(),
            "mean_predicted_probability": confidences.tolist(),
        },
        "model": str(model_path),
        "calibration": str(model_path.with_suffix(".calibration.json")),
        "reliability_plot": str(plot_path),
    }
    output = settings.results_dir / f"{name}-answerability.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    import mlflow

    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri
        if "://" in settings.mlflow_tracking_uri
        else Path(settings.mlflow_tracking_uri).resolve().as_uri()
    )
    mlflow.set_experiment(settings.phase3_mlflow_experiment)
    with mlflow.start_run(run_name=f"{name}-answerability"):
        mlflow.log_params(
            {
                "dataset": name,
                "label_proxy": _ANSWERABILITY_LABEL,
                "evidence_k": _ANSWERABILITY_EVIDENCE_K,
                "train_queries": split_queries["train"],
                "calibration_queries": split_queries["dev"],
                "test_queries": split_queries["test"],
                "calibration_source": calibration_source,
                "iterations": settings.ltr_iterations,
                "seed": settings.random_seed,
            }
        )
        mlflow.log_metrics(result["test_metrics"])
        mlflow.log_artifact(str(output))
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(
            str(model_path.with_suffix(".calibration.json")), artifact_path="models"
        )
    del model
    gc.collect()
    return output


__all__ = [
    "_train_answerability_model",
    "aggregate_answerability_features",
    "evaluate_answerability",
    "predict_answerability",
    "compute_ece",
    "reliability_diagram",
]
