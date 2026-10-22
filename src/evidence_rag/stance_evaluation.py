"""Held-out SciFact stance evaluation for the pinned NLI adapter."""

from __future__ import annotations

import gc
import json
import os
import tarfile
import tempfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import requests

from evidence_rag.config import Settings
from evidence_rag.datasets import load_beir
from evidence_rag.evidence_layer import _stance
from evidence_rag.models import Document

_SCIFACT_ARCHIVE = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
_MODEL_ID = "rishhh/verisci-claim-verifier-sentence-nli"
_MODEL_REVISION = "479bd57f1e3ed6a3f8775c3b8d1b5ca58562a111"
_LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
_SOURCE_LABELS = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}


def _claims_path(settings: Settings) -> Path:
    path = settings.cache_dir / "scifact" / "claims_dev.jsonl"
    if path.exists():
        return path
    response = requests.get(_SCIFACT_ARCHIVE, timeout=120)
    response.raise_for_status()
    with tarfile.open(fileobj=BytesIO(response.content), mode="r:gz") as archive:
        member = archive.getmember("data/claims_dev.jsonl")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("SciFact archive does not contain claims_dev.jsonl")
        contents = stream.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".jsonl")
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _stance_examples(
    path: Path, documents: dict[str, Document]
) -> list[tuple[str, str, Document, str]]:
    examples: list[tuple[str, str, Document, str]] = []
    for line in path.read_text().splitlines():
        claim = json.loads(line)
        claim_id = str(claim["id"])
        evidence = claim.get("evidence", {})
        if evidence:
            for doc_id, rationales in evidence.items():
                document = documents.get(str(doc_id))
                if document is None:
                    continue
                for rationale in rationales:
                    label = _SOURCE_LABELS.get(rationale.get("label", "").upper())
                    if label:
                        examples.append((claim_id, claim["claim"], document, label))
        else:
            for doc_id in claim.get("cited_doc_ids", []):
                document = documents.get(str(doc_id))
                if document is not None:
                    examples.append((claim_id, claim["claim"], document, "NOT_ENOUGH_INFO"))
    return examples


def _metrics(gold: list[str], predicted: list[str]) -> dict[str, float]:
    accuracy = sum(
        first == second for first, second in zip(gold, predicted, strict=True)
    ) / len(gold)
    per_class = []
    for label in _LABELS:
        true_positive = sum(
            a == label and b == label for a, b in zip(gold, predicted, strict=True)
        )
        false_positive = sum(
            a != label and b == label for a, b in zip(gold, predicted, strict=True)
        )
        false_negative = sum(
            a == label and b != label for a, b in zip(gold, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_class.append(2 * true_positive / denominator if denominator else 0.0)
    return {"accuracy": accuracy, "macro_f1": sum(per_class) / len(_LABELS)}


def _cluster_bootstrap_ci(
    claim_ids: list[str], gold: list[str], predicted: list[str], samples: int, seed: int
) -> dict[str, list[float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, claim_id in enumerate(claim_ids):
        grouped[claim_id].append(index)
    ids = np.asarray(list(grouped))
    rng = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = {"accuracy": [], "macro_f1": []}
    for sampled_ids in rng.choice(ids, size=(samples, len(ids)), replace=True):
        indices = [index for claim_id in sampled_ids for index in grouped[str(claim_id)]]
        values = _metrics(
            [gold[index] for index in indices], [predicted[index] for index in indices]
        )
        for name, value in values.items():
            metrics[name].append(value)
    return {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in metrics.items()
    }


def evaluate_stance(settings: Settings) -> Path:
    """Evaluate support/refute/NEI on the labelled SciFact dev claims."""
    settings.ensure_directories()
    claims_path = _claims_path(settings)
    benchmark = load_beir("scifact", split="test")
    documents = benchmark.documents_by_id
    examples = _stance_examples(claims_path, documents)
    if not examples:
        raise ValueError("No SciFact dev claim-evidence pairs matched the BEIR corpus")

    claim_ids: list[str] = []
    gold: list[str] = []
    predicted: list[str] = []
    for claim_id, claim, document, label in examples:
        output = _stance(claim, document)
        scores = (output.supports, output.contradicts, output.neutral)
        claim_ids.append(claim_id)
        gold.append(label)
        predicted.append(_LABELS[int(np.argmax(scores))])
    del benchmark
    gc.collect()

    result = {
        "dataset": "SciFact",
        "split": "dev",
        "source": "allenai/scifact claims_dev.jsonl paired with the BEIR SciFact corpus",
        "model": _MODEL_ID,
        "model_revision": _MODEL_REVISION,
        "label_mapping": {
            "SUPPORT": "SUPPORTS",
            "CONTRADICT": "REFUTES",
            "empty annotated evidence paired with cited_doc_ids": "NOT_ENOUGH_INFO",
        },
        "claim_count": len(set(claim_ids)),
        "pair_count": len(gold),
        "class_counts": dict(Counter(gold)),
        "metrics": _metrics(gold, predicted),
        "ci95": _cluster_bootstrap_ci(
            claim_ids, gold, predicted, settings.bootstrap_samples, settings.random_seed
        ),
        "scores_calibrated": False,
    }
    output = settings.results_dir / "scifact-stance.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output
