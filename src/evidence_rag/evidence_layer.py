"""Evidence layer: study-design weighting, retraction filter, stance detection."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from evidence_rag.models import Document
from evidence_rag.reranking import Candidate, FEATURE_NAMES, build_features

# Placeholder for biomedical NLI model; in production, load a real model.
# For unit tests, we can mock this.
_STANCE_MODEL = None


def _load_stance_model() -> Any:
    """Load a biomedical NLI model for stance detection.
    Returns a callable that takes (query, document_text) and returns
    probabilities for [supports, contradicts, neutral].
    """
    global _STANCE_MODEL
    if _STANCE_MODEL is None:
        # In a real implementation, we would load a model here.
        # For now, we return a uniform distribution as placeholder.
        # This is intentional to avoid fabricating data.
        def _placeholder(query: str, document_text: str) -> list[float]:
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        _STANCE_MODEL = _placeholder
    return _STANCE_MODEL


@dataclass(frozen=True, slots=True)
class StanceOutput:
    """Stance detection output for a query-document pair."""
    supports: float
    contradicts: float
    neutral: float


def _stance(query: str, document: Document) -> StanceOutput:
    """Run stance detection on a query-document pair."""
    model = _load_stance_model()
    probs = model(query, document.text)
    return StanceOutput(supports=probs[0], contradicts=probs[1], neutral=probs[2])


def study_design_weight(document: Document) -> float:
    """Return study-design weight from PublicationType."""
    # Reuse evidence_level from reranking for consistency
    from evidence_rag.reranking import _evidence_level

    return _evidence_level(document)


def is_retracted(document: Document) -> bool:
    """Return True if document is retracted."""
    return document.is_retracted


def evidence_features(
    query: str, document: Document, candidate: Candidate
) -> np.ndarray:
    """Build feature vector for evidence layer (study-design, stance, etc.)."""
    base = build_features(query, document, candidate)
    stance = _stance(query, document)
    # Append stance probabilities and study-design weight
    return np.asarray(
        [
            *base,
            stance.supports,
            stance.contradicts,
            stance.neutral,
            study_design_weight(document),
        ],
        dtype=np.float32,
    )


def answerability_features(
    query: str, document: Document, candidate: Candidate
) -> np.ndarray:
    """Build feature vector for answerability model.
    For now, reuse evidence_features as a placeholder.
    In a real implementation, we might use different features.
    """
    return evidence_features(query, document, candidate)


__all__ = [
    "StanceOutput",
    "_stance",
    "study_design_weight",
    "is_retracted",
    "evidence_features",
    "answerability_features",
]