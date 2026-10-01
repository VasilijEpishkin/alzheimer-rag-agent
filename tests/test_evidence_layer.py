"""Unit tests for evidence_layer.py."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_rag.models import Document
from evidence_rag.evidence_layer import (
    StanceOutput,
    _stance,
    answerability_features,
    evidence_features,
    is_retracted,
    study_design_weight,
)
from evidence_rag.reranking import Candidate


def test_study_design_weight():
    """Test study_design_weight returns correct weights based on publication types."""
    # Meta-analysis or systematic review -> 5.0
    doc_meta = Document(
        doc_id="1", title="Meta", text="", publication_types=("Meta-Analysis",)
    )
    assert study_design_weight(doc_meta) == 5.0

    doc_sys = Document(
        doc_id="2", title="Sys", text="", publication_types=("Systematic Review",)
    )
    assert study_design_weight(doc_sys) == 5.0

    # Randomized controlled trial -> 4.0
    doc_rct = Document(
        doc_id="3", title="RCT", text="", publication_types=("Randomized Controlled Trial",)
    )
    assert study_design_weight(doc_rct) == 4.0

    # Case reports -> 1.0
    doc_case = Document(
        doc_id="4", title="Case", text="", publication_types=("Case Reports",)
    )
    assert study_design_weight(doc_case) == 1.0

    # Unknown publication type -> 0.0
    doc_unknown = Document(
        doc_id="5", title="Unknown", text="", publication_types=("Some Other Type",)
    )
    assert study_design_weight(doc_unknown) == 0.0

    # Multiple publication types -> max weight
    doc_multi = Document(
        doc_id="6",
        title="Multi",
        text="",
        publication_types=("Case Reports", "Randomized Controlled Trial"),
    )
    assert study_design_weight(doc_multi) == 4.0

    # Empty publication types -> 0.0
    doc_empty = Document(doc_id="7", title="Empty", text="", publication_types=())
    assert study_design_weight(doc_empty) == 0.0


def test_is_retracted():
    """Test is_retracted correctly identifies retracted documents."""
    # Retracted publication type -> but is_retracted field must be set True
    doc_retracted = Document(
        doc_id="1",
        title="Retracted",
        text="",
        publication_types=("Retracted Publication",),
        is_retracted=True,
    )
    assert is_retracted(doc_retracted) is True

    doc_retracted2 = Document(
        doc_id="2",
        title="Retracted2",
        text="",
        publication_types=("Retraction of Publication",),
        is_retracted=True,
    )
    assert is_retracted(doc_retracted2) is True

    # Not retracted
    doc_not = Document(
        doc_id="3", title="Not", text="", publication_types=("Journal Article",), is_retracted=False
    )
    assert is_retracted(doc_not) is False

    # Empty publication types
    doc_empty = Document(doc_id="4", title="Empty", text="", publication_types=(), is_retracted=False)
    assert is_retracted(doc_empty) is False


def test__stance_placeholder():
    """Test that _stance returns uniform probabilities (placeholder model)."""
    # The placeholder model returns [1/3, 1/3, 1/3] for any input.
    query = "What is the effect of drug X on disease Y?"
    document = Document(doc_id="1", title="Some paper", text="This is the abstract.")
    stance = _stance(query, document)
    assert isinstance(stance, StanceOutput)
    assert stance.supports == pytest.approx(1.0 / 3.0)
    assert stance.contradicts == pytest.approx(1.0 / 3.0)
    assert stance.neutral == pytest.approx(1.0 / 3.0)


def test_evidence_features():
    """Test evidence_features builds the expected feature vector."""
    # We'll create a minimal candidate and document.
    query = "test query"
    document = Document(
        doc_id="1",
        title="Test Title",
        text="Test abstract",
        year=2020,
        mesh_headings=("Mesh1", "Mesh2"),
        publication_types=("Randomized Controlled Trial",),
        is_retracted=False,
    )
    candidate = Candidate(
        query_id="q1",
        doc_id="1",
        bm25_score=0.5,
        dense_score=0.3,
        bm25_rank=2,
        dense_rank=3,
        cross_score=0.1,
    )

    features = evidence_features(query, document, candidate)

    # Expected length: original features (from build_features) + 3 stance + 1 study_design_weight
    # build_features returns 12 features (see reranking.py: FEATURE_NAMES has 12 items)
    # Then we add 3 stance probabilities and 1 study_design weight -> total 16
    assert features.shape == (16,)
    assert features.dtype == np.float32

    # We can also check that the first 12 features are the same as build_features.
    from evidence_rag.reranking import build_features

    base_features = build_features(query, document, candidate)
    np.testing.assert_array_equal(features[:12], base_features)

    # The next three should be the stance probabilities (uniform in placeholder)
    np.testing.assert_array_equal(
        features[12:15], np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    )
    # The last should be the study design weight (4.0 for RCT)
    assert features[15] == 4.0


def test_answerability_features():
    """Test answerability_features currently just calls evidence_features (placeholder)."""
    query = "test query"
    document = Document(
        doc_id="1",
        title="Test Title",
        text="Test abstract",
        year=2020,
        mesh_headings=("Mesh1",),
        publication_types=("Journal Article",),
        is_retracted=False,
    )
    candidate = Candidate(
        query_id="q1",
        doc_id="1",
        bm25_score=0.5,
        dense_score=0.3,
        bm25_rank=2,
        dense_rank=3,
        cross_score=0.1,
    )

    feats1 = evidence_features(query, document, candidate)
    feats2 = answerability_features(query, document, candidate)
    np.testing.assert_array_equal(feats1, feats2)