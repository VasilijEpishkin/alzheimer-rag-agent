"""Unit tests for answerability.py."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_rag.answerability import compute_ece, reliability_diagram


def test_compute_ece_perfect_calibration():
    """Test ECE calculation with perfectly calibrated predictions."""
    # Perfect calibration: confidence equals accuracy in each bin
    # Let's make a perfectly calibrated example:
    # 10 samples in each bin: [0,0.1], (0.1,0.2], ..., (0.9,1.0]
    # In bin i, we want accuracy = confidence = bin midpoint
    n_bins = 10
    n_samples_per_bin = 10
    probs = []
    labels = []
    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins
        bin_midpoint = (bin_lower + bin_upper) / 2
        # Add n_samples_per_bin samples with confidence = bin_midpoint
        probs.extend([bin_midpoint] * n_samples_per_bin)
        # For labels: bin_midpoint fraction should be 1, rest 0
        n_ones = int(round(bin_midpoint * n_samples_per_bin))
        labels.extend([1] * n_ones + [0] * (n_samples_per_bin - n_ones))
    
    probs = np.array(probs)
    labels = np.array(labels)
    
    ece = compute_ece(probs, labels, n_bins=n_bins)
    # Should be close to 0 (not exactly due to rounding in label generation)
    assert ece < 0.06  # Allow a small margin for rounding


def test_compute_ece_worst_calibration():
    """Test ECE calculation with worst case calibration."""
    # All predictions are 0.6 but actual labels are all 0
    probs = np.full(10, 0.6)
    labels = np.zeros(10, dtype=int)
    ece = compute_ece(probs, labels, n_bins=10)
    # Expected ECE: |0.6 - 0.0| = 0.6 (since all samples in one bin around 0.6)
    assert ece == pytest.approx(0.6, abs=0.01)


def test_reliability_diagram_uniform():
    """Test reliability diagram with uniform predictions and labels."""
    # Predictions uniform in [0,1], labels random with mean 0.5
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, size=1000)
    # Generate labels such that P(label=1) = probs (perfect calibration)
    labels = rng.binomial(1, probs)
    
    bin_centers, bin_accuracies, bin_confidences = reliability_diagram(
        probs, labels, n_bins=10
    )
    
    # Check that we have 10 bins
    assert len(bin_centers) == 10
    assert len(bin_accuracies) == 10
    assert len(bin_confidences) == 10
    
    # Check bin centers are as expected
    expected_centers = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    np.testing.assert_array_almost_equal(bin_centers, expected_centers)
    
    # For perfect calibration, accuracies and confidences should be close to bin centers
    # (not checking exact values due to randomness, but they should be close)


def test_reliability_diagram_edge_cases():
    """Test reliability diagram with edge case predictions."""
    # All predictions 0.0
    probs = np.array([0.0, 0.0, 0.0])
    labels = np.array([0, 0, 0])
    bin_centers, bin_accuracies, bin_confidences = reliability_diagram(
        probs, labels, n_bins=10
    )
    # First bin should have all samples
    assert bin_accuracies[0] == 0.0
    assert bin_confidences[0] == 0.0
    # Other bins should be empty (0.0)
    assert np.all(bin_accuracies[1:] == 0.0)
    assert np.all(bin_confidences[1:] == 0.0)
    
    # All predictions 1.0
    probs = np.array([1.0, 1.0, 1.0])
    labels = np.array([1, 1, 1])
    bin_centers, bin_accuracies, bin_confidences = reliability_diagram(
        probs, labels, n_bins=10
    )
    # Last bin should have all samples
    assert bin_accuracies[-1] == 1.0
    assert bin_confidences[-1] == 1.0
    # Other bins should be empty
    assert np.all(bin_accuracies[:-1] == 0.0)
    assert np.all(bin_confidences[:-1] == 0.0)


__all__ = [
    "test_compute_ece_perfect_calibration",
    "test_compute_ece_worst_calibration",
    "test_reliability_diagram_uniform",
    "test_reliability_diagram_edge_cases",
]