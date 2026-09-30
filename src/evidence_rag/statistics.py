from collections.abc import Mapping

import numpy as np


def bootstrap_confidence_intervals(
    per_query: Mapping[str, Mapping[str, float]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    if not per_query:
        return {}
    query_values = list(per_query.values())
    metrics = list(query_values[0])
    matrix = np.array([[row[metric] for metric in metrics] for row in query_values], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty((samples, len(metrics)), dtype=float)
    for start in range(0, samples, 1_000):
        count = min(1_000, samples - start)
        indices = rng.integers(0, len(matrix), size=(count, len(matrix)))
        means[start : start + count] = matrix[indices].mean(axis=1)
    bounds = np.percentile(means, [2.5, 97.5], axis=0)
    return {
        metric: (float(bounds[0, index]), float(bounds[1, index]))
        for index, metric in enumerate(metrics)
    }


def paired_permutation_test(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> float:
    common = sorted(first.keys() & second.keys())
    if not common:
        raise ValueError("Paired test requires at least one common query")
    differences = np.array([first[key] - second[key] for key in common], dtype=float)
    observed = abs(float(differences.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, samples, 1_000):
        count = min(1_000, samples - start)
        signs = rng.choice((-1.0, 1.0), size=(count, len(differences)))
        permuted = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed))
    return (extreme + 1) / (samples + 1)
