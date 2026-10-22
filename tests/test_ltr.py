from pathlib import Path

import numpy as np
import pytest

from evidence_rag.ltr import train_and_predict


@pytest.mark.parametrize(("kind", "suffix"), [("catboost", "cbm"), ("xgboost", "json")])
def test_rankers_train_predict_and_persist(kind: str, suffix: str, tmp_path: Path) -> None:
    features = np.asarray(
        [
            [2.0, 0.9],
            [0.1, 0.2],
            [1.5, 0.8],
            [0.2, 0.1],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
    model_path = tmp_path / f"ranker.{suffix}"

    predictions = train_and_predict(
        kind,  # type: ignore[arg-type]
        features,
        labels,
        [2, 2],
        features,
        labels,
        [2, 2],
        features,
        ["score", "other"],
        model_path,
        seed=42,
        iterations=10,
    )

    predictions, best_iteration = predictions
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    assert best_iteration >= 1
    assert model_path.is_file()
