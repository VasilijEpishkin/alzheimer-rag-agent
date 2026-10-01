import gc
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

ModelKind = Literal["catboost", "xgboost"]


def train_and_predict(
    kind: ModelKind,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_group_sizes: Sequence[int],
    test_features: np.ndarray,
    feature_names: Sequence[str],
    model_path: Path,
    *,
    seed: int,
    iterations: int,
) -> np.ndarray:
    """Train one ranker, predict once, persist it, then release its memory."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "catboost":
        from catboost import CatBoostRanker, Pool

        group_ids = np.repeat(np.arange(len(train_group_sizes)), train_group_sizes)
        train_pool = Pool(
            train_features,
            label=train_labels,
            group_id=group_ids,
            feature_names=list(feature_names),
        )
        model = CatBoostRanker(
            loss_function="YetiRank",
            iterations=iterations,
            depth=6,
            learning_rate=0.05,
            random_seed=seed,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(train_pool)
        predictions = np.asarray(model.predict(test_features), dtype=np.float64)
        model.save_model(model_path)
    elif kind == "xgboost":
        from xgboost import XGBRanker

        model = XGBRanker(
            objective="rank:pairwise",
            n_estimators=iterations,
            max_depth=6,
            learning_rate=0.05,
            random_state=seed,
            n_jobs=2,
            tree_method="hist",
        )
        model.fit(
            train_features,
            train_labels,
            group=np.asarray(train_group_sizes, dtype=np.int32),
            verbose=False,
        )
        predictions = np.asarray(model.predict(test_features), dtype=np.float64)
        model.save_model(model_path)
    else:
        raise ValueError(f"Unknown LTR model kind: {kind}")
    del model
    gc.collect()
    return predictions
