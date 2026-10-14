"""Competition metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import TARGET_COLUMNS


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute macro ROC-AUC, skipping labels without both classes."""
    scores: list[float] = []
    for column in range(y_true.shape[1]):
        if np.unique(y_true[:, column]).size < 2:
            continue
        scores.append(float(roc_auc_score(y_true[:, column], y_score[:, column])))
    if not scores:
        raise ValueError("ROC-AUC is undefined: no target contains both classes")
    return float(np.mean(scores))


def validate_targets(columns: list[str]) -> None:
    missing = [column for column in TARGET_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
