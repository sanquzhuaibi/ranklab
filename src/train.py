"""Fit a ranking scorer.

Query-level weights from the label module are passed through as sample
weights so high-engagement requests are not drowned out.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config


def train_ranker(matrix: List[List[float]], y: Sequence[int], qids: Sequence[str], row_w: Sequence[float]):
    x = np.asarray(matrix, dtype=float)
    labels = np.asarray(y, dtype=int)
    sample_w = np.asarray(row_w, dtype=float)
    model = HistGradientBoostingClassifier(
        max_depth=config.MAX_DEPTH,
        learning_rate=config.LEARNING_RATE,
        max_iter=config.MAX_ITER,
        random_state=config.RANDOM_SEED,
    )
    model.fit(x, labels, sample_weight=sample_w)
    return model


def predict_scores(model, matrix: List[List[float]]) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    proba = model.predict_proba(x)
    positive = list(model.classes_).index(1)
    return proba[:, positive]
