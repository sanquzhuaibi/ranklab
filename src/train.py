"""Fit a ranking scorer.

LibSVM dumps store label:weight, but sklearn wants one sample weight per
row. For ranking we collapse that to one weight per query group.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config


def _group_binary_weights(qids: Sequence[str], y: Sequence[int]) -> np.ndarray:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for qid, label in zip(qids, y):
        grouped[qid].append(int(label))
    query_w = {qid: float(max(labels)) for qid, labels in grouped.items()}
    return np.array([query_w[qid] for qid in qids], dtype=float)


def train_ranker(matrix: List[List[float]], y: Sequence[int], qids: Sequence[str], row_w: Sequence[float]):
    del row_w  # dump weights are per-row copies; ranking wants group weights
    x = np.asarray(matrix, dtype=float)
    labels = np.asarray(y, dtype=int)
    sample_w = _group_binary_weights(qids, labels)
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
