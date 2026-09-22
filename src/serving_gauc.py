"""Serving-side GAUC.

This is the number the ranker dashboard uses. Offline experiments that
report a different `gauc` are not comparable to production.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def serving_gauc(y: Sequence[int], pred: Sequence[float], qids: Sequence[str]) -> dict:
    """Unweighted mean of per-request ROC-AUC on binary click.

    - label is click 0/1, never raw engage depth
    - each request counts as 1, regardless of how many clicks it has
    - requests with a single class are skipped (undefined AUC)
    """
    buckets = defaultdict(list)
    for qid, label, score in zip(qids, y, pred):
        buckets[str(qid)].append((int(label), float(score)))
    vals = []
    skipped = 0
    for pairs in buckets.values():
        labels = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]
        if len(set(labels)) < 2:
            skipped += 1
            continue
        vals.append(float(roc_auc_score(labels, scores)))
    return {
        "serving_gauc": float(np.mean(vals)) if vals else float("nan"),
        "serving_gauc_requests": len(vals),
        "serving_gauc_skipped": skipped,
        "serving_gauc_weight": "equal_request",
    }
