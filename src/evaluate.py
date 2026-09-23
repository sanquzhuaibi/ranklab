"""Offline metrics used by this repo's ranking loop."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config


def _grouped(qids: Sequence[str], y: Sequence[int], pred: Sequence[float]):
    buckets: Dict[str, List[tuple]] = defaultdict(list)
    for qid, label, score in zip(qids, y, pred):
        buckets[qid].append((int(label), float(score)))
    return buckets


def ndcg_at(ranked_labels: List[int], k: int) -> float:
    k = min(k, len(ranked_labels))
    dcg = sum((2**lab - 1) / math.log2(i + 2) for i, lab in enumerate(ranked_labels[:k]))
    ideal = sorted(ranked_labels, reverse=True)
    idcg = sum((2**lab - 1) / math.log2(i + 2) for i, lab in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def group_auc(
    y: np.ndarray,
    pred: np.ndarray,
    qids: Sequence[str],
    query_weights: Dict[str, float] | None = None,
):
    """Per-query ROC-AUC, then a mean across queries."""
    buckets = _grouped(qids, y.tolist(), pred.tolist())
    vals, weights = [], []
    skipped = 0
    for qid, pairs in buckets.items():
        labels = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]
        if len(set(labels)) < 2:
            skipped += 1
            continue
        vals.append(float(roc_auc_score(labels, scores)))
        if query_weights is not None:
            weights.append(float(query_weights.get(qid, 1.0)))
        else:
            weights.append(1.0)
    gauc = float(np.average(vals, weights=weights)) if vals else float("nan")
    return gauc, skipped, len(vals)


def evaluate(
    y: Sequence[int],
    pred: Sequence[float],
    qids: Sequence[str],
    query_weights: Dict[str, float] | None = None,
) -> dict:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(pred, dtype=float)
    auc = float(roc_auc_score(y_arr, p_arr))
    gauc, skipped, eligible = group_auc(y_arr, p_arr, qids, query_weights)
    buckets = _grouped(qids, y, pred)
    ndcg = {f"@{k}": [] for k in config.NDCG_KS}
    for pairs in buckets.values():
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        labels = [p[0] for p in pairs]
        for k in config.NDCG_KS:
            ndcg[f"@{k}"].append(ndcg_at(labels, k))
    return {
        "primary": config.PRIMARY_METRIC,
        "roc_auc": auc,
        "gauc": gauc,
        "gauc_skipped_groups": skipped,
        "gauc_eligible_groups": eligible,
        "ndcg": {k: float(np.mean(v)) for k, v in ndcg.items()},
        "n_rows": int(len(y_arr)),
        "n_queries": int(len(buckets)),
        "feature_dim": None,
    }
