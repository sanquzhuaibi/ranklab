"""Offline metrics.

Primary metric is Recall@10. v2 GAUC weights each request by max(engage_score).
"""
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


def recall_at(ranked_labels: List[int], k: int) -> float:
    total = sum(ranked_labels)
    if total <= 0:
        return 0.0
    k = min(k, len(ranked_labels))
    return sum(ranked_labels[:k]) / total


def group_auc(y: np.ndarray, pred: np.ndarray, qids: Sequence[str], engages: Sequence[str] | None = None):
    """v2 offline GAUC.

    Query AUC weighted by max(engage_score) so deep-interaction requests
    move the number more. v2 offline definition.
    """
    from .labels import parse_engage

    buckets = _grouped(qids, y.tolist(), pred.tolist())
    eng_by_q = defaultdict(list)
    if engages is not None:
        for qid, raw in zip(qids, engages):
            eng_by_q[qid].append(parse_engage(raw))
    vals, weights = [], []
    skipped = 0
    for qid, pairs in buckets.items():
        labels = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]
        if len(set(labels)) < 2:
            skipped += 1
            continue
        vals.append(float(roc_auc_score(labels, scores)))
        if qid in eng_by_q:
            weights.append(max(eng_by_q[qid]))
        else:
            weights.append(float(sum(labels)))
    gauc = float(np.average(vals, weights=weights)) if vals else float("nan")
    return gauc, skipped, len(vals)


def ctr_deciles(y: np.ndarray, pred: np.ndarray) -> List[dict]:
    order = np.sort(pred)
    n = len(pred)
    bins = []
    for b in range(10):
        lo = order[int(n * b / 10)]
        hi = order[min(n - 1, int(n * (b + 1) / 10) - 1)]
        mask = (pred >= lo) & (pred <= hi)
        cnt = int(mask.sum())
        pos = int(y[mask].sum()) if cnt else 0
        bins.append({"bucket": b + 1, "rows": cnt, "positives": pos, "ctr": pos / cnt if cnt else 0.0})
    return bins


def evaluate(y: Sequence[int], pred: Sequence[float], qids: Sequence[str], engages: Sequence[str] | None = None) -> dict:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(pred, dtype=float)
    auc = float(roc_auc_score(y_arr, p_arr))
    gauc, skipped, eligible = group_auc(y_arr, p_arr, qids, engages)
    buckets = _grouped(qids, y, pred)
    ndcg = {f"@{k}": [] for k in config.NDCG_KS}
    recall = {f"@{k}": [] for k in config.NDCG_KS}
    sizes = []
    for pairs in buckets.values():
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        labels = [p[0] for p in pairs]
        sizes.append(len(labels))
        for k in config.NDCG_KS:
            ndcg[f"@{k}"].append(ndcg_at(labels, k))
            recall[f"@{k}"].append(recall_at(labels, k))
    metrics = {
        "primary": config.PRIMARY_METRIC,
        "roc_auc": auc,
        "gauc": gauc,
        "gauc_definition": "v2_offline_weighted_by_max_engage",
        "gauc_skipped_groups": skipped,
        "gauc_eligible_groups": eligible,
        "ndcg": {k: float(np.mean(v)) for k, v in ndcg.items()},
        "recall": {k: float(np.mean(v)) for k, v in recall.items()},
        "recall@10": float(np.mean(recall["@10"])),
        "n_rows": int(len(y_arr)),
        "n_queries": int(len(buckets)),
        "group_size_min": int(min(sizes)),
        "group_size_max": int(max(sizes)),
        "group_size_mean": float(np.mean(sizes)),
        "ctr_deciles": ctr_deciles(y_arr, p_arr),
        "feature_dim": None,
    }
    return metrics
