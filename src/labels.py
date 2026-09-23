"""Label and query-weight construction from the dump's engage_score field."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def parse_engage(raw: str) -> float:
    if raw in ("", "NA", "null", "None", "\\N"):
        return 0.0
    return float(raw)


def binary_label(raw: str) -> int:
    return 1 if parse_engage(raw) >= 1 else 0


def query_weight(qids: Sequence[str], engages: Sequence[str]) -> Dict[str, float]:
    weights: Dict[str, float] = defaultdict(float)
    for qid, raw in zip(qids, engages):
        weights[qid] = max(weights[qid], parse_engage(raw))
    return dict(weights)


def make_labels(
    qids: Sequence[str], engages: Sequence[str]
) -> Tuple[List[int], List[float], Dict[str, float]]:
    labels = [binary_label(raw) for raw in engages]
    group_w = query_weight(qids, engages)
    row_w = [group_w[qid] for qid in qids]
    return labels, row_w, group_w
