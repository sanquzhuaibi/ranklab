"""Label and query-weight construction from the dump."""
from __future__ import annotations

import csv
from typing import Dict, List, Sequence, Tuple

from . import config


def parse_engage(raw: str) -> float:
    if raw in ("", "NA", "null", "None", "\\N"):
        return 0.0
    return float(raw)


def binary_label(raw: str) -> int:
    return 1 if parse_engage(raw) >= 1 else 0


def load_query_weights() -> Dict[str, float]:
    path = config.DATA_DIR / "query_weight.csv"
    weights: Dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            weights[row["query_id"]] = float(row["weight"])
    return weights


def make_labels(
    qids: Sequence[str], engages: Sequence[str]
) -> Tuple[List[int], List[float], Dict[str, float]]:
    labels = [binary_label(raw) for raw in engages]
    stored = load_query_weights()
    row_w = [stored.get(qid, 1.0) for qid in qids]
    group_w = {qid: stored.get(qid, 1.0) for qid in qids}
    return labels, row_w, group_w
