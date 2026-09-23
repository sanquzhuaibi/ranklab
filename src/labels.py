"""Label construction from the dump's engage_score field."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def parse_engage(raw: str) -> float:
    if raw in ("", "NA", "null", "None", "\\N"):
        return 0.0
    return float(raw)


def binary_label(raw: str) -> int:
    return 1 if parse_engage(raw) >= 1 else 0


def make_labels(
    qids: Sequence[str], engages: Sequence[str]
) -> Tuple[List[int], List[float], Dict[str, float]]:
    labels = [binary_label(raw) for raw in engages]
    row_w = [1.0] * len(labels)
    group_w = {qid: 1.0 for qid in qids}
    return labels, row_w, group_w
