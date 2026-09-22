"""Positional feature builder.

Warehouse dumps 16 numeric slots after meta. Names below are the v2 mapping
used in training — keep in sync with the dump, not with ad-hoc docs.
"""
from __future__ import annotations

import csv
import math
from typing import Dict, List, Sequence, Tuple

# 16 names for 16 numeric slots. source/dt are dump leftover columns.
FEATURE_SLOTS = [
    "quality",
    "freshness",
    "title_len",
    "author_score",
    "topic_match",
    "hist_ctr",
    "dwell_sec",
    "share_rate",
    "tag_n",
    "tag_unique_n",
    "like_per_exp",
    "share_per_exp",
    "city_match",
    "json_missing",
    "source",
    "dt",
]
META_DROP = {"source", "dt"}
MISSING_TOKENS = {"", "NA", "null", "None", "\\N"}


def _to_float(raw: str) -> float:
    if raw in MISSING_TOKENS:
        return 0.0
    value = float(raw)
    if not math.isfinite(value):
        return 0.0
    return value


def used_feature_names() -> List[str]:
    return [name for name in FEATURE_SLOTS if name not in META_DROP]


def load_rows(path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_vector(row: Dict[str, str], fieldnames: Sequence[str]) -> List[float]:
    numeric_keys = [key for key in fieldnames if key not in ("query_id", "item_id", "engage_score", "dt")]
    values = [_to_float(row.get(key, "")) for key in numeric_keys]
    if len(values) < len(FEATURE_SLOTS):
        values.extend([0.0] * (len(FEATURE_SLOTS) - len(values)))
    values = values[: len(FEATURE_SLOTS)]
    return [value for name, value in zip(FEATURE_SLOTS, values) if name not in META_DROP]


def build_matrix(path) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
    rows = load_rows(path)
    if not rows:
        raise SystemExit(f"empty dataset: {path}")
    fieldnames = list(rows[0].keys())
    qids, items, engages, matrix = [], [], [], []
    for row in rows:
        qids.append(row["query_id"])
        items.append(row["item_id"])
        engages.append(row.get("engage_score", ""))
        matrix.append(row_vector(row, fieldnames))
    return qids, items, engages, matrix
