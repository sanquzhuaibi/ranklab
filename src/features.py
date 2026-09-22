"""Feature builder. Column names come from the warehouse schema."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from . import config

SCHEMA = json.loads(Path(config.SCHEMA_JSON).read_text(encoding="utf-8"))
FEATURE_SLOTS: List[str] = list(SCHEMA["features"])
META_COLS = set(SCHEMA["meta"])
MISSING_TOKENS = {"", "NA", "null", "None", "\\N"}


def _to_float(raw: str) -> float:
    if raw in MISSING_TOKENS:
        return 0.0
    value = float(raw)
    if not math.isfinite(value):
        return 0.0
    return value


def used_feature_names() -> List[str]:
    return list(FEATURE_SLOTS)


def load_rows(path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_vector(row: Dict[str, str], feature_names: Sequence[str]) -> List[float]:
    return [_to_float(row.get(name, "")) for name in feature_names]


def build_matrix(path) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
    rows = load_rows(path)
    if not rows:
        raise SystemExit(f"empty dataset: {path}")
    names = used_feature_names()
    qids, items, engages, matrix = [], [], [], []
    for row in rows:
        qids.append(row["query_id"])
        items.append(row["item_id"])
        engages.append(row.get("engage_score", ""))
        matrix.append(row_vector(row, names))
    return qids, items, engages, matrix
