"""Pre-train sanity checks. Pipeline refuses to start if this fails."""
from __future__ import annotations

import csv
import json
from collections import Counter
from typing import Dict, List

from . import config
from .features import FEATURE_SLOTS, META_COLS, MISSING_TOKENS

DRIFT_COLS = ("topic_match", "city_match")
DRIFT_ABS_TOL = 0.08


def _load(path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: List[Dict[str, str]], col: str) -> float:
    vals = []
    for row in rows:
        raw = row.get(col, "")
        if raw in MISSING_TOKENS:
            continue
        vals.append(float(raw))
    return float(sum(vals) / len(vals)) if vals else 0.0


def run_data_check() -> dict:
    schema = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))
    train = _load(config.TRAIN_CSV)
    test = _load(config.TEST_CSV)
    train_cols = set(train[0].keys()) if train else set()
    test_cols = set(test[0].keys()) if test else set()
    expected = set(schema["meta"]) | set(schema["features"])
    schema_ok = train_cols == expected and test_cols == expected
    empty_qid = sum(1 for row in train + test if not row.get("query_id"))
    drift = {}
    drift_ok = True
    for col in DRIFT_COLS:
        tr = _mean(train, col)
        te = _mean(test, col)
        diff = abs(tr - te)
        drift[col] = {"train": round(tr, 6), "test": round(te, 6), "abs_diff": round(diff, 6)}
        if diff > DRIFT_ABS_TOL:
            drift_ok = False
    train_dates = sorted(Counter(row["dt"] for row in train))
    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_queries": len({row["query_id"] for row in train}),
        "test_queries": len({row["query_id"] for row in test}),
        "train_partitions": train_dates,
        "feature_slots": list(FEATURE_SLOTS),
        "meta_cols": sorted(META_COLS),
        "schema_ok": schema_ok,
        "empty_query_id": empty_qid,
        "feature_drift": drift,
        "drift_ok": drift_ok,
        "passed": bool(schema_ok and empty_qid == 0 and drift_ok and train and test),
    }
    return report
