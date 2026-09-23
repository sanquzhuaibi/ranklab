"""Pre-train dump validation. Pipeline refuses to start if this fails."""
from __future__ import annotations

import csv
import json
from typing import Dict, List

from . import config
from .features import FEATURE_SLOTS, META_COLS, MISSING_TOKENS

# Frozen dump contract from the 2026-09-09 export.
EXPECTED_TRAIN_ROWS = 1944
EXPECTED_TEST_ROWS = 320
EXPECTED_TRAIN_QUERIES = 277
EXPECTED_TEST_QUERIES = 80
EXPECTED_TRAIN_PARTITIONS = 20
ENGAGE_VOCAB = {"", "0", "1", "3", "4", "10"}
REQUEST_FEATURE_TOL = {
    "topic_match": 0.08,
    "city_match": 0.08,
    "json_missing": 0.05,
}


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


def _ok(name: str, passed: bool, **detail):
    out = {"check": name, "ok": bool(passed)}
    out.update(detail)
    return out


def run_data_check() -> dict:
    schema = json.loads(config.SCHEMA_JSON.read_text(encoding="utf-8"))
    train = _load(config.TRAIN_CSV)
    test = _load(config.TEST_CSV)
    expected_cols = set(schema["meta"]) | set(schema["features"])
    train_cols = set(train[0].keys()) if train else set()
    test_cols = set(test[0].keys()) if test else set()

    train_q = {row["query_id"] for row in train}
    test_q = {row["query_id"] for row in test}
    train_dates = sorted({row["dt"] for row in train})
    test_dates = sorted({row["dt"] for row in test})
    empty_key = sum(
        1 for row in train + test if not row.get("query_id") or not row.get("item_id")
    )
    dup_train = len(train) - len({(row["query_id"], row["item_id"]) for row in train})
    dup_test = len(test) - len({(row["query_id"], row["item_id"]) for row in test})
    engage_bad = [
        row["engage_score"]
        for row in train + test
        if row.get("engage_score", "") not in ENGAGE_VOCAB
    ]

    request_features = {}
    request_ok = True
    for col, tol in REQUEST_FEATURE_TOL.items():
        tr = _mean(train, col)
        te = _mean(test, col)
        diff = abs(tr - te)
        request_features[col] = {
            "train": round(tr, 6),
            "test": round(te, 6),
            "abs_diff": round(diff, 6),
            "tol": tol,
        }
        if diff > tol:
            request_ok = False

    checks = [
        _ok(
            "row_counts",
            len(train) == EXPECTED_TRAIN_ROWS and len(test) == EXPECTED_TEST_ROWS,
            train_rows=len(train),
            test_rows=len(test),
            expected_train=EXPECTED_TRAIN_ROWS,
            expected_test=EXPECTED_TEST_ROWS,
        ),
        _ok(
            "query_counts",
            len(train_q) == EXPECTED_TRAIN_QUERIES and len(test_q) == EXPECTED_TEST_QUERIES,
            train_queries=len(train_q),
            test_queries=len(test_q),
        ),
        _ok(
            "query_overlap",
            not (train_q & test_q),
            overlap=len(train_q & test_q),
        ),
        _ok(
            "schema",
            train_cols == expected_cols and test_cols == expected_cols,
            feature_slots=list(FEATURE_SLOTS),
            meta_cols=sorted(META_COLS),
        ),
        _ok("keys", empty_key == 0, empty_query_or_item=empty_key),
        _ok(
            "duplicates",
            dup_train == 0 and dup_test == 0,
            duplicate_train_pairs=dup_train,
            duplicate_test_pairs=dup_test,
        ),
        _ok(
            "partitions",
            len(train_dates) == EXPECTED_TRAIN_PARTITIONS and len(test_dates) == 1,
            train_partitions=train_dates,
            test_partitions=test_dates,
        ),
        _ok("engage_vocab", not engage_bad, unknown_values=sorted(set(engage_bad))[:10]),
        _ok("request_features", request_ok, stats=request_features),
    ]
    report = {
        "checks": checks,
        "passed": all(item["ok"] for item in checks),
        "n_checks": len(checks),
        "n_passed": sum(1 for item in checks if item["ok"]),
    }
    return report
