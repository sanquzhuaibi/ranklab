"""End-to-end v2 ranking run."""
from __future__ import annotations

import json
import sys

import joblib

from . import config
from .data_check import run_data_check
from .evaluate import evaluate
from .features import build_matrix, used_feature_names
from .labels import make_labels
from .train import predict_scores, train_ranker


def run(out_name: str = "run_metrics.json") -> dict:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    check = run_data_check()
    check_path = config.OUT_DIR / "data_check.json"
    check_path.write_text(json.dumps(check, indent=2), encoding="utf-8")
    if not check["passed"]:
        raise SystemExit(f"data_check failed, see {check_path}")
    q_tr, _, e_tr, x_tr = build_matrix(config.TRAIN_CSV)
    q_te, _, e_te, x_te = build_matrix(config.TEST_CSV)
    y_tr, w_tr, _ = make_labels(q_tr, e_tr)
    y_te, _, gw_te = make_labels(q_te, e_te)
    model = train_ranker(x_tr, y_tr, q_tr, w_tr)
    pred = predict_scores(model, x_te)
    metrics = evaluate(y_te, pred, q_te, gw_te)
    metrics["feature_dim"] = len(used_feature_names())
    metrics["feature_names"] = used_feature_names()
    metrics["train_rows"] = len(y_tr)
    metrics["label_mode"] = config.LABEL_MODE
    metrics["weight_mode"] = config.WEIGHT_MODE
    metrics["max_depth"] = config.MAX_DEPTH
    metrics["data_check"] = "passed"
    joblib.dump(model, config.OUT_DIR / "model.joblib")
    out = config.OUT_DIR / out_name
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "data_check": "passed",
        "primary": metrics["primary"],
        "roc_auc": metrics["roc_auc"],
        "gauc": metrics["gauc"],
        "recall@10": metrics["recall@10"],
        "ndcg": metrics["ndcg"],
        "feature_dim": metrics["feature_dim"],
        "n_queries": metrics["n_queries"],
    }, indent=2))
    print(f"wrote {out}")
    return metrics


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "run_metrics.json"
    run(name)
