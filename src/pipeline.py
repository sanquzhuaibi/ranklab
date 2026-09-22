"""End-to-end v2 ranking run."""
from __future__ import annotations

import json
import sys

import joblib

from . import config
from .evaluate import evaluate
from .features import build_matrix, used_feature_names
from .labels import make_labels
from .train import predict_scores, train_ranker


def run(out_name: str = "run_metrics.json") -> dict:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    q_tr, _, e_tr, x_tr = build_matrix(config.TRAIN_CSV)
    q_te, _, e_te, x_te = build_matrix(config.TEST_CSV)
    y_tr, w_tr, _ = make_labels(q_tr, e_tr)
    y_te, _, _ = make_labels(q_te, e_te)
    model = train_ranker(x_tr, y_tr, q_tr, w_tr)
    pred = predict_scores(model, x_te)
    metrics = evaluate(y_te, pred, q_te, e_te)
    metrics["feature_dim"] = len(used_feature_names())
    metrics["feature_names"] = used_feature_names()
    metrics["train_rows"] = len(y_tr)
    metrics["label_mode"] = config.LABEL_MODE
    metrics["weight_mode"] = config.WEIGHT_MODE
    metrics["max_depth"] = config.MAX_DEPTH
    joblib.dump(model, config.OUT_DIR / "model.joblib")
    out = config.OUT_DIR / out_name
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("primary", "roc_auc", "gauc", "recall@10", "ndcg", "feature_dim", "n_queries", "group_size_min", "group_size_max")}, indent=2))
    print(f"wrote {out}")
    return metrics


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "run_metrics.json"
    run(name)
