from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SCHEMA_JSON = DATA_DIR / "feature_schema.json"

# v2 experiment notes (2026-09-09):
# 1) label / group weight = max(engage_score) so deep interactions count more
# 2) 16-d feature dump from warehouse, including derived ratios
# 3) offline GAUC switched to max(engage)-weighted; primary report is Recall@10
LABEL_MODE = "binary_ge1"
WEIGHT_MODE = "group_max_engage"
PRIMARY_METRIC = "recall@10"
RECALL_K = 10
NDCG_KS = (1, 3, 5, 10)

# sklearn HGB stands in for the pairwise ranker we use online
MAX_DEPTH = 6
LEARNING_RATE = 0.1
MAX_ITER = 120
RANDOM_SEED = 20260903
