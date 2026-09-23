from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SCHEMA_JSON = DATA_DIR / "feature_schema.json"

PRIMARY_METRIC = "gauc"
NDCG_KS = (1, 3)

MAX_DEPTH = 6
LEARNING_RATE = 0.1
MAX_ITER = 120
RANDOM_SEED = 20260903
