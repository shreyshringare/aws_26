"""Central configuration: paths, seeds, blocking and decision hyper-parameters."""
import os
from pathlib import Path

SEED = 42

PKG_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("ER_CACHE", PKG_DIR / "cache"))
N_JOBS = int(os.environ.get("ER_JOBS", max(1, (os.cpu_count() or 2) - 2)))

# --- blocking ---------------------------------------------------------------
# Tokens whose document frequency in the S2/S3 pool (per country) exceeds this
# are dropped from the blocking index: they carry little signal and explode cost.
BLOCK_MAX_DF = 3000
# Candidates kept per S1 entity after blocking-score ranking.
BLOCK_TOPK = 40
# Rows of S1 processed per sparse-matmul chunk.
BLOCK_CHUNK = 20000
# Relative weights of key families inside the blocking vector.
BLOCK_W = {"n": 1.0, "nb": 1.0, "a": 0.6, "ab": 1.0, "p": 0.8, "nm": 0.7}

# --- model ------------------------------------------------------------------
N_FOLDS = 5
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_estimators=600,
    verbose=-1,
    seed=SEED,
)

# --- decision (defaults, overwritten by tuning) -------------------------------
T_MATCH = 0.5
T_EMPTY = 0.5
