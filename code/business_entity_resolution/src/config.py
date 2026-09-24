"""Central configuration: paths, seeds, sampling, blocking, model and decoding hyper-parameters."""
import os
from pathlib import Path

SEED = 42

PKG_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_DIR.parents[1]  # student_resource/
CACHE_DIR = Path(os.environ.get("ER_CACHE", PKG_DIR / "cache"))
# Bump when normalize/blocking/features change so stale parquet caches are ignored.
CACHE_TAG = os.environ.get("ER_CACHE_TAG", "v1")
N_JOBS = int(os.environ.get("ER_JOBS", max(1, (os.cpu_count() or 2) - 2)))

# --- sampling -----------------------------------------------------------------
DEV_FRAC = 0.10    # share of train S1 entities used by --eval-only dev runs
TRAIN_FRAC = 0.15  # share of train S1 entities used to fit models in full runs

# --- blocking -----------------------------------------------------------------
BLOCK_MAX_DF = 2000   # drop keys whose pool document frequency exceeds this
BLOCK_TOPK = 40       # candidates kept per S1 entity
BLOCK_CHUNK = 4000    # S1 rows per sparse-matmul chunk
BLOCK_W = {"n": 1.0, "nb": 1.0, "nm": 0.5, "a": 0.6, "ab": 1.5, "p": 1.0}

# --- model --------------------------------------------------------------------
N_FOLDS = 5
EARLY_STOP = 50
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.08,
    num_leaves=127,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_estimators=2000,
    verbose=-1,
    random_state=SEED,
    n_jobs=N_JOBS,
)

# --- stage 2 ------------------------------------------------------------------
TRIANGLE_TOPM = 5     # supporters considered per S1 group
TRIANGLE_PMIN = 0.01  # candidates below this stage-1 score get no triangle features

# --- decoding -----------------------------------------------------------------
P_MIN = 0.01
EF_DRAWS = 128
EF_KMAX = 12
T_MATCH = 0.5
T_EMPTY = 0.5
