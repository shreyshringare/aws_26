"""Stage-1 pair features: string similarities, set overlaps, record statistics and noise features.

Country is deliberately absent: every feature is computed identically for any label.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from . import config
from .noise import NOISE_FEATS, noise_row

ROW_COLS = ["name_core", "legal", "legal_first", "nums", "postcode", "addr_alpha",
            "name_norm", "name_raw", "addr_norm", "addr_raw"]
FUZZ = [
    ("name_norm", "ratio", fuzz.ratio), ("name_norm", "tset", fuzz.token_set_ratio),
    ("name_norm", "tsort", fuzz.token_sort_ratio), ("name_norm", "partial", fuzz.partial_ratio),
    ("name_norm", "jw", JaroWinkler.normalized_similarity),
    ("name_core", "ratio", fuzz.ratio), ("name_core", "tset", fuzz.token_set_ratio),
    ("name_core", "jw", JaroWinkler.normalized_similarity),
    ("addr_norm", "ratio", fuzz.ratio), ("addr_norm", "tset", fuzz.token_set_ratio),
    ("addr_norm", "partial", fuzz.partial_ratio), ("addr_alpha", "tset", fuzz.token_set_ratio),
]
PY_FEATS = ["f_name_jac", "f_addr_jac", "f_nums_jac", "f_num1_eq", "f_pc_eq", "f_pc_missing",
            "f_legal_code", "f_core_len_diff"] + NOISE_FEATS


def add_frequency_columns(s1n, pooln):
    """Add log pool-frequency of name_core to both frames (how common a business name is)."""
    vc = pooln["name_core"].value_counts()
    pooln["name_freq"] = np.log1p(pooln["name_core"].map(vc).to_numpy(np.float32))
    s1n["name_freq"] = np.log1p(s1n["name_core"].map(vc).fillna(0).to_numpy(np.float32))


def _jac(a, b):
    """Jaccard similarity of two sets (0 when both empty)."""
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _legal_code(a, b):
    """0 both empty, 1 equal, 2 exactly one empty, 3 both present but different."""
    if not a and not b:
        return 0
    if a == b:
        return 1
    return 2 if (not a or not b) else 3


def _py_chunk(A, B, generic):
    """Set/number/legal features plus noise features for a chunk of row-tuple pairs."""
    out = np.zeros((len(A), len(PY_FEATS)), np.float32)
    for i, (a, b) in enumerate(zip(A, B)):
        ca, cb = set(a[0].split()), set(b[0].split())
        na, nb = a[3].split(), b[3].split()
        out[i, 0] = _jac(ca, cb)
        out[i, 1] = _jac(set(a[5].split()), set(b[5].split()))
        out[i, 2] = _jac(set(na), set(nb))
        out[i, 3] = float(na[0] == nb[0]) if (na and nb) else -1.0
        out[i, 4] = float(bool(a[4]) and a[4] == b[4])
        out[i, 5] = float(not a[4] or not b[4])
        out[i, 6] = _legal_code(a[1], b[1])
        out[i, 7] = abs(len(ca) - len(cb))
        out[i, 8:] = noise_row(a, b, generic)
    return out


def _take(frame, col, idx):
    """Positional gather of one column as a Python list."""
    return frame[col].take(idx).tolist()


def _features_chunk(c, s1n, pooln, generic, n_jobs):
    """All stage-1 features for one chunk of candidate pairs."""
    ia, ib = c["s1_rid"].to_numpy(), c["pool_rid"].to_numpy()
    out = {"s1_rid": ia, "pool_rid": ib}
    for col, name, scorer in FUZZ:
        out[f"f_{col}_{name}"] = process.cpdist(_take(s1n, col, ia), _take(pooln, col, ib), scorer=scorer,
                                                workers=n_jobs, dtype=np.float32)
    A = list(zip(*[_take(s1n, k, ia) for k in ROW_COLS]))
    B = list(zip(*[_take(pooln, k, ib) for k in ROW_COLS]))
    step = max(1, len(A) // (n_jobs * 4) + 1)
    parts = Parallel(n_jobs=n_jobs)(delayed(_py_chunk)(A[i:i + step], B[i:i + step], generic)
                                    for i in range(0, len(A), step))
    py = np.vstack(parts) if parts else np.zeros((0, len(PY_FEATS)), np.float32)
    for j, name in enumerate(PY_FEATS):
        out[name] = py[:, j]
    out["f_block_score"] = c["block_score"].to_numpy(np.float32)
    out["f_block_rank"] = c["block_rank"].to_numpy(np.float32)
    for col in c.columns[c.columns.str.startswith("score_")]:  # per-channel blocking scores
        out[f"f_{col}"] = c[col].to_numpy(np.float32)
    out["f_is_s3"] = pooln["is_s3"].to_numpy()[ib].astype(np.float32)
    for side, frame, idx in (("a", s1n, ia), ("b", pooln, ib)):
        out[f"f_freq_{side}"] = frame["name_freq"].to_numpy()[idx].astype(np.float32)
        out[f"f_script_{side}"] = frame["script_flag"].to_numpy()[idx].astype(np.float32)
        out[f"f_addr_empty_{side}"] = (frame["addr_norm"].take(idx) == "").to_numpy().astype(np.float32)
        out[f"f_land_{side}"] = (frame["landmark"].take(idx) != "").to_numpy().astype(np.float32)
        out[f"f_ntok_{side}"] = frame["name_core"].take(idx).str.count(" ").to_numpy().astype(np.float32) + 1
    return pd.DataFrame(out)


def build_features(cands, s1n, pooln, generic, n_jobs=None, chunk=2_000_000):
    """Stage-1 features for every candidate pair, in candidate order (float32 f_* columns)."""
    n_jobs = n_jobs or config.N_JOBS
    parts = [_features_chunk(cands.iloc[i:i + chunk], s1n, pooln, generic, n_jobs)
             for i in range(0, len(cands), chunk)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
