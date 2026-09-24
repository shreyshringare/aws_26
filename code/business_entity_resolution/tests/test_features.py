"""Tests for stage-1 pair features."""
import numpy as np
import pandas as pd

from src.features import add_frequency_columns, build_features
from src.normalize import normalize_frame


def _f(rows):
    return normalize_frame(pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]))


def test_build_features_shapes_and_values():
    s1 = _f([["S1-1", "Valley Holdings Inc", "719 Mountain Road, Addison, VT", "US"]])
    pool = _f([["S3-1", "Vlley Holdings", "719 Mountain Rd, Addison, Vermont", "US"],
               ["S2-2", "Other Thing LLC", "1 Main St, Austin, TX", "US"]])
    add_frequency_columns(s1, pool)
    cands = pd.DataFrame({"s1_rid": [0, 0], "pool_rid": [0, 1],
                          "block_score": np.float32([9, 1]), "block_rank": np.int16([0, 1]),
                          "score_name": np.float32([6, 0]), "score_addr": np.float32([3, 1])})
    f = build_features(cands, s1, pool, set(), n_jobs=1)
    assert list(f["pool_rid"]) == [0, 1]
    assert list(f["f_score_name"]) == [6, 0] and list(f["f_score_addr"]) == [3, 1]
    assert f.loc[0, "f_addr_jac"] == 1.0 and f.loc[0, "f_num1_eq"] == 1.0
    assert f.loc[0, "f_nz_edit1_tokens"] == 1 and f.loc[0, "f_is_s3"] == 1
    assert f.loc[0, "f_name_core_tset"] > f.loc[1, "f_name_core_tset"]
    assert not any("country" in c for c in f.columns)
    assert all(f[c].dtype == np.float32 for c in f.columns if c.startswith("f_"))
