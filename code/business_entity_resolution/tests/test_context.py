"""Tests for stage-2 context and triangle features."""
import numpy as np
import pandas as pd

from src.context import add_stage2_features
from src.normalize import normalize_frame


def test_stage2_features():
    pool = normalize_frame(pd.DataFrame([
        ["S2-1", "Blue Ocean Ltd", "5 Sea Rd, Goa", "India"],
        ["S3-1", "Blue Ocean", "5 Sea Road, Goa", "India"],
        ["S3-2", "Red Rock", "9 Hill Rd, Pune", "India"],
    ], columns=["entity_id", "business_name", "business_address", "country"]))
    feats = pd.DataFrame({"s1_rid": [0, 0, 0, 1], "pool_rid": [0, 1, 2, 2],
                          "p1": np.float32([0.9, 0.6, 0.3, 0.8])})
    out = add_stage2_features(feats, pool, topm=5, pmin=0.01)
    assert list(out["pool_rid"]) == [0, 1, 2, 2]
    r = out.set_index(["s1_rid", "pool_rid"])
    assert r.loc[(0, 0), "f2_rank"] == 1 and r.loc[(0, 2), "f2_rank"] == 3
    assert r.loc[(0, 2), "f2_best_other"] == np.float32(0.8)
    assert r.loc[(0, 1), "f2_support"] > r.loc[(0, 2), "f2_support"]
    assert r.loc[(0, 1), "f2_sim_top_other_src"] > 0.8
