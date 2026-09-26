"""Tests for stage-2 context and triangle features."""
import numpy as np
import pandas as pd

from src.context import add_stage2_features, cluster_features
from src.normalize import normalize_frame


def _pool():
    return normalize_frame(pd.DataFrame([
        ["S2-1", "Blue Ocean Ltd", "5 Sea Rd, Goa", "India"],
        ["S3-1", "Blue Ocean", "5 Sea Road, Goa", "India"],
        ["S3-2", "Red Rock", "9 Hill Rd, Pune", "India"],
    ], columns=["entity_id", "business_name", "business_address", "country"]))


def test_stage2_features():
    pool = _pool()
    feats = pd.DataFrame({"s1_rid": [0, 0, 0, 1], "pool_rid": [0, 1, 2, 2],
                          "p1": np.float32([0.9, 0.6, 0.3, 0.8])})
    out = add_stage2_features(feats, pool, topm=5, pmin=0.01)
    assert list(out["pool_rid"]) == [0, 1, 2, 2]
    r = out.set_index(["s1_rid", "pool_rid"])
    assert r.loc[(0, 0), "f2_rank"] == 1 and r.loc[(0, 2), "f2_rank"] == 3
    assert r.loc[(0, 2), "f2_best_other"] == np.float32(0.8)
    assert r.loc[(0, 1), "f2_support"] > r.loc[(0, 2), "f2_support"]
    assert r.loc[(0, 1), "f2_sim_top_other_src"] > 0.8
    assert "f2_comp_size" in out.columns  # cluster_features wired into add_stage2_features


def test_cluster_features_cross_source_component():
    """Two cross-source (S2/S3) near-duplicate candidates in the same S1 group form one cluster;
    an unrelated third candidate, and a candidate in a different S1 group, are singleton clusters."""
    pool = _pool()
    feats = pd.DataFrame({"s1_rid": [0, 0, 0, 1], "pool_rid": [0, 1, 2, 2],
                          "p1": np.float32([0.9, 0.6, 0.3, 0.8])})
    out = cluster_features(feats, pool)
    r = out.set_index(["s1_rid", "pool_rid"])
    assert r.loc[(0, 0), "f2_comp_size"] == 2
    assert r.loc[(0, 1), "f2_comp_size"] == 2
    assert r.loc[(0, 0), "f2_comp_n_sources"] == 2
    assert r.loc[(0, 0), "f2_comp_mean_p"] == np.float32(0.75)
    assert r.loc[(0, 0), "f2_comp_max_other_p"] == np.float32(0.6)
    assert r.loc[(0, 1), "f2_comp_max_other_p"] == np.float32(0.9)
    assert r.loc[(0, 2), "f2_comp_size"] == 1
    assert r.loc[(0, 2), "f2_comp_n_sources"] == 1
    assert r.loc[(0, 2), "f2_comp_max_other_p"] == 0.0
    assert r.loc[(1, 2), "f2_comp_size"] == 1
    # _comp_id (Exp J2 plumbing, docs/RESEARCH_JOINT_S2S3.md Option B): exposed as a raw grouping
    # id -- not an "f2_"-prefixed feature, so model.feature_cols() never picks it up as model input.
    assert r.loc[(0, 0), "_comp_id"] == r.loc[(0, 1), "_comp_id"]
    assert r.loc[(0, 2), "_comp_id"] != r.loc[(0, 0), "_comp_id"]
    assert r.loc[(1, 2), "_comp_id"] != r.loc[(0, 2), "_comp_id"]
