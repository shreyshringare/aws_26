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


def test_cluster_features_only_considers_top_m_candidates_per_group():
    """Regression guard: cluster_features must cap pair generation to a group's top-m candidates
    by p1 (like triangle_features already does), never a full O(group_size^2) self-join -- at full
    test scale (~1.7M S1 entities x ~40 candidates each) an uncapped self-join is a real
    memory/runtime blowup (sum of group_size^2 across all groups, not group_size*topm).
    Two genuinely near-duplicate cross-source candidates ranked OUTSIDE the top-m (by p1) must NOT
    be clustered together, even though an uncapped comparison would cluster them."""
    ents = [f"S2-{i}" if i % 2 == 0 else f"S3-{i}" for i in range(8)] + ["S2-x", "S3-x"]
    names = [f"Unrelated {i}" for i in range(8)] + ["Blue Ocean Ltd", "Blue Ocean"]
    addrs = [f"{i} Nowhere St" for i in range(8)] + ["5 Sea Rd, Goa", "5 Sea Road, Goa"]
    pool = normalize_frame(pd.DataFrame({"entity_id": ents, "business_name": names,
                                         "business_address": addrs, "country": ["India"] * len(ents)}))
    # top-5 by p1 = indices 0-4; the near-duplicate pair (indices 8, 9) is ranked last (8th, 9th)
    p1 = np.float32([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.2, 0.15])
    feats = pd.DataFrame({"s1_rid": [0] * 10, "pool_rid": list(range(10)), "p1": p1})
    out = cluster_features(feats, pool)
    r = out.set_index("pool_rid")
    assert r.loc[8, "f2_comp_size"] == 1
    assert r.loc[9, "f2_comp_size"] == 1
