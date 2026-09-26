"""Stage-2 features from stage-1 scores (differentiator #1a).

Group context: how a candidate ranks inside its S1 group.
Competition: how strongly the same pool record is claimed by another S1 (one-to-one signal).
Triangle support: true S2 and S3 matches of one entity also resemble each other, so a
candidate similar to the group's other high-scoring candidates gets support.
On train these are built from OUT-OF-FOLD p1 only.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from scipy.sparse.csgraph import connected_components

from . import config


def group_context(df):
    """Rank, gap to max, confident count and size inside each S1 group; competition across S1s."""
    g = df.groupby("s1_rid")["p1"]
    df["f2_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    df["f2_gap_max"] = (g.transform("max") - df["p1"]).astype(np.float32)
    df["f2_n_hi"] = (df["p1"] > 0.5).groupby(df["s1_rid"]).transform("sum").astype(np.float32)
    df["f2_gsize"] = g.transform("size").astype(np.float32)
    o = df[["pool_rid", "p1"]].sort_values(["pool_rid", "p1"], ascending=[True, False])
    r = o.groupby("pool_rid").cumcount()
    first = o.loc[r == 0].set_index("pool_rid")["p1"]
    second = o.loc[r == 1].set_index("pool_rid")["p1"]
    is_top = pd.Series(False, index=df.index)
    is_top.loc[o.index[(r == 0).to_numpy()]] = True
    f = first.reindex(df["pool_rid"]).to_numpy()
    s = second.reindex(df["pool_rid"]).fillna(0).to_numpy()
    df["f2_best_other"] = np.where(is_top.to_numpy(), s, f).astype(np.float32)
    df["f2_gap_other"] = (df["p1"] - df["f2_best_other"]).astype(np.float32)
    df["f2_n_claims"] = df.groupby("pool_rid")["p1"].transform("size").astype(np.float32)
    return df


def triangle_features(df, pooln, topm, pmin):
    """Support of each candidate from the group's top-m other candidates (name+address similarity)."""
    d = df[["s1_rid", "pool_rid", "p1"]]
    top = d.sort_values(["s1_rid", "p1"], ascending=[True, False])
    top = top[top.groupby("s1_rid").cumcount() < topm].rename(columns={"pool_rid": "j", "p1": "pj"})
    act = d.loc[d["p1"] >= pmin, ["s1_rid", "pool_rid"]].rename(columns={"pool_rid": "i"})
    pr = act.merge(top, on="s1_rid")
    pr = pr[pr["i"] != pr["j"]].reset_index(drop=True)
    cols = ["f2_support", "f2_n_supporters", "f2_max_sim_top", "f2_sim_top_other_src"]
    if len(pr) == 0:
        return df.assign(**{c: np.float32(0) for c in cols})
    i, j = pr["i"].to_numpy(), pr["j"].to_numpy()
    sim = (process.cpdist(pooln["name_norm"].take(i).tolist(), pooln["name_norm"].take(j).tolist(),
                          scorer=fuzz.token_set_ratio, workers=config.N_JOBS, dtype=np.float32)
           + process.cpdist(pooln["addr_norm"].take(i).tolist(), pooln["addr_norm"].take(j).tolist(),
                            scorer=fuzz.token_set_ratio, workers=config.N_JOBS, dtype=np.float32)) / 200.0
    s3 = pooln["is_s3"].to_numpy()
    pr["sim"] = sim
    pr["w"] = pr["pj"].to_numpy() * sim
    pr["sup"] = ((pr["pj"] > 0.5) & (pr["sim"] > 0.8)).astype(np.float32)
    g = pr.groupby(["s1_rid", "i"])
    agg = pd.DataFrame({"f2_support": g["w"].sum(), "f2_n_supporters": g["sup"].sum(),
                        "f2_max_sim_top": g["sim"].max()})
    other = pr[s3[i] != s3[j]].sort_values(["s1_rid", "i", "pj"], ascending=[True, True, False])
    first_other = other.drop_duplicates(["s1_rid", "i"]).set_index(["s1_rid", "i"])["sim"]
    agg["f2_sim_top_other_src"] = first_other.reindex(agg.index).fillna(-1.0)
    agg = agg.reset_index().rename(columns={"i": "pool_rid"})
    out = df.merge(agg, on=["s1_rid", "pool_rid"], how="left")
    out["f2_sim_top_other_src"] = out["f2_sim_top_other_src"].fillna(-1.0)
    out[cols] = out[cols].fillna(0.0).astype(np.float32)
    return out


CLUSTER_COLS = ["f2_comp_size", "f2_comp_n_sources", "f2_comp_mean_p", "f2_comp_max_other_p"]
CLUSTER_ID_COL = "_comp_id"  # not "f2_"-prefixed on purpose: an arbitrary id must never be a model feature


def cluster_features(df, pooln, sim_threshold=0.8):
    """Cross-source (S2/S3) cluster features (differentiator #1a extension, docs/IDEAS.md B.6).

    Within each S1 group, two candidates are joined into the same cluster if they come from
    different pool sources (S2/S3) and their name+address similarity exceeds `sim_threshold` (the
    same threshold `triangle_features`'s "sup" indicator uses) -- connected components of that
    graph give each candidate: cluster size, how many distinct sources the cluster spans, the
    cluster's mean p1, and the best OTHER member's p1 (0 for a singleton cluster). Built from
    `p1` only (OOF on train), never in-fold, same discipline as `triangle_features`. Also exposes
    the raw cluster id as `_comp_id` (Exp J2 plumbing, `decide.select_expected_f05`'s `comp_col`)
    -- deliberately not "f2_"-prefixed so `model.feature_cols` never treats it as a model input.
    """
    d = df[["s1_rid", "pool_rid", "p1"]].reset_index(drop=True)
    n = len(d)
    if n == 0:
        return df.assign(**{c: np.float32(0) for c in CLUSTER_COLS}, **{CLUSTER_ID_COL: np.int64(0)})
    d["_i"] = np.arange(n)
    is_s3 = pooln["is_s3"].to_numpy()
    m = d.merge(d, on="s1_rid", suffixes=("_a", "_b"))
    m = m[m["_i_a"] < m["_i_b"]]
    m = m[is_s3[m["pool_rid_a"].to_numpy()] != is_s3[m["pool_rid_b"].to_numpy()]]
    if len(m):
        sim = (process.cpdist(pooln["name_norm"].take(m["pool_rid_a"]).tolist(),
                              pooln["name_norm"].take(m["pool_rid_b"]).tolist(),
                              scorer=fuzz.token_set_ratio, workers=config.N_JOBS, dtype=np.float32)
               + process.cpdist(pooln["addr_norm"].take(m["pool_rid_a"]).tolist(),
                                pooln["addr_norm"].take(m["pool_rid_b"]).tolist(),
                                scorer=fuzz.token_set_ratio, workers=config.N_JOBS, dtype=np.float32)) / 200.0
        m = m[sim > sim_threshold]
    if len(m):
        graph = sp.coo_matrix((np.ones(len(m)), (m["_i_a"], m["_i_b"])), shape=(n, n))
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(n)
    d["_comp"], d["_is_s3"] = labels, is_s3[d["pool_rid"].to_numpy()]
    g = d.groupby("_comp")
    d["f2_comp_size"] = g["p1"].transform("size").astype(np.float32)
    d["f2_comp_n_sources"] = g["_is_s3"].transform("nunique").astype(np.float32)
    d["f2_comp_mean_p"] = g["p1"].transform("mean").astype(np.float32)
    o = d[["_comp", "p1"]].sort_values(["_comp", "p1"], ascending=[True, False])
    r = o.groupby("_comp").cumcount()
    first = o.loc[r == 0].set_index("_comp")["p1"]
    second = o.loc[r == 1].set_index("_comp")["p1"]
    is_top = pd.Series(False, index=d.index)
    is_top.loc[o.index[(r == 0).to_numpy()]] = True
    f = first.reindex(d["_comp"]).to_numpy()
    s = second.reindex(d["_comp"]).fillna(0).to_numpy()
    d["f2_comp_max_other_p"] = np.where(is_top.to_numpy(), s, f).astype(np.float32)
    d[CLUSTER_ID_COL] = d["_comp"].astype(np.int64)
    return pd.concat([df.reset_index(drop=True), d[CLUSTER_COLS + [CLUSTER_ID_COL]]], axis=1)


def add_stage2_features(df, pooln, topm=None, pmin=None):
    """All stage-2 features; preserves row order of `df` (which must contain p1)."""
    topm = topm or config.TRIANGLE_TOPM
    pmin = config.TRIANGLE_PMIN if pmin is None else pmin
    df = group_context(df.reset_index(drop=True))
    df = triangle_features(df, pooln, topm, pmin)
    return cluster_features(df, pooln)
