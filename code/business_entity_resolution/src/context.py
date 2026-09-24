"""Stage-2 features from stage-1 scores (differentiator #1a).

Group context: how a candidate ranks inside its S1 group.
Competition: how strongly the same pool record is claimed by another S1 (one-to-one signal).
Triangle support: true S2 and S3 matches of one entity also resemble each other, so a
candidate similar to the group's other high-scoring candidates gets support.
On train these are built from OUT-OF-FOLD p1 only.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

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


def add_stage2_features(df, pooln, topm=None, pmin=None):
    """All stage-2 features; preserves row order of `df` (which must contain p1)."""
    topm = topm or config.TRIANGLE_TOPM
    pmin = config.TRIANGLE_PMIN if pmin is None else pmin
    df = group_context(df.reset_index(drop=True))
    return triangle_features(df, pooln, topm, pmin)
