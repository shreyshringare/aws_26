"""Decision layer (differentiator #1b): calibration, global one-to-one, expected-F0.5 subset selection.

Per S1 entity the metric is F0.5 = 1.25*TP / (k + 0.25*G). Instead of a fixed threshold we pick,
for every entity, the k (possibly 0) that maximises the expected F0.5 under the calibrated
probabilities. A thresholded decoder is kept as the baseline; tuning keeps whichever scores higher.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config


def fit_calibrator(p, y):
    """Isotonic calibration of OOF scores against labels."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(p, np.float64), np.asarray(y, np.float64))
    return iso


def temper(p, tau):
    """Sharpen (tau < 1) or soften (tau > 1) probabilities in logit space."""
    if tau == 1.0:
        return p
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / tau))


def correlated_hits(P, C, rng, n_draws):
    """Two-level correlated Bernoulli draw (Exp J2, docs/RESEARCH_JOINT_S2S3.md Option B).

    P, C are (batch, kmax): P is each slot's calibrated probability, C is its cluster id
    (from `context.cluster_features`; negative = no cluster, treated as its own singleton).
    Slots sharing a cluster id draw a shared latent `z_c ~ Bernoulli(max P in the cluster)`
    once per draw; each slot then independently draws conditioned on `z_c` at a rate
    `a_i = min(1, P_i / q_c)` chosen so that `E[hit_i] = q_c * a_i = P_i` exactly -- the
    marginal probability of any single slot is unchanged, only the correlation between
    cluster-mates is added. A slot with no cluster-mates (including C < 0) reduces to
    `q_c = P_i`, `a_i = 1`, i.e. plain `Bernoulli(P_i)`.
    Returns a (batch, n_draws, kmax) bool array.
    """
    b, kmax = P.shape
    idx = np.arange(kmax)
    same = (C[:, :, None] == C[:, None, :]) & (C[:, :, None] >= 0)
    q = np.where(same, P[:, None, :], 0).max(axis=2)
    q = np.maximum(q, P)
    a = np.clip(np.divide(P, q, out=np.ones_like(P), where=q > 0), 0.0, 1.0)
    leader_raw = np.where(same, idx[None, None, :], kmax).min(axis=2)
    leader = np.where(leader_raw < kmax, leader_raw, idx[None, :])
    u = rng.random((b, n_draws, kmax), dtype=np.float32)
    u_shared = np.take_along_axis(u, np.broadcast_to(leader[:, None, :], u.shape), axis=2)
    z = u_shared < q[:, None, :]
    v = rng.random((b, n_draws, kmax), dtype=np.float32)
    return z & (v < a[:, None, :])


def one_to_one(df):
    """Keep each pool record only in the S1 group where its probability is highest."""
    if len(df) == 0:
        return df
    return df.loc[df.groupby("pool_rid")["p"].idxmax()]


def select_threshold(df, t_match, t_empty):
    """Baseline: keep p >= t_match, but predict empty when the group's best p < t_empty."""
    best = df.groupby("s1_rid")["p"].transform("max")
    return df[(df["p"] >= t_match) & (best >= t_empty)]


def select_expected_f05(df, extra, p_min=None, kmax=None, n_draws=None, seed=None, batch=50_000,
                        comp_col=None):
    """Per S1 group choose top-k (k >= 0) maximising Monte-Carlo expected F0.5.

    `extra` = expected true matches outside the scored candidates (missed by blocking).
    `comp_col` (Exp J2, docs/RESEARCH_JOINT_S2S3.md Option B): optional name of a cluster-id
    column in `df` (e.g. from `context.cluster_features`). When given, candidates are drawn
    with `correlated_hits` instead of independently, so cluster-mates move together in the
    simulation. Omitting it (default) reproduces the pre-Exp-J2 independent-draw behavior
    exactly, unchanged.
    """
    p_min = config.P_MIN if p_min is None else p_min
    kmax = kmax or config.EF_KMAX
    n_draws = n_draws or config.EF_DRAWS
    seed = config.SEED if seed is None else seed
    tot = df.groupby("s1_rid")["p"].sum()
    d = df[df["p"] >= p_min].sort_values(["s1_rid", "p"], ascending=[True, False])
    pos = d.groupby("s1_rid").cumcount().to_numpy()
    d, pos = d[pos < kmax], pos[pos < kmax]
    if len(d) == 0:
        return d[["s1_rid", "pool_rid", "p"]]
    gid, groups = pd.factorize(d["s1_rid"], sort=False)
    P = np.zeros((len(groups), kmax), np.float32)
    P[gid, pos] = d["p"].to_numpy(np.float32)
    C = None
    if comp_col:
        C = np.full((len(groups), kmax), -1, np.int64)
        C[gid, pos] = d[comp_col].to_numpy(np.int64)
    ext = np.maximum(tot.reindex(groups).to_numpy(np.float32) - P.sum(1), 0) + np.float32(extra)
    rng = np.random.default_rng(seed)
    kk = np.arange(1, kmax + 1, dtype=np.float32)
    ks = np.zeros(len(groups), np.int16)
    for s in range(0, len(groups), batch):
        pb, eb = P[s:s + batch], ext[s:s + batch]
        if C is not None:
            hits = correlated_hits(pb, C[s:s + batch], rng, n_draws)
        else:
            hits = rng.random((len(pb), n_draws, kmax), dtype=np.float32) < pb[:, None, :]
        tp = np.cumsum(hits, axis=2, dtype=np.float32)
        g = tp[:, :, -1] + eb[:, None]
        ef = (1.25 * tp / (kk + 0.25 * g[:, :, None])).mean(1)
        ef[pb == 0] = -1.0
        e0 = np.prod(1 - pb, axis=1) * np.exp(-eb)
        ks[s:s + batch] = np.where(e0 >= ef.max(1), 0, ef.argmax(1) + 1)
    return d[pos < ks[gid]][["s1_rid", "pool_rid", "p"]]


def apply_decoder(df, dec, calibrator, extra, comp_col=None):
    """Calibrate -> temper -> optional global one-to-one -> chosen subset selection.

    `dec.get("one_to_one", True)`: hard one-to-one (each pool record kept only in its single
    best-scoring S1 group) is applied unless a decoder explicitly opts out with
    `one_to_one: False`. Defaulting to True keeps behavior unchanged for any decoder dict that
    predates this flag (see IDEAS.md B.8: this was previously unconditional).

    `comp_col` (Exp J2): name of a cluster-id column in `df` (see `select_expected_f05`). Only
    used when the decoder explicitly requests it via `dec.get("joint")`; otherwise ignored, so
    passing `comp_col` alone can't change behavior -- both `joint: True` and a real `comp_col`
    are required.
    """
    cols = ["s1_rid", "pool_rid", "p"] + ([comp_col] if comp_col else [])
    d = df[cols].copy()
    if calibrator is not None:
        d["p"] = calibrator.predict(d["p"].to_numpy(np.float64)).astype(np.float32)
    d["p"] = temper(d["p"].to_numpy(), dec.get("tau", 1.0))
    if dec.get("one_to_one", True):
        d = one_to_one(d)
    if dec["method"] == "threshold":
        return select_threshold(d, dec["t_match"], dec["t_empty"])
    use_comp = comp_col if (comp_col and dec.get("joint")) else None
    return select_expected_f05(d, extra * dec.get("lam", 1.0), comp_col=use_comp)


def tune(df, calibrator, extra, scorer, comp_col=None):
    """Grid-search both decoders on OOF scores with the exact metric; return (best params, all results).

    Also searches `one_to_one` (True vs False) as a grid dimension over every point, so hard
    global one-to-one assignment (an assumption measured only on train, IDEAS.md B.8) is kept
    only when it actually beats skipping it, on OOF, same as tau/lam/t_match/t_empty.

    `comp_col` (Exp J2, docs/RESEARCH_JOINT_S2S3.md): when given, every ef05 grid point is also
    searched with `joint` True vs False (threshold points are untouched -- `joint` only means
    anything for the ef05 decoder). Omitting `comp_col` (default) reproduces the pre-Exp-J2 grid
    exactly, same size as before.
    """
    grid = [{"method": "threshold", "t_match": tm, "t_empty": te, "tau": 1.0}
            for tm in config.GRID_T_MATCH for te in config.GRID_T_EMPTY]
    grid += [{"method": "ef05", "tau": tau, "lam": lam} for tau in config.GRID_TAU for lam in config.GRID_LAM]
    results = []
    for o2o in (True, False):
        for base in grid:
            variants = [dict(base, one_to_one=o2o)]
            if comp_col and base["method"] == "ef05":
                variants = [dict(v, joint=j) for v in variants for j in (True, False)]
            for dec in variants:
                s = scorer(apply_decoder(df, dec, calibrator, extra, comp_col=comp_col))
                results.append((s, dec))
                print(f"  {dec} -> {s:.5f}", flush=True)
    best = max(results, key=lambda r: r[0])
    return dict(best[1], oof_f05=round(best[0], 5)), results


def tune_segments(df, calibrator, extra, scorer, segment_col, comp_col=None):
    """Tune one decoder per distinct value of `segment_col` (docs/NEXT_STEPS.md item 15).

    `segment_col` must be a country-agnostic column already in `df` (e.g. `is_s3`,
    `f_addr_empty_a`/`_b`, `f_script_a`/`_b`) -- never country, per CLAUDE.md's
    country-is-a-blocking-key-only rule. `scorer` is shared across segments (it should itself
    already restrict scoring to the segment's rows, e.g. via a mask, the same convention
    `run_pipeline.Scorer` already uses). Returns {segment value: (dec, grid results)}.

    Caveat: global one-to-one competition (`decide.one_to_one`) is applied independently within
    each segment's own `tune`/`apply_decoder` call, not across segments -- correct when
    `segment_col` is a property of the pool record itself (e.g. `is_s3`, which source a candidate
    is from), since one_to_one's competition is over `pool_rid` and a pool record's source never
    changes across S1 groups. It is NOT correct for an S1-side segment (e.g. an S1-level
    address-empty flag) where the same pool record could appear in two different S1 groups whose
    segment differs -- don't use an S1-side column here without checking that case specifically.
    """
    return {seg: tune(g, calibrator, extra, scorer, comp_col=comp_col) for seg, g in df.groupby(segment_col)}


def apply_segmented_decoder(df, decoders, segment_col, calibrator, extra, comp_col=None):
    """Apply each segment's own decoder (from `tune_segments`) to its rows; concatenate results."""
    parts = [apply_decoder(g, decoders[seg], calibrator, extra, comp_col=comp_col)
             for seg, g in df.groupby(segment_col)]
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0][["s1_rid", "pool_rid", "p"]]
