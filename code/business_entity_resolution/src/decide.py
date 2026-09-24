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


def one_to_one(df):
    """Keep each pool record only in the S1 group where its probability is highest."""
    if len(df) == 0:
        return df
    return df.loc[df.groupby("pool_rid")["p"].idxmax()]


def select_threshold(df, t_match, t_empty):
    """Baseline: keep p >= t_match, but predict empty when the group's best p < t_empty."""
    best = df.groupby("s1_rid")["p"].transform("max")
    return df[(df["p"] >= t_match) & (best >= t_empty)]


def select_expected_f05(df, extra, p_min=None, kmax=None, n_draws=None, seed=None, batch=50_000):
    """Per S1 group choose top-k (k >= 0) maximising Monte-Carlo expected F0.5.

    `extra` = expected true matches outside the scored candidates (missed by blocking).
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
    ext = np.maximum(tot.reindex(groups).to_numpy(np.float32) - P.sum(1), 0) + np.float32(extra)
    rng = np.random.default_rng(seed)
    kk = np.arange(1, kmax + 1, dtype=np.float32)
    ks = np.zeros(len(groups), np.int16)
    for s in range(0, len(groups), batch):
        pb, eb = P[s:s + batch], ext[s:s + batch]
        hits = rng.random((len(pb), n_draws, kmax), dtype=np.float32) < pb[:, None, :]
        tp = np.cumsum(hits, axis=2, dtype=np.float32)
        g = tp[:, :, -1] + eb[:, None]
        ef = (1.25 * tp / (kk + 0.25 * g[:, :, None])).mean(1)
        ef[pb == 0] = -1.0
        e0 = np.prod(1 - pb, axis=1) * np.exp(-eb)
        ks[s:s + batch] = np.where(e0 >= ef.max(1), 0, ef.argmax(1) + 1)
    return d[pos < ks[gid]][["s1_rid", "pool_rid", "p"]]


def apply_decoder(df, dec, calibrator, extra):
    """Calibrate -> temper -> global one-to-one -> chosen subset selection."""
    d = df[["s1_rid", "pool_rid", "p"]].copy()
    if calibrator is not None:
        d["p"] = calibrator.predict(d["p"].to_numpy(np.float64)).astype(np.float32)
    d["p"] = temper(d["p"].to_numpy(), dec.get("tau", 1.0))
    d = one_to_one(d)
    if dec["method"] == "threshold":
        return select_threshold(d, dec["t_match"], dec["t_empty"])
    return select_expected_f05(d, extra * dec.get("lam", 1.0))


def tune(df, calibrator, extra, scorer):
    """Grid-search both decoders on OOF scores with the exact metric; return (best params, all results)."""
    grid = []
    for tm in (0.3, 0.4, 0.5, 0.6, 0.7):
        for te in (0.3, 0.4, 0.5, 0.6, 0.7):
            grid.append({"method": "threshold", "t_match": tm, "t_empty": te, "tau": 1.0})
    for tau in (0.8, 1.0, 1.25):
        for lam in (0.5, 1.0, 2.0):
            grid.append({"method": "ef05", "tau": tau, "lam": lam})
    results = []
    for dec in grid:
        s = scorer(apply_decoder(df, dec, calibrator, extra))
        results.append((s, dec))
        print(f"  {dec} -> {s:.5f}", flush=True)
    best = max(results, key=lambda r: r[0])
    return dict(best[1], oof_f05=round(best[0], 5)), results
