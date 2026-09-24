"""Exact challenge metric (macro F0.5 per S1 entity) and diagnostic breakdowns."""
import numpy as np
import pandas as pd


def f05_entity(pred, true):
    """F0.5 for one S1 entity given predicted and true id sets (challenge rules)."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def score(pred, gt, s1_ids=None):
    """Macro F0.5 over `s1_ids` (default: all gt keys). Returns (mean, per-entity array)."""
    ids = list(gt.keys()) if s1_ids is None else list(s1_ids)
    vals = np.fromiter(
        (f05_entity(set(pred.get(s, ())), gt.get(s, set())) for s in ids),
        dtype=np.float64, count=len(ids),
    )
    return float(vals.mean()) if len(vals) else float("nan"), vals


def breakdown(pred, gt, s1_ids, country):
    """Score all / per country / singletons / matched. `country` aligns with `s1_ids`."""
    ids = list(s1_ids)
    _, vals = score(pred, gt, ids)
    country = np.asarray(country)
    single = np.array([len(gt.get(s, ())) == 0 for s in ids])
    out = {"all": vals.mean()}
    for c in pd.unique(country):
        out[f"country={c}"] = vals[country == c].mean()
    out["singletons"] = vals[single].mean() if single.any() else float("nan")
    out["matched"] = vals[~single].mean() if (~single).any() else float("nan")
    out["n"] = len(ids)
    return out


def blocking_report(cands, gt, s1_ids):
    """Blocking recall (pair level), perfect-recall-entity share and candidates/entity.

    `cands` maps s1_id -> iterable of candidate ids.
    """
    tot_true = hit = full = 0
    ncand = 0
    for s in s1_ids:
        t = gt.get(s, set())
        c = set(cands.get(s, ()))
        ncand += len(c)
        h = len(t & c)
        tot_true += len(t)
        hit += h
        full += h == len(t)
    n = max(len(s1_ids), 1)
    return {
        "pair_recall": hit / max(tot_true, 1),
        "entity_full_recall": full / n,
        "cands_per_entity": ncand / n,
    }
