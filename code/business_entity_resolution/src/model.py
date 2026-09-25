"""LightGBM models: GroupKFold out-of-fold training (grouped by S1 entity) and full refits."""
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from . import config


def feature_cols(df, prefix="f_"):
    """Model input columns: those starting with `prefix` (never ids, labels or country)."""
    return [c for c in df.columns if c.startswith(prefix)]


def fit_rows(idx, train_mask=None, sample_weight=None):
    """Restrict row positions `idx` to the training mask; return (rows, their weights or None)."""
    if train_mask is not None:
        idx = idx[np.asarray(train_mask)[idx]]
    return idx, (None if sample_weight is None else np.asarray(sample_weight)[idx])


def train_oof(X, y, groups, params=None, n_folds=None, train_mask=None, sample_weight=None):
    """Fit one LightGBM per GroupKFold fold; return (oof_pred float32, fitted models).

    Each fold fits only on its masked (and weighted) training rows but predicts every
    validation row, so the OOF vector covers all rows.
    """
    params = dict(config.LGB_PARAMS if params is None else params)
    n_folds = n_folds or config.N_FOLDS
    y = np.asarray(y)
    oof = np.zeros(len(y), np.float32)
    models = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        tr, w = fit_rows(tr, train_mask, sample_weight)
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr], sample_weight=w, eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(config.EARLY_STOP, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
        print(f"  fold {k}: best_iter={m.best_iteration_}", flush=True)
    return oof, models


def mean_best_iter(models):
    """Average best iteration across fold models (+10% for the larger full-data refit)."""
    its = [m.best_iteration_ or m.n_estimators for m in models]
    return max(10, int(np.mean(its) * 1.1))


def fit_full(X, y, n_estimators, params=None, train_mask=None, sample_weight=None):
    """Refit on all (masked, weighted) rows with a fixed number of trees (test inference and transfer)."""
    p = dict(config.LGB_PARAMS if params is None else params)
    p["n_estimators"] = n_estimators
    rows, w = fit_rows(np.arange(len(X)), train_mask, sample_weight)
    m = lgb.LGBMClassifier(**p)
    m.fit(X.iloc[rows], np.asarray(y)[rows], sample_weight=w)
    return m
