"""LightGBM models: GroupKFold out-of-fold training (grouped by S1 entity) and full refits."""
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from . import config


def feature_cols(df, prefix="f_"):
    """Model input columns: those starting with `prefix` (never ids, labels or country)."""
    return [c for c in df.columns if c.startswith(prefix)]


def train_oof(X, y, groups, params=None, n_folds=None):
    """Fit one LightGBM per GroupKFold fold; return (oof_pred float32, fitted models)."""
    params = dict(config.LGB_PARAMS if params is None else params)
    n_folds = n_folds or config.N_FOLDS
    y = np.asarray(y)
    oof = np.zeros(len(y), np.float32)
    models = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(config.EARLY_STOP, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
        print(f"  fold {k}: best_iter={m.best_iteration_}", flush=True)
    return oof, models


def mean_best_iter(models):
    """Average best iteration across fold models (+10% for the larger full-data refit)."""
    its = [m.best_iteration_ or m.n_estimators for m in models]
    return max(10, int(np.mean(its) * 1.1))


def fit_full(X, y, n_estimators, params=None):
    """Refit on all rows with a fixed number of trees (used for test inference and transfer)."""
    p = dict(config.LGB_PARAMS if params is None else params)
    p["n_estimators"] = n_estimators
    m = lgb.LGBMClassifier(**p)
    m.fit(X, np.asarray(y))
    return m
