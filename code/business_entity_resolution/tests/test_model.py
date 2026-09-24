"""Tests for GroupKFold LightGBM training."""
import numpy as np
import pandas as pd

from src.model import feature_cols, fit_full, fit_rows, mean_best_iter, train_oof


def test_train_oof_groups_and_signal():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"f_a": rng.random(n).astype(np.float32), "f_b": rng.random(n).astype(np.float32),
                      "s1_rid": np.repeat(np.arange(n // 10), 10)})
    y = (X["f_a"] > 0.7).astype(np.int8).to_numpy()
    cols = feature_cols(X)
    assert cols == ["f_a", "f_b"]
    oof, models = train_oof(X[cols], y, X["s1_rid"].to_numpy(),
                            params=dict(objective="binary", n_estimators=50, verbose=-1, random_state=0), n_folds=3)
    assert len(models) == 3 and oof.shape == (n,)
    assert ((oof > 0.5) == y).mean() > 0.95
    m = fit_full(X[cols], y, n_estimators=mean_best_iter(models))
    assert m.predict_proba(X[cols])[:, 1].shape == (n,)


def test_train_oof_mask_fits_subset_but_predicts_all_rows():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"f_a": rng.random(n).astype(np.float32)})
    y = (X["f_a"] > 0.5).astype(np.int8).to_numpy()
    groups = np.repeat(np.arange(n // 10), 10)
    mask = np.arange(n) % 3 != 0
    w = np.where(mask, 2.0, 0.0).astype(np.float32)
    params = dict(objective="binary", n_estimators=30, verbose=-1, random_state=0, min_child_samples=5)
    oof, models = train_oof(X, y, groups, params=params, n_folds=3, train_mask=mask, sample_weight=w)
    assert (oof > 0).all() and ((oof > 0.5) == y).mean() > 0.95  # masked-out rows are still predicted
    m = fit_full(X, y, n_estimators=20, params=params, train_mask=mask, sample_weight=w)
    assert m.predict_proba(X)[:, 1].shape == (n,)


def test_fit_rows():
    idx = np.array([0, 2, 3, 5])
    mask = np.array([True, True, False, True, True, False])
    w = np.arange(6, dtype=np.float32)
    rows, wr = fit_rows(idx, mask, w)
    assert list(rows) == [0, 3] and list(wr) == [0, 3]
    rows, wr = fit_rows(idx)
    assert list(rows) == [0, 2, 3, 5] and wr is None
