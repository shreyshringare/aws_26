"""Tests for GroupKFold LightGBM training."""
import numpy as np
import pandas as pd

from src.model import feature_cols, fit_full, mean_best_iter, train_oof


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
