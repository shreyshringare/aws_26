"""Tests for entity sampling and training-row selection."""
import numpy as np
import pandas as pd

from src.sampling import match_bin, stratified_s1, training_rows


def test_match_bin():
    assert list(match_bin([0, 1, 2, 3, 4, 5, 9])) == [0, 1, 1, 2, 2, 3, 3]


def test_stratified_s1_keeps_every_stratum_share():
    n = 4000
    s1n = pd.DataFrame({"rid": np.arange(n), "country": np.where(np.arange(n) % 4 == 0, "India", "US")})
    n_matches = np.arange(n) % 7  # bins 0, 1-2, 3-4, 5+
    rids = stratified_s1(s1n, n_matches, 0.1, seed=0)
    assert len(np.unique(rids)) == len(rids) and np.all(np.diff(rids) > 0)
    got = pd.crosstab(s1n["country"].to_numpy()[rids], match_bin(n_matches[rids]))
    full = pd.crosstab(s1n["country"], match_bin(n_matches))
    assert np.allclose(got.to_numpy(), full.to_numpy() * 0.1, atol=1)
    assert np.array_equal(stratified_s1(s1n, n_matches, 1.0, seed=0), s1n["rid"].to_numpy())
    assert np.array_equal(rids, stratified_s1(s1n, n_matches, 0.1, seed=0))


def test_training_rows_keeps_hard_rows_and_reweights_easy_negatives():
    rng = np.random.default_rng(1)
    n = 20000
    rank = rng.integers(0, 30, n)
    y = (rng.random(n) < 0.1).astype(np.int8)
    mask, w = training_rows(rank, y, keep_rank=10, easy_rate=0.25, seed=0)
    hard = (y == 1) | (rank < 10)
    assert mask[hard].all() and (w[hard] == 1).all()
    easy = mask & ~hard
    assert abs(easy.sum() / (~hard).sum() - 0.25) < 0.02
    assert (w[easy] == 4).all() and (w[~mask] == 0).all()
    assert abs(w.sum() - n) / n < 0.03  # weights keep the negative mass unbiased
    m1, w1 = training_rows(rank, y, easy_rate=1.0)
    assert m1.all() and (w1 == 1).all()
