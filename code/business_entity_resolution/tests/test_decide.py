"""Tests for the decision layer."""
import numpy as np
import pandas as pd

from src.decide import apply_decoder, one_to_one, select_expected_f05, select_threshold


def _df(rows):
    return pd.DataFrame(rows, columns=["s1_rid", "pool_rid", "p"])


def test_one_to_one_keeps_best_group():
    d = one_to_one(_df([[0, 7, 0.9], [1, 7, 0.6], [1, 8, 0.8]]))
    assert sorted(map(tuple, d[["s1_rid", "pool_rid"]].to_numpy())) == [(0, 7), (1, 8)]


def test_threshold_with_empty_gate():
    d = _df([[0, 1, 0.9], [0, 2, 0.4], [1, 3, 0.45]])
    s = select_threshold(d, t_match=0.4, t_empty=0.5)
    assert sorted(s["pool_rid"]) == [1, 2]


def test_expected_f05_prefers_confident_sets_and_empty():
    d = _df([[0, 1, 0.97], [0, 2, 0.95], [0, 3, 0.05], [1, 4, 0.08], [2, 5, 0.9]])
    s = select_expected_f05(d, extra=0.0, p_min=0.01, kmax=12, n_draws=256, seed=0)
    assert sorted(s["pool_rid"]) == [1, 2, 5]


def test_apply_decoder_threshold_roundtrip():
    d = _df([[0, 1, 0.9], [1, 1, 0.8], [1, 2, 0.2]])
    s = apply_decoder(d, {"method": "threshold", "t_match": 0.5, "t_empty": 0.5, "tau": 1.0}, None, 0.0)
    assert list(map(tuple, s[["s1_rid", "pool_rid"]].to_numpy())) == [(0, 1)]
