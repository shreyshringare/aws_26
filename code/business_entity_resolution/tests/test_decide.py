"""Tests for the decision layer."""
import numpy as np
import pandas as pd

from src import config
from src.decide import apply_decoder, one_to_one, select_expected_f05, select_threshold, tune


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


def test_apply_decoder_one_to_one_defaults_true_when_key_absent():
    """A decoder dict with no 'one_to_one' key must behave exactly as before this flag existed:
    the shared pool record 7 is kept only in its best-scoring S1 group."""
    d = _df([[0, 7, 0.9], [1, 7, 0.6], [1, 8, 0.8]])
    dec = {"method": "threshold", "t_match": 0.1, "t_empty": 0.1, "tau": 1.0}
    s = apply_decoder(d, dec, None, 0.0)
    assert sorted(map(tuple, s[["s1_rid", "pool_rid"]].to_numpy())) == [(0, 7), (1, 8)]


def test_apply_decoder_one_to_one_true_explicit_matches_default():
    d = _df([[0, 7, 0.9], [1, 7, 0.6], [1, 8, 0.8]])
    base = {"method": "threshold", "t_match": 0.1, "t_empty": 0.1, "tau": 1.0}
    default = apply_decoder(d, base, None, 0.0)
    explicit = apply_decoder(d, dict(base, one_to_one=True), None, 0.0)
    assert sorted(map(tuple, default[["s1_rid", "pool_rid"]].to_numpy())) == \
        sorted(map(tuple, explicit[["s1_rid", "pool_rid"]].to_numpy()))


def test_apply_decoder_one_to_one_false_keeps_duplicate_pool_assignment():
    """dec = {'one_to_one': False, ...} must skip one_to_one entirely: pool record 7 can be kept
    under both S1 groups 0 and 1 (this is exactly the IDEAS.md B.8 untested-on-test-set scenario)."""
    d = _df([[0, 7, 0.9], [1, 7, 0.6], [1, 8, 0.8]])
    dec = {"method": "threshold", "t_match": 0.1, "t_empty": 0.1, "tau": 1.0, "one_to_one": False}
    s = apply_decoder(d, dec, None, 0.0)
    assert sorted(map(tuple, s[["s1_rid", "pool_rid"]].to_numpy())) == [(0, 7), (1, 7), (1, 8)]


def test_tune_grid_doubles_over_one_to_one_and_tags_every_result():
    """tune() must try every existing grid point with both one_to_one=True and one_to_one=False,
    so the result list is exactly twice as long as before, and every dec carries a real bool flag."""
    d = _df([[0, 7, 0.9], [1, 7, 0.6], [1, 8, 0.8]])
    best, results = tune(d, None, 0.0, scorer=len)
    n_base = (len(config.GRID_T_MATCH) * len(config.GRID_T_EMPTY)) + (len(config.GRID_TAU) * len(config.GRID_LAM))
    assert len(results) == 2 * n_base
    assert all(isinstance(dec.get("one_to_one"), bool) for _, dec in results)
    assert {dec["one_to_one"] for _, dec in results} == {True, False}
    assert "one_to_one" in best


def test_tune_selects_one_to_one_false_when_it_scores_higher():
    """With a scorer that simply rewards more selected rows, one_to_one=False (which can only
    produce a superset of one_to_one=True's rows for the same thresholds) must win -- proving
    the flag is measured via the scorer callback, not asserted/hardcoded in code."""
    d = _df([[0, 7, 0.9], [1, 7, 0.9], [0, 8, 0.9], [1, 8, 0.9]])
    best, results = tune(d, None, 0.0, scorer=len)
    assert best["one_to_one"] is False
    sel = apply_decoder(d, best, None, 0.0)
    assert len(sel) == len(d)  # both duplicate-pool rows survive under the winning decoder
