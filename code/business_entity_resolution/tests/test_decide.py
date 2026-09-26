"""Tests for the decision layer."""
import numpy as np
import pandas as pd

from src import config, decide
from src.decide import (apply_decoder, correlated_hits, one_to_one, select_expected_f05, select_threshold,
                        select_threshold_vocab_aware, tune)


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


def test_correlated_hits_preserves_marginal_probabilities():
    """Exp J2 (docs/RESEARCH_JOINT_S2S3.md Option B): the two-level draw must not bias each slot's
    own marginal hit rate, even though slots sharing a cluster id are drawn together."""
    P = np.array([[0.7, 0.7]], np.float32)
    C = np.array([[0, 0]], np.int64)
    rng = np.random.default_rng(0)
    hits = correlated_hits(P, C, rng, n_draws=20_000)
    rate = hits.mean(axis=1)[0]
    assert abs(rate[0] - 0.7) < 0.02
    assert abs(rate[1] - 0.7) < 0.02


def test_correlated_hits_induces_positive_correlation_within_cluster():
    P = np.array([[0.5, 0.5]], np.float32)
    C = np.array([[0, 0]], np.int64)
    rng = np.random.default_rng(1)
    hits = correlated_hits(P, C, rng, n_draws=20_000)
    both = (hits[:, :, 0] & hits[:, :, 1]).mean()
    assert both > 0.25 + 0.05  # independent baseline is 0.5*0.5 = 0.25


def test_correlated_hits_independent_across_different_clusters():
    P = np.array([[0.5, 0.5]], np.float32)
    C = np.array([[0, 1]], np.int64)
    rng = np.random.default_rng(2)
    hits = correlated_hits(P, C, rng, n_draws=20_000)
    both = (hits[:, :, 0] & hits[:, :, 1]).mean()
    assert abs(both - 0.25) < 0.03


def test_select_expected_f05_uses_correlated_hits_only_when_comp_col_given(monkeypatch):
    d = pd.DataFrame({"s1_rid": [0, 0], "pool_rid": [1, 2], "p": np.float32([0.6, 0.6]), "_comp": [0, 0]})
    calls = []
    orig = decide.correlated_hits
    monkeypatch.setattr(decide, "correlated_hits", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    select_expected_f05(d, extra=0.0, seed=0, n_draws=50)
    assert calls == []
    select_expected_f05(d, extra=0.0, seed=0, n_draws=50, comp_col="_comp")
    assert calls == [1]


def test_apply_decoder_passes_comp_col_only_when_dec_requests_joint(monkeypatch):
    d = pd.DataFrame({"s1_rid": [0, 0], "pool_rid": [1, 2], "p": np.float32([0.6, 0.6]), "_comp": [0, 0]})
    calls = []
    orig = decide.correlated_hits
    monkeypatch.setattr(decide, "correlated_hits", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    apply_decoder(d, {"method": "ef05", "tau": 1.0, "lam": 0.0, "joint": False}, None, 0.0, comp_col="_comp")
    assert calls == []
    apply_decoder(d, {"method": "ef05", "tau": 1.0, "lam": 0.0, "joint": True}, None, 0.0, comp_col="_comp")
    assert calls == [1]


def test_tune_doubles_ef05_grid_over_joint_when_comp_col_given():
    """Threshold-method grid points are untouched by `joint` (it only means anything for ef05);
    ef05 points double again on top of the existing one_to_one doubling."""
    d = pd.DataFrame({"s1_rid": [0, 0], "pool_rid": [1, 2], "p": np.float32([0.9, 0.1]), "_comp": [0, 1]})
    n_thresh = len(config.GRID_T_MATCH) * len(config.GRID_T_EMPTY)
    n_ef05 = len(config.GRID_TAU) * len(config.GRID_LAM)
    best, results = tune(d, None, 0.0, scorer=len, comp_col="_comp")
    assert len(results) == 2 * (n_thresh + 2 * n_ef05)
    assert any(dec.get("joint") is True for _, dec in results)
    assert all("joint" not in dec for _, dec in results if dec["method"] == "threshold")


def test_tune_without_comp_col_keeps_grid_size_unchanged():
    """Backward compatibility: omitting comp_col must reproduce the pre-Exp-J2 grid exactly."""
    d = _df([[0, 7, 0.9], [1, 7, 0.6], [1, 8, 0.8]])
    n_base = (len(config.GRID_T_MATCH) * len(config.GRID_T_EMPTY)) + (len(config.GRID_TAU) * len(config.GRID_LAM))
    best, results = tune(d, None, 0.0, scorer=len)
    assert len(results) == 2 * n_base
    assert all("joint" not in dec for _, dec in results)


def test_select_threshold_vocab_aware_raises_gate_for_low_frequency_groups():
    """docs/NEXT_STEPS.md item 14 (France hardening): a group whose best candidate has low
    vocabulary-coverage (e.g. a rare/unseen name -- proxy for the France OOV risk) needs a higher
    bar before the decoder trusts a non-empty output; a well-covered group is unaffected."""
    d = pd.DataFrame({
        "s1_rid": [0, 0, 1, 1], "pool_rid": [1, 2, 3, 4],
        "p": np.float32([0.6, 0.4, 0.6, 0.4]), "freq": np.float32([0.0, 0.0, 5.0, 5.0]),
    })
    sel = select_threshold_vocab_aware(d, "freq", t_match=0.3, t_empty=0.5,
                                       low_freq_cutoff=1.0, empty_boost=0.2)
    assert 0 not in sel["s1_rid"].to_numpy()  # rare group gated (0.6 < 0.5 + 0.2)
    assert set(sel[sel["s1_rid"] == 1]["pool_rid"]) == {3, 4}  # common group unaffected


def test_select_threshold_vocab_aware_matches_plain_threshold_at_zero_boost():
    d = pd.DataFrame({"s1_rid": [0, 0], "pool_rid": [1, 2], "p": np.float32([0.6, 0.4]),
                      "freq": np.float32([0.0, 0.0])})
    plain = select_threshold(d, t_match=0.3, t_empty=0.5)
    vocab = select_threshold_vocab_aware(d, "freq", t_match=0.3, t_empty=0.5,
                                         low_freq_cutoff=1.0, empty_boost=0.0)
    assert plain.equals(vocab)


def test_tune_segments_calls_tune_once_per_segment(monkeypatch):
    """docs/NEXT_STEPS.md item 15: segment-specific thresholds tune independently per segment
    value (never per country -- segment_col here is a stand-in for is_s3/addr-empty/script)."""
    d = pd.DataFrame({"s1_rid": [0, 1], "pool_rid": [1, 2], "p": np.float32([0.9, 0.9]), "seg": ["a", "b"]})
    calls = []

    def spy(df, calibrator, extra, scorer, comp_col=None):
        calls.append(sorted(df["pool_rid"]))
        return {"tag": "dec"}, []

    import src.decide as decide_mod
    monkeypatch.setattr(decide_mod, "tune", spy)
    out = decide.tune_segments(d, None, 0.0, scorer=len, segment_col="seg")
    assert sorted(calls) == [[1], [2]]
    assert set(out) == {"a", "b"}


def test_apply_segmented_decoder_routes_each_segment_to_its_own_decoder(monkeypatch):
    d = pd.DataFrame({"s1_rid": [0, 1], "pool_rid": [1, 2], "p": np.float32([0.9, 0.9]), "seg": ["a", "b"]})
    calls = []

    def spy(df, dec, calibrator, extra, comp_col=None):
        calls.append((dec["tag"], sorted(df["pool_rid"])))
        return df[["s1_rid", "pool_rid", "p"]]

    import src.decide as decide_mod
    monkeypatch.setattr(decide_mod, "apply_decoder", spy)
    decoders = {"a": {"tag": "dec_a"}, "b": {"tag": "dec_b"}}
    out = decide.apply_segmented_decoder(d, decoders, "seg", None, 0.0)
    assert sorted(calls) == [("dec_a", [1]), ("dec_b", [2])]
    assert sorted(out["pool_rid"]) == [1, 2]


def test_tune_selects_one_to_one_false_when_it_scores_higher():
    """With a scorer that simply rewards more selected rows, one_to_one=False (which can only
    produce a superset of one_to_one=True's rows for the same thresholds) must win -- proving
    the flag is measured via the scorer callback, not asserted/hardcoded in code."""
    d = _df([[0, 7, 0.9], [1, 7, 0.9], [0, 8, 0.9], [1, 8, 0.9]])
    best, results = tune(d, None, 0.0, scorer=len)
    assert best["one_to_one"] is False
    sel = apply_decoder(d, best, None, 0.0)
    assert len(sel) == len(d)  # both duplicate-pool rows survive under the winning decoder
