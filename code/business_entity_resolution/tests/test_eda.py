"""Tests for the EDA v2 pure functions on tiny synthetic frames."""
import numpy as np
import pandas as pd

from src.eda import (NORM_COLS, cardinality, cardinality_hist, collision_negatives, explode_gt,
                     md_table, noise_flags, one_to_one, oov_rates, pair_key_sharing, rate_table)


def _rec(**kw):
    """One normalized record with defaults for every NORM_COLS column."""
    base = {c: "" for c in NORM_COLS}
    base.update(country="US", legal_first=0, script_flag=0)
    base.update(kw)
    return base


def test_cardinality_hist_and_one_to_one():
    """Histogram shares, per-country singleton rate and one-to-one count."""
    gt = pd.DataFrame({"source1_entity_id": ["A", "B", "C"],
                       "matched_entity_ids": ["S2-1,S3-1", "", "S3-2,S3-1"]})
    card = cardinality(gt)
    assert card["n"].tolist() == [2, 0, 2]
    assert card["n_s3"].tolist() == [1, 0, 2]
    hist, by = cardinality_hist(card, pd.Series(["US", "US", "India"]))
    assert np.isclose(hist.loc[2, "share_total"], 2 / 3)
    assert np.isclose(by.loc["US", "singleton"], 0.5)
    oto = one_to_one(explode_gt(gt))
    assert oto.loc[0, "ids_in_2plus_s1"] == 1 and oto.loc[0, "matched_pool_ids"] == 3


def test_noise_flags_rates():
    """Each synthetic pair triggers exactly the intended operators."""
    a = pd.DataFrame([
        _rec(name_norm="acme llc", name_core="acme", legal="llc", nums="3424", addr_norm="3424 elm st",
             addr_alpha="elm st", name_raw="Acme LLC"),
        _rec(name_norm="blue sky", name_core="blue sky", nums="12", addr_norm="12 oak rd", addr_alpha="oak rd",
             name_raw="Blue Sky", country="India"),
    ])
    b = pd.DataFrame([
        _rec(name_norm="llc acme", name_core="acme", legal="llc", legal_first=1, nums="424",
             addr_norm="424 elm st", addr_alpha="elm st", name_raw="LLC Acme"),
        _rec(name_norm="sky blue services", name_core="sky blue services", nums="", addr_norm="",
             name_raw="स्काई", country="India"),
    ])
    f = noise_flags(a, b, generic={"services"})
    assert f.loc[0, "legal_moved"] and f.loc[0, "num_digit_del"] and f.loc[0, "name_core_eq"]
    assert not f.loc[0, "name_norm_eq"]
    assert f.loc[1, "diff_generic_only"] and f.loc[1, "addr_empty_either"] and f.loc[1, "num_missing_one"]
    assert f.loc[1, "pool_name_non_latin"]
    r = rate_table(f, a["country"].to_numpy())
    assert r.loc["legal_moved", "ALL"] == 50.0
    assert r.loc["legal_moved", "US"] == 100.0 and r.loc["legal_moved", "India"] == 0.0


def test_pair_key_sharing_and_oov():
    """Key-family sharing on matched pairs; OOV rates against a vocab."""
    a = pd.DataFrame([_rec(name_core="acme", addr_alpha="elm", postcode="12345")])
    b = pd.DataFrame([_rec(name_core="zeta", addr_alpha="elm", postcode="12345")])
    s = pair_key_sharing(a, b, ["US"])
    assert s.loc["n", "share_pairs_sharing"] == 0 and s.loc["a", "share_pairs_sharing"] == 1
    assert s.loc["na", "share_pairs_sharing"] == 0
    assert s.loc["any", "share_pairs_sharing"] == 1
    tok, rec = oov_rates(["a b", "c", ""], {"a", "c"})
    assert np.isclose(tok, 1 / 3) and np.isclose(rec, 0.5)


def test_collision_negatives_and_md_table():
    """Negatives = pool count of the key minus matched ones; markdown renders."""
    pool_vc = pd.Series([5, 1], index=pd.Index(np.array([10, 20], dtype=np.uint64)))
    t = collision_negatives(np.array([10, 20, 30], dtype=np.uint64), ["US", "US", "India"],
                            [2, 1, 0], pool_vc)
    assert t.loc["US", "mean_neg"] == 1.5 and t.loc["India", "share_with_neg"] == 0.0
    md = md_table(t)
    assert md.startswith("| country |") and "ALL" in md
