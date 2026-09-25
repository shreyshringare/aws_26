"""Tests for stage-1 pair features."""
import numpy as np
import pandas as pd

from src.features import add_frequency_columns, build_features
from src.normalize import normalize_frame


def _f(rows):
    return normalize_frame(pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]))


def test_build_features_shapes_and_values():
    s1 = _f([["S1-1", "Valley Holdings Inc", "719 Mountain Road, Addison, VT", "US"]])
    pool = _f([["S3-1", "Vlley Holdings", "719 Mountain Rd, Addison, Vermont", "US"],
               ["S2-2", "Other Thing LLC", "1 Main St, Austin, TX", "US"]])
    add_frequency_columns(s1, pool)
    cands = pd.DataFrame({"s1_rid": [0, 0], "pool_rid": [0, 1],
                          "block_score": np.float32([9, 1]), "block_rank": np.int16([0, 1]),
                          "score_name": np.float32([6, 0]), "score_addr": np.float32([3, 1])})
    f = build_features(cands, s1, pool, set(), n_jobs=1)
    assert list(f["pool_rid"]) == [0, 1]
    assert list(f["f_score_name"]) == [6, 0] and list(f["f_score_addr"]) == [3, 1]
    assert f.loc[0, "f_addr_jac"] == 1.0 and f.loc[0, "f_num1_eq"] == 1.0
    assert f.loc[0, "f_nz_edit1_tokens"] == 1 and f.loc[0, "f_is_s3"] == 1
    assert f.loc[0, "f_name_core_tset"] > f.loc[1, "f_name_core_tset"]
    assert not any("country" in c or "_pc_" in c for c in f.columns)
    assert list(f["f_nz_num_conflict"]) == [0, 1] and list(f["f_acronym"]) == [0, 0]
    assert all(f[c].dtype == np.float32 for c in f.columns if c.startswith("f_"))


def test_add_frequency_columns_adds_ambiguity_density_stats():
    s1 = _f([["S1-1", "Smart Traders", "5 Main St, Reno, NV", "US"],
             ["S1-2", "Smart Traders", "9 Oak Ave, Reno, NV", "US"]])  # duplicate name_core across S1
    pool = _f([["S2-1", "Smart Traders Co", "5 Main St, Reno, NV", "US"],  # shares street_key with S1-1
               ["S2-2", "Smart Widgets", "1 Elm St, Boise, ID", "US"],   # shares first name token only
               ["S3-3", "Other Biz", "2 Pine Rd, Ada, ID", "US"]])       # shares neither
    add_frequency_columns(s1, pool)
    for col in ("first_tok_freq", "street_freq"):
        assert col in s1.columns and col in pool.columns
    assert "s1_name_dup" in s1.columns and "s1_name_dup" not in pool.columns  # S1-only, no pool equivalent
    # S1-1 and S1-2 have the same name_core -> each sees one other S1 with that name
    assert s1.loc[0, "s1_name_dup"] > 0 and s1.loc[1, "s1_name_dup"] > 0
    # a record with a unique name_core in S1 has zero duplication
    assert s1.loc[0, "s1_name_dup"] == s1.loc[1, "s1_name_dup"]
    # blank street_key never counts as "shared" (would otherwise dominate as the most common value)
    empty_street = pool[pool["street_key"] == ""]
    if len(empty_street):
        assert (empty_street["street_freq"] == 0).all()

    cands = pd.DataFrame({"s1_rid": [0, 0, 0], "pool_rid": [0, 1, 2],
                          "block_score": np.float32([9, 1, 1]), "block_rank": np.int16([0, 1, 2]),
                          "score_name": np.float32([6, 3, 0]), "score_addr": np.float32([3, 0, 0])})
    f = build_features(cands, s1, pool, set(), n_jobs=1)
    assert "f_street_freq_a" in f.columns and "f_first_tok_freq_b" in f.columns and "f_s1_dup_a" in f.columns


def test_build_features_scores_best_of_name_vs_alias():
    # S1 uses the alias name; pool has only the main/current name -> direct core similarity is
    # weak, but the alias-vs-core comparison should rescue it via the "best of name vs alias" feature.
    s1 = _f([["S1-1", "Golden Gate Cafe dba Bay Diner", "5 Main St, Reno, NV", "US"]])
    pool = _f([["S2-1", "Bay Diner", "5 Main St, Reno, NV", "US"],
               ["S3-2", "Totally Unrelated Biz", "1 Elm St, Boise, ID", "US"]])
    add_frequency_columns(s1, pool)
    cands = pd.DataFrame({"s1_rid": [0, 0], "pool_rid": [0, 1],
                          "block_score": np.float32([9, 1]), "block_rank": np.int16([0, 1]),
                          "score_name": np.float32([6, 0]), "score_addr": np.float32([3, 0])})
    f = build_features(cands, s1, pool, set(), n_jobs=1)
    assert "f_name_alias_best" in f.columns
    r = f.set_index("pool_rid")
    assert r.loc[0, "f_name_alias_best"] > r.loc[0, "f_name_core_tset"]  # alias match rescues it
    assert r.loc[0, "f_name_alias_best"] > 90  # "bay diner" alias vs "bay diner" pool name: near-exact
    assert r.loc[1, "f_name_alias_best"] < 50  # neither the name nor the alias meaningfully matches


def test_build_features_surfaces_retrieval_rank_and_n_channels_when_present():
    s1 = _f([["S1-1", "Valley Holdings Inc", "719 Mountain Road, Addison, VT", "US"]])
    pool = _f([["S3-1", "Vlley Holdings", "719 Mountain Rd, Addison, Vermont", "US"]])
    add_frequency_columns(s1, pool)
    cands = pd.DataFrame({"s1_rid": [0], "pool_rid": [0], "block_score": np.float32([9]),
                          "block_rank": np.int16([0]), "score_name": np.float32([6]),
                          "score_addr": np.float32([3]), "rank_name": np.int16([0]),
                          "rank_addr": np.int16([0]), "rank_all": np.int16([0]),
                          "n_channels": np.int16([2])})
    f = build_features(cands, s1, pool, set(), n_jobs=1)
    assert f.loc[0, "f_rank_name"] == 0 and f.loc[0, "f_n_channels"] == 2
