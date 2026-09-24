"""Tests for key generation and per-country IDF blocking."""
import pandas as pd

from src.blocking import FAMS, generate_candidates, record_keys
from src.config import ADDR_FAMS, NAME_FAMS
from src.normalize import normalize_frame


def _frame(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def test_record_keys_families():
    keys = record_keys("peak highland fast", "veterans pkwy marshfield wi", "3029_veterans", "")
    fams = {f for f, _ in keys}
    assert fams == {"n", "nb", "nm", "np", "nc", "n2", "a", "ab"}
    assert ("nm", "fhp") in keys and ("ab", "3029_veterans") in keys
    assert ("np", "hglnd") in keys and ("nc", "peakhighlandfast") in keys
    assert {k for f, k in keys if f == "n2"} == {"highland_peak", "fast_peak", "fast_highland"}


def test_record_keys_bridge_domains_reorders_and_scripts():
    def fam(name, f):
        return {k for g, k in record_keys(name, "", "", "") if g == f}
    assert fam("nexloix advisory", "nc") == fam("nexloixadvisory", "nc") == {"nexloixadvisory"}
    assert fam("abc", "nc") == set()
    assert fam("great trading", "n2") & fam("great private trading", "n2") == {"great_trading"}
    assert fam("perfect marketing", "np") == fam("prphektt maarketting", "np")


def test_generate_candidates_finds_match_and_respects_country():
    s1 = normalize_frame(_frame([
        ["S1-1", "Peak Highland Fast Inc", "3029 Veterans Parkway, Marshfield, WI", "US"],
        ["S1-2", "Zephay Labs Inc", "2621 Cotten Road, Tyler, TX", "US"],
        ["S1-3", "Team Ecole", "175 Boulevard Roosevelt, Bordeaux", "France"],
    ]), None)
    pool = normalize_frame(_frame([
        ["S3-9", "Peak Highland Fast", "3029 Veterans Pkwy, Marshfield, Wisconsin", "US"],
        ["S2-7", "Zephay Labs", "2621 Cotten Rd, Tyler, Texas", "US"],
        ["S2-8", "Team Ecole SARL", "175 BD ROOSEVELT, BORDEAUX", "France"],
        ["S2-5", "Team Ecole", "175 Boulevard Roosevelt, Bordeaux", "US"],
    ]), None)
    c = generate_candidates(s1, pool, topk=2, max_df=10, chunk=2, n_jobs=1)
    top = c[c.block_rank == 0].set_index("s1_rid")["pool_rid"].to_dict()
    assert top == {0: 0, 1: 1, 2: 2}
    fr = c[c.s1_rid == 2]
    assert set(fr.pool_rid) == {2}  # US record S2-5 never offered to a France S1


def test_name_channel_keeps_name_only_match_crowded_out_by_address():
    s1 = normalize_frame(_frame([["S1-1", "Smart Marketing Ltd", "12 Main Street, Springfield, IL", "US"]]), None)
    pool = normalize_frame(_frame([
        ["S2-1", "Smart Marketing Solutions", "12 Main St, Springfield, Illinois", "US"],
        ["S3-2", "Smart Marketing Limited", "", "US"],
    ]), None)
    single = generate_candidates(s1, pool, channels={"all": (FAMS, 1)}, max_df=10, chunk=1, n_jobs=1)
    assert list(single.pool_rid) == [0]  # the address-sharing sister business wins the joint score
    multi = generate_candidates(s1, pool, channels={"name": (NAME_FAMS, 1), "addr": (ADDR_FAMS, 1),
                                                    "all": (FAMS, 1)}, max_df=10, chunk=1, n_jobs=1)
    assert set(multi.pool_rid) == {0, 1}
    assert list(multi.columns) == ["s1_rid", "pool_rid", "block_score", "block_rank", "score_name", "score_addr"]
    r = multi.set_index("pool_rid")
    assert r.loc[1, "score_name"] > r.loc[0, "score_name"] and r.loc[1, "score_addr"] == 0
    assert (r.block_rank == 0).all()
