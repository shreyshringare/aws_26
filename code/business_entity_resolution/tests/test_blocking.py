"""Tests for key generation and per-country IDF blocking."""
import pandas as pd

from src.blocking import generate_candidates, record_keys
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
