"""Tests for key generation and per-country IDF blocking."""
import pandas as pd

from src import config
from src.blocking import FAMS, cross_keys, generate_candidates, name_keys, recall_at_ks, record_keys
from src.config import ADDR_FAMS, NAME_FAMS
from src.normalize import normalize_frame


def _frame(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def test_record_keys_families():
    keys = record_keys("peak highland fast", "veterans pkwy marshfield wi", "3029_veterans")
    fams = {f for f, _ in keys}
    assert fams == {"n", "nb", "nm", "np", "nc", "n2", "a", "ab", "na"}
    assert ("na", "peak@marshfield") in keys and ("na", "fast@wi") in keys
    assert ("nm", "fhp") in keys and ("ab", "3029_veterans") in keys
    assert ("np", "hglnd") in keys and ("nc", "peakhighlandfast") in keys
    assert {k for f, k in keys if f == "n2"} == {"highland_peak", "fast_peak", "fast_highland"}


def test_record_keys_bridge_domains_reorders_and_scripts():
    def fam(name, f):
        return {k for g, k in record_keys(name, "", "") if g == f}
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
    assert list(multi.columns) == ["s1_rid", "pool_rid", "block_score", "block_rank", "score_name",
                                    "score_addr", "rank_name", "rank_addr", "rank_all", "n_channels"]
    r = multi.set_index("pool_rid")
    assert r.loc[1, "score_name"] > r.loc[0, "score_name"] and r.loc[1, "score_addr"] == 0
    assert (r.block_rank == 0).all()


def test_generate_candidates_reports_per_channel_rank_and_n_channels():
    """A pair found only by the name channel gets a missing (=k) rank in addr, and vice versa;
    n_channels counts only the non-'all' channels that actually retrieved the pair."""
    s1 = normalize_frame(_frame([["S1-1", "Peak Highland Fast Inc", "3029 Veterans Parkway, Marshfield, WI", "US"]]),
                          None)
    pool = normalize_frame(_frame([
        ["S3-9", "Peak Highland Fast", "", "US"],                                    # name-only match
        ["S2-7", "Totally Different Biz", "3029 Veterans Pkwy, Marshfield, Wisconsin", "US"],  # addr-only match
    ]), None)
    c = generate_candidates(s1, pool, channels={"name": (NAME_FAMS, 5), "addr": (ADDR_FAMS, 5), "all": (FAMS, 10)},
                            max_df=10, chunk=1, n_jobs=1)
    r = c.set_index("pool_rid")
    K_NAME, K_ADDR = 5, 5
    assert r.loc[0, "rank_name"] < K_NAME and r.loc[0, "rank_addr"] == K_ADDR  # name found it, addr didn't
    assert r.loc[1, "rank_addr"] < K_ADDR and r.loc[1, "rank_name"] == K_NAME  # addr found it, name didn't
    assert r.loc[0, "n_channels"] == 1 and r.loc[1, "n_channels"] == 1  # each found by exactly one sub-channel


def test_cross_keys_limits_and_min_length():
    keys = cross_keys("a sri ram traders co".split(), "b main rd main x gandhi nagar pune road".split())
    names = {k.split("@")[0] for _, k in keys}
    places = {k.split("@")[1] for _, k in keys}
    assert names == {"sri", "ram", "traders"}  # first 3 tokens with len >= 2
    assert places == {"main", "rd", "gandhi", "nagar", "pune", "road"}  # 6 distinct, len >= 2
    assert len(keys) == 18 and cross_keys([], ["main"]) == set()


def test_cross_family_ranks_common_name_at_its_address():
    s1 = normalize_frame(_frame([["S1-1", "Sri Traders", "Gandhi Nagar, Pune", "India"]]), None)
    pool = normalize_frame(_frame([
        ["S2-1", "Sri Traders", "Station Road, Nashik", "India"],
        ["S2-2", "Sri Traders", "Gandhi Nagar, Pune", "India"],
        ["S2-3", "Sri Traders", "Mall Road, Agra", "India"],
    ]), None)
    only_na = {f: float(f == "na") for f in FAMS}
    c = generate_candidates(s1, pool, channels={"all": (FAMS, 1)}, weights=only_na, max_df=10, chunk=1, n_jobs=1)
    assert list(c.pool_rid) == [1]


def test_name_key_mode_defaults_to_keys_and_matches_all_name_fams():
    """The default config.NAME_KEY_MODE ('keys') is the toggle's off position: NAME_KEY_FAMS
    must equal NAME_FAMS exactly, so committed behavior is unchanged unless someone flips
    ER_NAME_KEY_MODE=features_only (Prompt 8 / EDA_TEAMMATE.md §9 measurement toggle)."""
    assert config.NAME_KEY_MODE == "keys"
    assert config.NAME_KEY_FAMS == NAME_FAMS


def test_name_keys_features_only_mode_drops_abbrev_families(monkeypatch):
    """With NAME_KEY_FAMS narrowed to just 'n' (the features_only toggle position), name_keys
    and record_keys stop emitting nb/nm/np/nc/n2 as blocking-key families; n/a/ab/na (address
    and cross-key families, untouched by this toggle) still appear."""
    monkeypatch.setattr(config, "NAME_KEY_FAMS", ("n",))
    toks = "peak highland fast".split()
    nk = name_keys(toks)
    assert {f for f, _ in nk} == {"n"}
    assert nk == {("n", "peak"), ("n", "highland"), ("n", "fast")}

    keys = record_keys("peak highland fast", "veterans pkwy marshfield wi", "3029_veterans")
    fams = {f for f, _ in keys}
    assert fams == {"n", "a", "ab", "na"}
    assert not fams & {"nb", "nm", "np", "nc", "n2"}


def test_name_keys_features_only_mode_can_drop_all_name_families(monkeypatch):
    """An empty NAME_KEY_FAMS (the most extreme features_only position) drops the 'n' family
    too, proving the gate is driven purely by config, not hardcoded to always keep 'n'."""
    monkeypatch.setattr(config, "NAME_KEY_FAMS", ())
    assert name_keys("peak highland fast".split()) == set()


def test_recall_at_ks_matches_generate_candidates_per_k_and_sweeps_monotonically():
    """recall_at_ks must agree with calling generate_candidates separately at each K (the naive,
    non-reused way), while computing all K's in one pass; recall/cands_per_entity must be
    non-decreasing in K and plateau once every scored (nonzero) pair is already retrieved."""
    s1 = normalize_frame(_frame([
        ["S1-1", "Peak Highland Fast Inc", "3029 Veterans Parkway, Marshfield, WI", "US"],
        ["S1-2", "Zephay Labs Inc", "2621 Cotten Road, Tyler, TX", "US"],
    ]), None)
    pool = normalize_frame(_frame([
        ["S3-9", "Peak Highland Fast", "3029 Veterans Pkwy, Marshfield, Wisconsin", "US"],
        ["S3-10", "Peak Highland Fast LLC", "1000 Other Ave, Nowhere, WI", "US"],
        ["S2-7", "Zephay Labs", "2621 Cotten Rd, Tyler, Texas", "US"],
    ]), None)
    # s1=0 has two true matches (an exact-ish one and a name-only-similar one that should rank
    # lower); s1=1 has one. The name-only match should only become reachable once K >= 2.
    pos = pd.DataFrame({"s1_rid": [0, 0, 1], "pool_rid": [0, 1, 2]})

    rep = recall_at_ks(s1, pool, pos, ks=[1, 2, 3], max_df=10, chunk=2, n_jobs=1)
    assert set(rep) == {1, 2, 3}
    assert rep[1]["pair_recall"] < rep[2]["pair_recall"] == rep[3]["pair_recall"] == 1.0
    assert rep[1]["cands_per_entity"] < rep[2]["cands_per_entity"] == rep[3]["cands_per_entity"]

    for k in (1, 2, 3):
        c = generate_candidates(s1, pool, channels={"all": (FAMS, k)}, max_df=10, chunk=2, n_jobs=1)
        hit = sum(1 for _, r in pos.iterrows()
                  if ((c.s1_rid == r.s1_rid) & (c.pool_rid == r.pool_rid)).any())
        assert rep[k]["pair_recall"] == hit / len(pos)
        assert rep[k]["cands_per_entity"] == len(c) / len(s1)


def test_recall_at_ks_no_shared_keys_returns_zero_recall_and_zero_cands():
    """When no country/key overlap exists at all, every K reports 0 recall, 0 cands/entity
    (the 'no countries matched' short-circuit path)."""
    s1 = normalize_frame(_frame([["S1-1", "Zzz Unique Corp", "1 Nowhere Ln", "US"]]), None)
    pool = normalize_frame(_frame([["S2-1", "Totally Different Biz", "Other Place", "France"]]), None)
    pos = pd.DataFrame({"s1_rid": [0], "pool_rid": [0]})
    rep = recall_at_ks(s1, pool, pos, ks=[1, 5], max_df=10, chunk=2, n_jobs=1)
    assert rep[1] == {"pair_recall": 0.0, "cands_per_entity": 0.0}
    assert rep[5] == {"pair_recall": 0.0, "cands_per_entity": 0.0}


def test_recall_at_ks_empty_ks_returns_empty_dict():
    """An empty `ks` has nothing to sweep; recall_at_ks must return {} rather than crash on
    `ks[-1]` (reported bug: bare IndexError on ks=[])."""
    s1 = normalize_frame(_frame([["S1-1", "Zzz Unique Corp", "1 Nowhere Ln", "US"]]), None)
    pool = normalize_frame(_frame([["S2-1", "Zzz Unique Corp", "1 Nowhere Ln", "US"]]), None)
    pos = pd.DataFrame({"s1_rid": [0], "pool_rid": [0]})
    assert recall_at_ks(s1, pool, pos, ks=[], max_df=10, chunk=2, n_jobs=1) == {}


def test_recall_at_ks_channel_param_restricts_scoring_families():
    """Passing `fams` restricts the sweep to one channel's families (e.g. addr-only), matching
    the "what if I raised K on the addr channel" use case from IDEAS.md."""
    s1 = normalize_frame(_frame([["S1-1", "Peak Highland Fast Inc",
                                  "3029 Veterans Parkway, Marshfield, WI", "US"]]), None)
    pool = normalize_frame(_frame([
        ["S3-9", "Totally Unrelated Name", "3029 Veterans Pkwy, Marshfield, Wisconsin", "US"],
    ]), None)
    pos = pd.DataFrame({"s1_rid": [0], "pool_rid": [0]})
    addr_only = recall_at_ks(s1, pool, pos, ks=[1], fams=ADDR_FAMS, max_df=10, chunk=2, n_jobs=1)
    name_only = recall_at_ks(s1, pool, pos, ks=[1], fams=NAME_FAMS, max_df=10, chunk=2, n_jobs=1)
    assert addr_only[1]["pair_recall"] == 1.0  # address overlaps despite unrelated names
    assert name_only[1]["pair_recall"] == 0.0  # no name overlap at all
