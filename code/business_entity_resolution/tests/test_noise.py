"""Tests for noise-channel features."""
from src.noise import NOISE_FEATS, ROW_COLS, acronym, digit_deletion, fit_generic_vocab, noise_row


def _row(**kw):
    """Row tuple in ROW_COLS order with empty defaults."""
    base = dict(name_core="", legal="", legal_first=0, nums="", addr_alpha="",
                name_norm="", name_raw="", addr_norm="", addr_raw="")
    base.update(kw)
    return tuple(base[k] for k in ROW_COLS)


def test_digit_deletion():
    assert digit_deletion("3424", "424") == 1
    assert digit_deletion("1201", "120") == 1
    assert digit_deletion("136", "36") == 1
    assert digit_deletion("3424", "3424") == 0
    assert digit_deletion("3424", "12") == 0


def test_generic_vocab():
    s1 = ["davis society care", "peak fast", "blue ocean"] * 100
    pool = ["davis society", "peak fast services", "blue ocean"] * 100
    v = fit_generic_vocab(s1, pool, min_count=50, min_share=0.5)
    assert v == {"care", "services"}


def test_generic_vocab_min_length_and_default_count():
    s1 = ["davis co care", "blue ocean"] * 400
    pool = ["davis care", "blue ocean xyz"] * 400
    assert fit_generic_vocab(s1, pool) == {"xyz"}  # 'co' too short
    assert fit_generic_vocab(s1[:500], pool[:500]) == set()  # 250 < default min_count 300


def test_acronym():
    assert acronym("m e".split(), "mahalakshmi engineering".split()) == 1
    assert acronym("ibm".split(), "international business machines".split()) == 1
    assert acronym("tata consultancy services".split(), "tcs".split()) == 1
    assert acronym("xy traders".split(), "alpha beta".split()) == 0
    assert acronym("valley".split(), "vlley".split()) == 0
    assert acronym([], []) == 0


def test_noise_row_values():
    a = _row(name_core="valley holdings", legal="inc", legal_first=0, nums="719",
             addr_alpha="mountain rd addison vt", name_norm="valley holdings inc",
             name_raw="Valley Holdings Inc", addr_norm="719 mountain rd addison vt", addr_raw="719 Mountain Road")
    b = _row(name_core="vlley holdings services", legal="inc", legal_first=1, nums="19",
             addr_alpha="mountain rd addison vt", name_norm="inc vlley holdings services",
             name_raw="Inc Vlley Holdings Services", addr_norm="19 mountain rd addison vt", addr_raw="19 Mountain Rd")
    vals = dict(zip(NOISE_FEATS, noise_row(a, b, {"services"})))
    assert vals["f_nz_num_digit_del"] == 1
    assert vals["f_nz_legal_moved"] == 1
    assert vals["f_nz_edit1_tokens"] == 1
    assert vals["f_nz_name_generic_only"] == 0  # 'vlley'/'valley' also differ
    assert vals["f_nz_num_conflict"] == 0  # digit deletion is noise, not a conflict
    assert vals["f_acronym"] == 0


def test_num_conflict():
    def conflict(x, y):
        return dict(zip(NOISE_FEATS, noise_row(_row(nums=x), _row(nums=y), set())))["f_nz_num_conflict"]
    assert conflict("719 12", "54 12") == 1
    assert conflict("719", "719") == 0
    assert conflict("719", "19") == 0
    assert conflict("719", "") == 0
