"""Tests for noise-channel features."""
from src.noise import NOISE_FEATS, digit_deletion, fit_generic_vocab, noise_row

ROW = ("name_core", "legal", "legal_first", "nums", "postcode", "addr_alpha",
       "name_norm", "name_raw", "addr_norm", "addr_raw")


def _row(**kw):
    base = dict(name_core="", legal="", legal_first=0, nums="", postcode="", addr_alpha="",
                name_norm="", name_raw="", addr_norm="", addr_raw="")
    base.update(kw)
    return tuple(base[k] for k in ROW)


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
