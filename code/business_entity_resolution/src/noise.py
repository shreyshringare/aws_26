"""Noise-channel features (differentiator #2): explain *why* two records differ.

Each feature detects one perturbation operator observed in the training pairs
(house-number digit deletion, generic word add/drop, legal suffix moved, one-edit
typos, accent-only or abbreviation-only differences, trade-name swaps, acronyms), plus
one contradiction the operators cannot explain (two different house numbers).
Row tuples follow ROW_COLS order.
"""
from collections import Counter

from rapidfuzz.distance import DamerauLevenshtein

ROW_COLS = ["name_core", "legal", "legal_first", "nums", "addr_alpha",
            "name_norm", "name_raw", "addr_norm", "addr_raw"]
CORE, LEGAL, LEGAL_FIRST, NUMS, ADDR_ALPHA, NAME_NORM, NAME_RAW, ADDR_NORM, ADDR_RAW = range(len(ROW_COLS))

NOISE_FEATS = [
    "f_nz_num_digit_del", "f_nz_num_missing_one", "f_nz_name_generic_only", "f_nz_name_core_equal",
    "f_nz_legal_moved", "f_nz_edit1_tokens", "f_nz_addr_only", "f_nz_accent_only", "f_nz_addr_norm_only",
    "f_nz_num_conflict", "f_acronym",
]


def digit_deletion(a, b):
    """1 if the shorter number string equals the longer with exactly one digit removed."""
    if abs(len(a) - len(b)) != 1:
        return 0
    long_, short = (a, b) if len(a) > len(b) else (b, a)
    return int(any(long_[:i] + long_[i + 1:] == short for i in range(len(long_))))


def acronym(ta, tb):
    """1 if one name's initials (>= 2 tokens) equal a token of the other or its single letters joined.

    Covers "ibm" ~ "international business machines" and "m e" ~ "mahalakshmi engineering".
    """
    for x, y in ((ta, tb), (tb, ta)):
        if len(x) >= 2:
            ini = "".join(t[0] for t in x)
            if ini in y or ini == "".join(t for t in y if len(t) == 1):
                return 1
    return 0


def fit_generic_vocab(s1_cores, pool_cores, min_count=300, min_share=0.5, min_len=3):
    """Learn 'generic add-on' tokens: tokens that mostly occur in the symmetric difference of matched names.

    Tokens shorter than `min_len` (initials, transliteration fragments) are never generic.
    """
    diff, tot = Counter(), Counter()
    for a, b in zip(s1_cores, pool_cores):
        sa, sb = set(a.split()), set(b.split())
        tot.update(sa | sb)
        diff.update(sa ^ sb)
    return {t for t, n in diff.items() if len(t) >= min_len and n >= min_count and n / tot[t] >= min_share}


def _jac(a, b):
    """Jaccard similarity of two sets (0 when both empty)."""
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _edit1(ca, cb):
    """Count tokens of ca-cb (len >= 3) within Damerau-Levenshtein distance 1 of a token in cb-ca."""
    only_b = [u for u in cb - ca if len(u) >= 3]
    return sum(1 for t in ca - cb if len(t) >= 3
               and any(DamerauLevenshtein.distance(t, u, score_cutoff=1) <= 1 for u in only_b))


def noise_row(a, b, generic):
    """Noise-channel feature values for one pair of ROW_COLS tuples (order = NOISE_FEATS)."""
    ta, tb = a[CORE].split(), b[CORE].split()
    ca, cb = set(ta), set(tb)
    na, nb = a[NUMS].split(), b[NUMS].split()
    diff = ca ^ cb
    ddel = digit_deletion(na[0], nb[0]) if na and nb else 0
    return [
        ddel,
        int(bool(na) != bool(nb)),
        int(bool(diff) and diff <= generic),
        int(bool(ca) and ca == cb),
        int(bool(a[LEGAL]) and a[LEGAL] == b[LEGAL] and a[LEGAL_FIRST] != b[LEGAL_FIRST]),
        _edit1(ca, cb),
        int(not (ca & cb) and _jac(set(a[ADDR_ALPHA].split()), set(b[ADDR_ALPHA].split())) >= 0.6),
        int(a[NAME_NORM] == b[NAME_NORM] and a[NAME_RAW].lower() != b[NAME_RAW].lower()),
        int(bool(a[ADDR_NORM]) and a[ADDR_NORM] == b[ADDR_NORM] and a[ADDR_RAW].lower() != b[ADDR_RAW].lower()),
        int(bool(na) and bool(nb) and na[0] != nb[0] and not ddel),
        acronym(ta, tb),
    ]
