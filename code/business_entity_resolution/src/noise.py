"""Noise-channel features (differentiator #2): explain *why* two records differ.

Each feature detects one perturbation operator observed in the training pairs
(house-number digit deletion, generic word add/drop, legal suffix moved, one-edit
typos, accent-only or abbreviation-only differences, trade-name swaps).
Row tuples follow ROW_COLS order (see features.ROW_COLS).
"""
from collections import Counter

from rapidfuzz.distance import DamerauLevenshtein

NOISE_FEATS = [
    "f_nz_num_digit_del", "f_nz_num_missing_one", "f_nz_name_generic_only", "f_nz_name_core_equal",
    "f_nz_legal_moved", "f_nz_edit1_tokens", "f_nz_addr_only", "f_nz_accent_only", "f_nz_addr_norm_only",
]


def digit_deletion(a, b):
    """1 if the shorter number string equals the longer with exactly one digit removed."""
    if abs(len(a) - len(b)) != 1:
        return 0
    long_, short = (a, b) if len(a) > len(b) else (b, a)
    return int(any(long_[:i] + long_[i + 1:] == short for i in range(len(long_))))


def fit_generic_vocab(s1_cores, pool_cores, min_count=100, min_share=0.5):
    """Learn 'generic add-on' tokens: tokens that mostly occur in the symmetric difference of matched names."""
    diff, tot = Counter(), Counter()
    for a, b in zip(s1_cores, pool_cores):
        sa, sb = set(a.split()), set(b.split())
        tot.update(sa | sb)
        diff.update(sa ^ sb)
    return {t for t, n in diff.items() if n >= min_count and n / tot[t] >= min_share}


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
    ca, cb = set(a[0].split()), set(b[0].split())
    na, nb = a[3].split(), b[3].split()
    diff = ca ^ cb
    return [
        digit_deletion(na[0], nb[0]) if na and nb else 0,
        int(bool(na) != bool(nb)),
        int(bool(diff) and diff <= generic),
        int(bool(ca) and ca == cb),
        int(bool(a[1]) and a[1] == b[1] and a[2] != b[2]),
        _edit1(ca, cb),
        int(not (ca & cb) and _jac(set(a[5].split()), set(b[5].split())) >= 0.6),
        int(a[6] == b[6] and a[7].lower() != b[7].lower()),
        int(bool(a[8]) and a[8] == b[8] and a[9].lower() != b[9].lower()),
    ]
