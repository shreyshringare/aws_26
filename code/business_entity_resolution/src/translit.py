"""Learned native-script -> Latin token table, fitted on train matched name pairs only.

No external transliteration data is used: the table is estimated from positionally
aligned tokens of (S1 Latin name, pool native-script name) ground-truth pairs.
"""
from collections import Counter, defaultdict

from .normalize import normalize_name, split_raw


def fit_table(s1_names, pool_names, min_support=3, min_share=0.6):
    """Fit {native token: latin token} from matched name pairs with equal token counts."""
    counts = defaultdict(Counter)
    for a, b in zip(s1_names, pool_names):
        if not b or b.isascii():
            continue
        ta = normalize_name(a)[0].split()
        tb = split_raw(b)
        if len(ta) != len(tb):
            continue
        for native, latin in zip(tb, ta):
            if not native.isascii():
                counts[native][latin] += 1
    table = {}
    for tok, c in counts.items():
        best, n = c.most_common(1)[0]
        if n >= min_support and n / sum(c.values()) >= min_share:
            table[tok] = best
    return table


def save_table(table, path):
    """Write the table as a two-column UTF-8 TSV (native, latin)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k in sorted(table):
            f.write(f"{k}\t{table[k]}\n")


def load_table(path):
    """Read a table written by `save_table`."""
    with open(path, encoding="utf-8") as f:
        return dict(line.rstrip("\n").split("\t", 1) for line in f if line.strip())
