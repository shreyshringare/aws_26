"""Data picking: which S1 entities to use, and which candidate rows to fit the models on.

Entities are sampled per (country label, match-count bin) so every stratum keeps its share
(EDA §G.1: bins 0 / 1-2 / 3-4 / 5+). Country is only a sampling group, never a feature.
Training rows keep every positive and every hard negative (good blocking rank); easy negatives
are subsampled and up-weighted by 1/rate so the fitted probabilities stay unbiased.
"""
import numpy as np

from . import config

MATCH_BIN_EDGES = (1, 3, 5)  # -> bins 0, 1-2, 3-4, 5+


def match_bin(n_matches):
    """Match-count bin of each entity: 0 (none), 1 (1-2), 2 (3-4), 3 (5+)."""
    return np.searchsorted(MATCH_BIN_EDGES, np.asarray(n_matches), side="right")


def stratified_s1(s1n, n_matches, frac, seed=None):
    """Sorted S1 rids sampled with share `frac` inside every (country, match bin) stratum; frac >= 1 = all.

    n_matches[i] = number of true pool matches of the S1 row at position i.
    """
    if frac >= 1:
        return s1n["rid"].to_numpy()
    seed = config.SEED if seed is None else seed
    strata = s1n[["rid", "country"]].assign(bin=match_bin(n_matches))
    rids = strata.groupby(["country", "bin"]).sample(frac=frac, random_state=seed)["rid"].to_numpy()
    return np.sort(rids)


def training_rows(block_rank, y, keep_rank=None, easy_rate=None, seed=None):
    """Rows to fit on and their sample weights (0 for rows left out).

    Keeps all positives and all negatives with block_rank < keep_rank; other negatives are kept
    with probability easy_rate and weighted 1/easy_rate.
    """
    keep_rank = config.TRAIN_KEEP_RANK if keep_rank is None else keep_rank
    easy_rate = config.TRAIN_EASY_RATE if easy_rate is None else easy_rate
    seed = config.SEED if seed is None else seed
    block_rank, y = np.asarray(block_rank), np.asarray(y)
    if easy_rate >= 1:
        return np.ones(len(y), bool), np.ones(len(y), np.float32)
    hard = (y == 1) | (block_rank < keep_rank)
    easy = ~hard & (np.random.default_rng(seed).random(len(y)) < easy_rate)
    w = np.where(easy, np.float32(1.0 / easy_rate), np.float32(1.0)).astype(np.float32)
    mask = hard | easy
    w[~mask] = 0.0
    return mask, w
