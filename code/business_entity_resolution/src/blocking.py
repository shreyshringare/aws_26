"""Candidate generation (blocking): IDF-weighted key matching via sparse matmul, per country label.

Country labels are an open set: we loop over whatever labels appear in S1 and only
compare records that share the label (train shows 0 cross-country matches).
"""
import hashlib

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from . import config
from .normalize import phonetic

FAMS = config.NAME_FAMS + config.ADDR_FAMS + config.CROSS_FAMS
FAM_IDX = {f: i for i, f in enumerate(FAMS)}
CAND_COLS = ["s1_rid", "pool_rid", "block_score", "block_rank"]
PAIR_TOKS = 6  # only the first tokens of a name form unordered pairs (n2 grows quadratically)


def name_keys(toks):
    """Name-side (family, key) pairs.

    n: tokens; nb: adjacent bigrams; nm: sorted initials; np: phonetic skeletons
    (transliteration variants); nc: tokens concatenated (domain-style names);
    n2: unordered token pairs (reorders and inserted words).
    """
    keys = {("n", t) for t in toks}
    keys |= {("nb", f"{a}_{b}") for a, b in zip(toks, toks[1:])}
    keys |= {("np", phonetic(t)) for t in toks if len(t) >= 2}
    head = sorted(set(toks[:PAIR_TOKS]))
    keys |= {("n2", f"{a}_{b}") for i, a in enumerate(head) for b in head[i + 1:]}
    if toks:
        keys.add(("nm", "".join(sorted(t[0] for t in toks))))
    joined = "".join(toks)
    if len(toks) >= 2 or len(joined) >= 6:
        keys.add(("nc", joined))
    return keys


def cross_keys(toks, addr_toks):
    """na: first name-core tokens x first distinct address alpha tokens (both len >= 2).

    A common name token or a common place token alone exceeds BLOCK_MAX_DF; their pair is selective.
    """
    names = [t for t in toks if len(t) >= 2][:config.NA_NAME_TOKS]
    places = list(dict.fromkeys(t for t in addr_toks if len(t) >= 2))[:config.NA_ADDR_TOKS]
    return {("na", f"{n}@{p}") for n in names for p in places}


def record_keys(name_core, addr_alpha, street_key):
    """Return the set of (family, key) blocking keys for one normalized record."""
    toks, addr_toks = name_core.split(), addr_alpha.split()
    keys = name_keys(toks) | cross_keys(toks, addr_toks)
    keys |= {("a", t) for t in addr_toks}
    if street_key:
        keys.add(("ab", street_key))
    return keys


def key_hash(fam, key):
    """Stable signed 64-bit hash of a family-prefixed key (process-independent)."""
    return int.from_bytes(hashlib.blake2b(f"{fam}:{key}".encode(), digest_size=8).digest(),
                          "little", signed=True)


def _hash_chunk(cols, offset):
    """Hash keys for one chunk of records; returns (row, hash, family) arrays."""
    rows, hs, fs = [], [], []
    for i, rec in enumerate(zip(*cols)):
        for fam, key in record_keys(*rec):
            rows.append(offset + i)
            hs.append(key_hash(fam, key))
            fs.append(FAM_IDX[fam])
    return np.asarray(rows, np.int64), np.asarray(hs, np.int64), np.asarray(fs, np.int8)


def hash_frame(df, n_jobs=1, chunk=200_000):
    """Hash all blocking keys of a normalized frame in parallel chunks."""
    cols = [df[c].tolist() for c in ("name_core", "addr_alpha", "street_key")]
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_hash_chunk)([c[i:i + chunk] for c in cols], i) for i in range(0, len(df), chunk)
    )
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int8)
    return tuple(np.concatenate(x) for x in zip(*parts))


def build_index(s1_hash, pool_hash, n1, n2, max_df, channel_weights):
    """Build one weighted S1 matrix (n1 x V) per channel and the binary pool-transposed matrix (V x n2).

    Keys with pool document frequency > max_df are dropped. A channel's weight for a key is
    its family weight (0 = family not in the channel) times the key's idf.
    """
    r1, h1, f1 = s1_hash
    r2, h2, _ = pool_hash
    vocab, inv2 = np.unique(h2, return_inverse=True)
    df = np.bincount(inv2, minlength=len(vocab))
    keep = df <= max_df
    idf = np.log(n2 / np.maximum(df, 1)).astype(np.float32) + 1.0
    m = keep[inv2]
    pt = sp.csr_matrix((np.ones(int(m.sum()), np.float32), (inv2[m], r2[m])), shape=(len(vocab), n2))
    pos = np.clip(np.searchsorted(vocab, h1), 0, max(len(vocab) - 1, 0))
    hit = (vocab[pos] == h1) & keep[pos] if len(vocab) else np.zeros(len(h1), bool)
    rows, cols, fams, base = r1[hit], pos[hit], f1[hit], idf[pos[hit]]
    mats = []
    for fw in channel_weights:
        w = base * fw[fams]
        on = w > 0
        s = sp.csr_matrix((w[on], (rows[on], cols[on])), shape=(n1, len(vocab)))
        s.sum_duplicates()
        mats.append(s)
    return mats, pt


def _topk_rows(m, k):
    """Top-k columns per row of a CSR score matrix: (rows, cols, rank within row)."""
    counts = np.diff(m.indptr)
    rows = np.repeat(np.arange(m.shape[0]), counts)
    order = np.lexsort((-m.data, rows))
    rank = np.arange(len(order)) - np.repeat(m.indptr[:-1], counts)
    keep = rank < k
    sel = order[keep]
    return rows[sel], m.indices[sel].astype(np.int64), rank[keep]


def _score_chunk(s_chunks, pt, ks, row0):
    """Score one S1 chunk in every channel; union the per-channel top-k pairs.

    Returns (rows, cols, min rank over channels, [channel score of each pair]).
    """
    ms = [(s @ pt).tocsr() for s in s_chunks]
    r, c, k = (np.concatenate(x) for x in zip(*(_topk_rows(m, kk) for m, kk in zip(ms, ks))))
    order = np.lexsort((k, c, r))  # per (row, col): lowest rank first
    first = np.ones(len(order), bool)
    first[1:] = (np.diff(r[order]) != 0) | (np.diff(c[order]) != 0)
    sel = order[first]
    r, c, k = r[sel], c[sel], k[sel]
    scores = [np.asarray(m[r, c]).ravel() if len(r) else np.zeros(0, np.float32) for m in ms]
    return row0 + r, c, k.astype(np.int16), scores


def score_column(channel):
    """Candidate column holding a channel's score ('all' is the historical block_score)."""
    return "block_score" if channel == "all" else f"score_{channel}"


def block_country(s1, pool, channels, max_df, weights, chunk, n_jobs):
    """Candidates for one country: union of per-channel top-k via chunked sparse products."""
    fws = [np.array([weights[f] * (f in fams) for f in FAMS], np.float32) for fams, _ in channels.values()]
    ks = [k for _, k in channels.values()]
    mats, pt = build_index(hash_frame(s1, n_jobs), hash_frame(pool, n_jobs), len(s1), len(pool), max_df, fws)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_score_chunk)([s[i:i + chunk] for s in mats], pt, ks, i) for i in range(0, len(s1), chunk)
    )
    r, c, k, scores = zip(*parts)
    r, c = np.concatenate(r), np.concatenate(c)
    out = {"s1_rid": s1["rid"].to_numpy()[r], "pool_rid": pool["rid"].to_numpy()[c],
           "block_rank": np.concatenate(k)}
    for j, ch in enumerate(channels):
        out[score_column(ch)] = np.concatenate([p[j] for p in scores]).astype(np.float32)
    return pd.DataFrame(out)[candidate_columns(channels)]


def candidate_columns(channels):
    """Ordered candidate-frame columns for a channel configuration."""
    return CAND_COLS + [score_column(ch) for ch in channels if ch != "all"]


def generate_candidates(s1n, pooln, s1_rids=None, topk=None, max_df=None, weights=None,
                        chunk=None, n_jobs=None, channels=None):
    """Block every country label present in S1 against same-label pool records.

    channels = {name: (families, k)}; each channel scores pairs with only its families'
    weights, keeps its top-k per S1 row, and the candidate set is the union. It must contain
    "all" (its score is block_score); block_rank is the best rank over channels. Passing
    `topk` without `channels` gives the single-channel blocker {"all": (FAMS, topk)}.
    """
    if channels is None:
        channels = {"all": (FAMS, topk)} if topk else config.BLOCK_CHANNELS
    assert "all" in channels, "channels must include 'all' (it defines block_score)"
    max_df = max_df or config.BLOCK_MAX_DF
    weights = weights or config.BLOCK_W
    chunk = chunk or config.BLOCK_CHUNK
    n_jobs = n_jobs or config.N_JOBS
    if s1_rids is not None:
        s1n = s1n[s1n["rid"].isin(s1_rids)]
    out = []
    for c in pd.unique(s1n["country"]):
        a = s1n[s1n["country"] == c].reset_index(drop=True)
        b = pooln[pooln["country"] == c].reset_index(drop=True)
        if len(a) and len(b):
            out.append(block_country(a, b, channels, max_df, weights, chunk, n_jobs))
    if not out:
        return pd.DataFrame({k: pd.Series(dtype="int64") for k in candidate_columns(channels)})
    return pd.concat(out, ignore_index=True)
