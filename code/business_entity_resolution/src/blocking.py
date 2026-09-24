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

FAMS = ("n", "nb", "nm", "a", "ab", "p")
FAM_IDX = {f: i for i, f in enumerate(FAMS)}
CAND_COLS = ["s1_rid", "pool_rid", "block_score", "block_rank"]


def record_keys(name_core, addr_alpha, street_key, postcode):
    """Return the set of (family, key) blocking keys for one normalized record."""
    toks = name_core.split()
    keys = {("n", t) for t in toks}
    keys |= {("nb", f"{a}_{b}") for a, b in zip(toks, toks[1:])}
    if toks:
        keys.add(("nm", "".join(sorted(t[0] for t in toks))))
    keys |= {("a", t) for t in addr_alpha.split()}
    if street_key:
        keys.add(("ab", street_key))
    if postcode:
        keys.add(("p", postcode))
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
    cols = [df[c].tolist() for c in ("name_core", "addr_alpha", "street_key", "postcode")]
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_hash_chunk)([c[i:i + chunk] for c in cols], i) for i in range(0, len(df), chunk)
    )
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int8)
    return tuple(np.concatenate(x) for x in zip(*parts))


def build_index(s1_hash, pool_hash, n1, n2, max_df, weights):
    """Build weighted S1 matrix (n1 x V) and binary pool-transposed matrix (V x n2).

    Keys with pool document frequency > max_df are dropped; weights = family weight * idf.
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
    fw = np.array([weights[f] for f in FAMS], np.float32)
    s = sp.csr_matrix((idf[pos[hit]] * fw[f1[hit]], (r1[hit], pos[hit])), shape=(n1, len(vocab)))
    s.sum_duplicates()
    return s, pt


def _topk(s_chunk, pt, topk, row0):
    """Score one S1 chunk against the pool and keep the top-k pool rows per S1 row."""
    m = (s_chunk @ pt).tocsr()
    counts = np.diff(m.indptr)
    rows = np.repeat(np.arange(m.shape[0]), counts)
    order = np.lexsort((-m.data, rows))
    rank = np.arange(len(order)) - np.repeat(m.indptr[:-1], counts)
    keep = rank < topk
    sel = order[keep]
    return row0 + rows[sel], m.indices[sel].astype(np.int64), m.data[sel], rank[keep].astype(np.int16)


def block_country(s1, pool, topk, max_df, weights, chunk, n_jobs):
    """Candidates for one country: S1 rows x pool rows via chunked sparse products."""
    s, pt = build_index(hash_frame(s1, n_jobs), hash_frame(pool, n_jobs), len(s1), len(pool), max_df, weights)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_topk)(s[i:i + chunk], pt, topk, i) for i in range(0, len(s1), chunk)
    )
    r, c, sc, k = (np.concatenate(x) for x in zip(*parts))
    return pd.DataFrame({
        "s1_rid": s1["rid"].to_numpy()[r],
        "pool_rid": pool["rid"].to_numpy()[c],
        "block_score": sc.astype(np.float32),
        "block_rank": k,
    })


def generate_candidates(s1n, pooln, s1_rids=None, topk=None, max_df=None, weights=None,
                        chunk=None, n_jobs=None):
    """Block every country label present in S1 against same-label pool records."""
    topk = topk or config.BLOCK_TOPK
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
            out.append(block_country(a, b, topk, max_df, weights, chunk, n_jobs))
    if not out:
        return pd.DataFrame({k: pd.Series(dtype="int64") for k in CAND_COLS})
    return pd.concat(out, ignore_index=True)
