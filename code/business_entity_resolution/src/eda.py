"""Quantified, reproducible EDA (v2): numbers that justify the modelling decisions.

Run from `code/business_entity_resolution/`:

    python -m src.eda --data ../../dataset --out ../../docs/EDA.md [--pairs 200000] [--seed 42]

Uses the cached normalized train parquet files (`cache/v1_norm_train_source{1,2,3}.parquet`)
and normalizes only *samples* of the test sources (with the learned transliteration table),
so peak memory stays a few GB. Every analysis is a pure function returning a small DataFrame;
`main` assembles them into a markdown report. Country labels are treated as an open set:
all groupings loop over whatever labels are present (France appears only in test).
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import blocking
from .io_utils import load_tsv
from .noise import _edit1, digit_deletion, fit_generic_vocab
from .normalize import normalize_frame

CACHE = Path(__file__).resolve().parents[1] / "cache"
NON_LATIN = re.compile(r"[^\x00-ɏ]")
NORM_COLS = ["entity_id", "country", "name_norm", "name_core", "legal", "legal_first", "addr_norm",
             "addr_alpha", "nums", "postcode", "street_key", "landmark", "name_raw", "script_flag"]
QS = (0.1, 0.5, 0.9)


# ----------------------------------------------------------------------------- formatting
def md_table(df, digits=3):
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency).

    Floats are rounded to `digits`; the index is included when it is not a RangeIndex.
    """
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def fmt(v):
        """Format one cell."""
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:,.{digits}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- loading
def sample_parquet(path, cols, n=None, seed=42):
    """Read `cols` of a parquet file and return a random sample of `n` rows (all rows if n is None)."""
    t = pq.read_table(path, columns=cols)
    if n is not None and n < t.num_rows:
        idx = np.sort(np.random.default_rng(seed).choice(t.num_rows, n, replace=False))
        t = t.take(pa.array(idx))
    return t.to_pandas()


def read_parquet_ids(path, cols, ids):
    """Read `cols` of the parquet rows whose entity_id is in `ids` (arrow-side filtering)."""
    t = pq.read_table(path, columns=cols)
    t = t.filter(pc.is_in(t["entity_id"], value_set=pa.array(list(ids))))
    return t.to_pandas()


def count_countries_tsv(path):
    """Row count per country label of a raw TSV (reads only the country column)."""
    t = pacsv.read_csv(
        str(path), parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(include_columns=["country"], strings_can_be_null=False))
    return t.column("country").to_pandas().value_counts()


def sample_raw_tsv(path, n, seed=42):
    """Random sample of `n` rows of a raw TSV (full file read in Arrow, then subsampled)."""
    df = load_tsv(path)
    if n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- A. volumes & fields
def ntok(s):
    """Whitespace token count per string of a Series."""
    return s.str.split().str.len().fillna(0).astype(int)


def profile(df):
    """One-row profile of a normalized frame: size, script / empty-address shares, length quantiles."""
    nl, al = ntok(df["name_norm"]), ntok(df["addr_norm"])
    out = {"rows": len(df),
           "non_latin": float(df["script_flag"].astype(int).mean()),
           "name_non_latin": float(df["name_raw"].str.contains(NON_LATIN).mean()),
           "addr_empty": float((df["addr_norm"] == "").mean()),
           "legal_found": float((df["legal"] != "").mean()),
           "postcode_found": float((df["postcode"] != "").mean()),
           "num_found": float((df["nums"] != "").mean()),
           "landmark_found": float((df["landmark"] != "").mean())}
    for q in QS:
        out[f"name_p{int(q * 100)}"] = float(nl.quantile(q))
    for q in QS:
        out[f"addr_p{int(q * 100)}"] = float(al.quantile(q))
    return out


def profile_by(df, keys):
    """`profile` for every group of `keys` (open label set), as a DataFrame indexed by the keys."""
    rows = []
    for k, g in df.groupby(keys, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, k)), **profile(g)})
    return pd.DataFrame(rows).set_index(keys)


# ----------------------------------------------------------------------------- B. match structure
def explode_gt(gt):
    """Explode a raw ground-truth frame into positive pairs (s1_id, cand_id)."""
    g = gt[gt["matched_entity_ids"] != ""]
    ids = g["matched_entity_ids"].str.split(",")
    p = pd.DataFrame({"s1_id": np.repeat(g["source1_entity_id"].to_numpy(), ids.str.len()),
                      "cand_id": np.concatenate(ids.to_numpy()) if len(g) else np.array([], object)})
    return p


def cardinality(gt, max_k=11):
    """Per-S1 counts: total, S2 and S3 matches; returns frame with n, n_s2, n_s3 per S1 id."""
    m = gt["matched_entity_ids"]
    n = np.where(m == "", 0, m.str.count(",") + 1)
    return pd.DataFrame({"s1_id": gt["source1_entity_id"].to_numpy(), "n": np.minimum(n, max_k),
                         "n_s2": m.str.count("S2-").to_numpy(), "n_s3": m.str.count("S3-").to_numpy()})


def cardinality_hist(card, country, max_k=11):
    """Share of S1 entities with k = 0..max_k matches (total, S2-only, S3-only) plus per-country singleton rate."""
    hist = pd.DataFrame({
        "k": range(max_k + 1),
        "share_total": np.bincount(card["n"], minlength=max_k + 1)[:max_k + 1] / len(card),
        "share_s2": np.bincount(np.minimum(card["n_s2"], max_k), minlength=max_k + 1)[:max_k + 1] / len(card),
        "share_s3": np.bincount(np.minimum(card["n_s3"], max_k), minlength=max_k + 1)[:max_k + 1] / len(card),
    })
    c = card.assign(country=country.to_numpy())
    by = c.groupby("country").agg(s1=("n", "size"), singleton=("n", lambda x: float((x == 0).mean())),
                                  mean_matches=("n", "mean"), mean_s2=("n_s2", "mean"),
                                  mean_s3=("n_s3", "mean"),
                                  p90_matches=("n", lambda x: float(np.quantile(x, 0.9))))
    return hist, by


def one_to_one(pairs):
    """Count pool ids linked to >1 S1 entity (0 means the one-to-one constraint is exact)."""
    vc = pairs["cand_id"].value_counts()
    return pd.DataFrame([{"matched_pool_ids": len(vc), "ids_in_2plus_s1": int((vc > 1).sum()),
                          "max_s1_per_id": int(vc.max()) if len(vc) else 0}])


def pool_unmatched(pool, matched_ids):
    """Share of pool records (by source prefix and country) that match no S1 entity."""
    p = pool.assign(src=pool["entity_id"].str[:2],
                    unmatched=~pool["entity_id"].isin(matched_ids))
    return p.groupby(["src", "country"]).agg(rows=("unmatched", "size"),
                                             unmatched=("unmatched", "mean"))


# ----------------------------------------------------------------------------- C. noise operators
def _num0(s):
    """First whitespace token of a nums string ('' when absent)."""
    return s.split()[0] if s else ""


def jaccard(a, b):
    """Jaccard of two token strings (NaN when both are empty)."""
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else np.nan


def noise_flags(a, b, generic):
    """Boolean noise-operator flags for aligned pair sides `a` (S1) and `b` (pool).

    Both frames carry the normalized columns (NORM_COLS); rows are aligned by position.
    Returns one row per pair with a boolean column per operator plus `addr_jac`.
    """
    rows = []
    for ra, rb in zip(a.itertuples(index=False), b.itertuples(index=False)):
        ca, cb = set(ra.name_core.split()), set(rb.name_core.split())
        diff = ca ^ cb
        na, nb = _num0(ra.nums), _num0(rb.nums)
        la, lb = ra.legal, rb.legal
        rows.append((
            ra.name_norm == rb.name_norm,
            ra.name_core == rb.name_core,
            bool(ca) and ca == cb and ra.name_core != rb.name_core,
            bool(la) and bool(lb) and la != lb,
            bool(la) != bool(lb),
            bool(la) and la == lb and int(ra.legal_first) != int(rb.legal_first),
            bool(diff) and diff <= generic,
            _edit1(ca, cb) >= 1,
            bool(ca) and bool(cb) and not (ca & cb),
            bool(NON_LATIN.search(rb.name_raw)),
            ra.addr_norm == rb.addr_norm and ra.addr_norm != "",
            ra.addr_norm == "" or rb.addr_norm == "",
            bool(na) and na == nb,
            bool(na) and bool(nb) and digit_deletion(na, nb) == 1,
            bool(na) != bool(nb),
            bool(na) and bool(nb) and na != nb and digit_deletion(na, nb) == 0,
            bool(ra.street_key) and ra.street_key == rb.street_key,
            bool(ra.postcode) and ra.postcode == rb.postcode,
            bool(ra.postcode) != bool(rb.postcode) or not (ra.postcode or rb.postcode),
            bool(ra.postcode) and bool(rb.postcode) and ra.postcode != rb.postcode,
            bool(ra.landmark) or bool(rb.landmark),
            jaccard(ra.addr_alpha, rb.addr_alpha),
        ))
    cols = ["name_norm_eq", "name_core_eq", "token_set_eq_order_diff", "legal_differs",
            "legal_missing_one", "legal_moved", "diff_generic_only", "typo_edit1", "no_shared_core",
            "pool_name_non_latin", "addr_norm_eq", "addr_empty_either", "num_eq", "num_digit_del",
            "num_missing_one", "num_other", "street_key_eq", "postcode_eq", "postcode_missing",
            "postcode_differs", "landmark_either", "addr_jac"]
    return pd.DataFrame(rows, columns=cols)


def rate_table(flags, country):
    """Percent of pairs flagged by each operator, overall and per country (columns)."""
    f = flags.drop(columns=["addr_jac"]).astype(float) * 100
    out = {"ALL": f.mean()}
    for c in sorted(pd.unique(country)):
        out[c] = f[np.asarray(country) == c].mean()
    t = pd.DataFrame(out)
    t.index.name = "operator"
    return t


def jaccard_quantiles(flags, country):
    """Quantiles of address alpha-token Jaccard (non-empty pairs), overall and per country."""
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    j = flags["addr_jac"]
    out = {"ALL": j.quantile(qs)}
    for c in sorted(pd.unique(country)):
        out[c] = j[np.asarray(country) == c].quantile(qs)
    t = pd.DataFrame(out)
    t.index = [f"p{int(q * 100)}" for q in qs]
    t.index.name = "quantile"
    return t


# ----------------------------------------------------------------------------- D. ambiguity
def key_hash(country, core):
    """uint64 hash of (country, name_core) pairs, vectorized."""
    s = pd.Series(np.asarray(country, dtype=object)) + "\x1f" + pd.Series(np.asarray(core, dtype=object))
    return pd.util.hash_array(s.to_numpy(dtype=object))


def lookup_counts(vc, keys):
    """Look up counts of `keys` in a value_counts Series indexed by uint64 hash (0 when absent)."""
    return vc.reindex(keys).fillna(0).to_numpy().astype(np.int64)


def s1_name_sharing(s1_country, s1_key, s1_core_empty):
    """Share of S1 entities (non-empty core) whose (country, name_core) is shared with >=1 other S1."""
    vc = pd.Series(s1_key).value_counts()
    n = lookup_counts(vc, s1_key)
    d = pd.DataFrame({"country": s1_country, "shared": n >= 2, "n": n})[~np.asarray(s1_core_empty)]
    return d.groupby("country").agg(s1=("shared", "size"), share_name_shared=("shared", "mean"),
                                    p99_same_name=("n", lambda x: float(np.quantile(x, .99))),
                                    max_same_name=("n", "max")), vc


def collision_negatives(sample_key, sample_country, matched_same_key, pool_vc):
    """Per sampled S1: pool records with the exact same (country, name_core) that are NOT matches."""
    neg = lookup_counts(pool_vc, sample_key) - np.asarray(matched_same_key)
    d = pd.DataFrame({"country": sample_country, "neg": np.maximum(neg, 0)})
    agg = dict(s1_sampled=("neg", "size"), share_with_neg=("neg", lambda x: float((x > 0).mean())),
               mean_neg=("neg", "mean"), p90_neg=("neg", lambda x: float(np.quantile(x, .9))),
               p99_neg=("neg", lambda x: float(np.quantile(x, .99))), max_neg=("neg", "max"))
    return pd.concat([d.assign(country="ALL"), d]).groupby("country").agg(**agg)


# ----------------------------------------------------------------------------- E. shift
def arrow_vocab(paths, col, country=None):
    """Set of whitespace tokens of `col` over parquet files (optionally one country label only)."""
    vocab = set()
    for p in paths:
        t = pq.read_table(p, columns=[col, "country"])
        if country is not None:
            t = t.filter(pc.equal(t["country"], country))
        u = pc.unique(pc.list_flatten(pc.utf8_split_whitespace(t[col])))
        vocab.update(u.to_pylist())
        del t
    vocab.discard("")
    return vocab


def oov_rates(texts, vocab):
    """(token OOV share, record share with >=1 OOV token) of token strings against `vocab`."""
    tot = oov = rec = nrec = 0
    for s in texts:
        toks = s.split()
        if not toks:
            continue
        k = sum(t not in vocab for t in toks)
        tot += len(toks)
        oov += k
        rec += k > 0
        nrec += 1
    return (oov / tot if tot else np.nan), (rec / nrec if nrec else np.nan)


def top_tokens(texts, vocab, n=15):
    """Most frequent tokens with an in-train-vocab flag (used to spot dictionary gaps)."""
    vc = pd.Series([t for s in texts for t in s.split()]).value_counts().head(n)
    return pd.DataFrame({"token": vc.index, "count": vc.to_numpy(),
                         "in_train_vocab": [t in vocab for t in vc.index]})


# ----------------------------------------------------------------------------- F. blocking keys
def key_df_stats(frame, frac, max_df=2000):
    """Per blocking-key family: document-frequency quantiles over a pool sample, scaled by 1/frac.

    Document frequency is counted per country label (as in blocking). Returns the stats table
    and {country: (sorted hash vocab, scaled df)} for later lookups.
    """
    per_fam, lookup = {f: [] for f in blocking.FAMS}, {}
    for c, g in frame.groupby("country"):
        _, h, f = blocking.hash_frame(g.reset_index(drop=True), n_jobs=1)
        vocab, first, inv = np.unique(h, return_index=True, return_inverse=True)
        df = np.bincount(inv.ravel(), minlength=len(vocab)) / frac
        lookup[c] = (vocab, df)
        fam_of_key = f[first]
        for i, fam in enumerate(blocking.FAMS):
            per_fam[fam].append(df[fam_of_key == i])
    rows = []
    for fam in blocking.FAMS:
        d = np.concatenate(per_fam[fam]) if per_fam[fam] else np.zeros(0)
        if not len(d):
            continue
        big = d > max_df
        rows.append({"family": fam, "keys_in_sample": len(d),
                     "df_p50": float(np.quantile(d, .5)), "df_p90": float(np.quantile(d, .9)),
                     "df_p99": float(np.quantile(d, .99)), "df_max": float(d.max()),
                     "share_keys_df_gt_max": float(big.mean()),
                     "share_postings_dropped": float((d[big]).sum() / d.sum())})
    return pd.DataFrame(rows).set_index("family"), lookup


def pair_key_sharing(a, b, country, lookup=None, max_df=2000):
    """Share of matched pairs sharing >=1 blocking key of each family (and >=1 key with df <= max_df)."""
    fams = blocking.FAMS
    rows = []
    for ra, rb, c in zip(a.itertuples(index=False), b.itertuples(index=False), country):
        ka = blocking.record_keys(ra.name_core, ra.addr_alpha, ra.street_key)
        kb = blocking.record_keys(rb.name_core, rb.addr_alpha, rb.street_key)
        shared = ka & kb
        r = {f: False for f in fams}
        usable = {f: False for f in fams}
        vocab, df = lookup.get(c, (np.zeros(0, np.int64), np.zeros(0))) if lookup else (None, None)
        for fam, key in shared:
            r[fam] = True
            if lookup is not None and not usable[fam]:
                h = blocking.key_hash(fam, key)
                i = np.searchsorted(vocab, h)
                d = df[i] if i < len(vocab) and vocab[i] == h else 0.0
                usable[fam] = d <= max_df
        rows.append([r[f] for f in fams] + [any(r.values())] + [usable[f] for f in fams]
                    + [any(usable.values())])
    cols = list(fams) + ["any"] + [f"{f}_usable" for f in fams] + ["any_usable"]
    d = pd.DataFrame(rows, columns=cols)
    only = {f: float((d[f] & ~d[[g for g in fams if g != f]].any(axis=1)).mean()) for f in fams}
    only_u = {f: float((d[f"{f}_usable"] & ~d[[f"{g}_usable" for g in fams if g != f]].any(axis=1)).mean())
              for f in fams}
    out = pd.DataFrame({"share_pairs_sharing": d[list(fams) + ["any"]].mean().to_numpy(),
                        "share_sharing_usable": d[[f"{f}_usable" for f in fams] + ["any_usable"]].mean().to_numpy(),
                        "only_this_family_usable": [only_u[f] for f in fams] + [np.nan],
                        "only_this_family": [only[f] for f in fams] + [np.nan]},
                       index=pd.Index(list(fams) + ["any"], name="family"))
    return out


# ----------------------------------------------------------------------------- main
def log(msg, t0):
    """Print a timestamped progress line."""
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def peak_mem_mb():
    """Peak resident memory of this process in MB (psutil / Windows API when available, else NaN)."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2 ** 20
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            """PROCESS_MEMORY_COUNTERS (Win32)."""
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wintypes.DWORD]
        fn(ctypes.c_void_p(-1), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2 ** 20
    except Exception:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return float("nan")


def main(argv=None):
    """Build the EDA v2 markdown report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="../../dataset")
    ap.add_argument("--out", default="../../docs/EDA.md")
    ap.add_argument("--pairs", type=int, default=200_000)
    ap.add_argument("--sample", type=int, default=200_000, help="rows per source for profiles")
    ap.add_argument("--s1-sample", type=int, default=50_000, help="S1 entities for collision negatives")
    ap.add_argument("--key-frac", type=float, default=0.1, help="pool fraction for key df stats")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args(argv)
    t0 = time.time()
    data = Path(args.data)
    paths = {s: CACHE / f"v1_norm_train_source{s}.parquet" for s in (1, 2, 3)}
    rep = [f"# EDA v2 — quantified (train + test samples)\n",
           f"Generated by `python -m src.eda --data {args.data} --out {args.out} --pairs {args.pairs} "
           f"--seed {args.seed}` (sample={args.sample:,} rows/source, key-frac={args.key_frac}). "
           "All shares are fractions (0–1) unless the column says %. Country labels are an open set; "
           "France exists only in test.\n"]

    # ---------------- A
    log("A: volumes", t0)
    vol = []
    for s in (1, 2, 3):
        vc = pq.read_table(paths[s], columns=["country"])["country"].to_pandas().value_counts()
        vol += [{"split": "train", "source": f"S{s}", "country": k, "rows": int(v)} for k, v in vc.items()]
        vc = count_countries_tsv(data / "test" / f"test_source{s}.tsv")
        vol += [{"split": "test", "source": f"S{s}", "country": k, "rows": int(v)} for k, v in vc.items()]
    vol = pd.DataFrame(vol)
    vol["share_of_split_source"] = vol["rows"] / vol.groupby(["split", "source"])["rows"].transform("sum")

    log("A: samples (train parquet + normalized test samples)", t0)
    from .translit import load_table
    table = load_table(CACHE / "v1_translit.tsv")
    samp = []
    for s in (1, 2, 3):
        tr = sample_parquet(paths[s], NORM_COLS, args.sample, args.seed + s)
        samp.append(tr.assign(split="train", source=f"S{s}"))
        raw = sample_raw_tsv(data / "test" / f"test_source{s}.tsv", args.sample, args.seed + s)
        te = normalize_frame(raw, table, n_jobs=args.n_jobs)
        del raw
        samp.append(te[[c for c in NORM_COLS]].assign(split="test", source=f"S{s}"))
        log(f"  sampled source {s}", t0)
    samp = pd.concat(samp, ignore_index=True)
    prof = profile_by(samp, ["split", "source", "country"])

    rep += ["## A. Volumes & fields\n", "### A1. Rows per split × source × country\n",
            md_table(vol.set_index(["split", "source", "country"])), "",
            f"### A2. Field profile per split × source × country (sample {args.sample:,} rows/source)\n",
            "`non_latin` = any char beyond Latin Extended-B in name+address; `addr_empty` = normalized "
            "address empty; `legal_found` = a legal suffix recognised by our dictionary; name/addr_pXX = "
            "token-count quantiles of name_norm / addr_norm.\n",
            md_table(prof), ""]

    # ---------------- B
    log("B: ground truth", t0)
    gt = load_tsv(data / "train" / "train_ground_truth.tsv")
    s1c = pq.read_table(paths[1], columns=["entity_id", "country"]).to_pandas()
    ctry = gt["source1_entity_id"].map(pd.Series(s1c["country"].to_numpy(), index=s1c["entity_id"]))
    card = cardinality(gt)
    hist, by = cardinality_hist(card, ctry)
    pairs = explode_gt(gt)
    oto = one_to_one(pairs)
    pool_ids = pd.concat([pq.read_table(paths[s], columns=["entity_id", "country"]).to_pandas()
                          for s in (2, 3)], ignore_index=True)
    unm = pool_unmatched(pool_ids, pd.Index(pairs["cand_id"].unique()))
    del pool_ids
    rep += ["## B. Match structure (train)\n",
            "### B1. Matches per S1 entity (k capped at 11)\n", md_table(hist, 4), "",
            "### B2. Per country\n", md_table(by), "",
            "### B3. One-to-one check\n", md_table(oto), "",
            "### B4. Pool records matching no S1 entity\n", md_table(unm), ""]

    # ---------------- C
    log("C: matched-pair sample", t0)
    rng = np.random.default_rng(args.seed)
    ps = pairs.iloc[np.sort(rng.choice(len(pairs), min(args.pairs, len(pairs)), replace=False))]
    ps = ps.reset_index(drop=True)
    s1s = read_parquet_ids(paths[1], NORM_COLS, set(ps["s1_id"]))
    pls = pd.concat([read_parquet_ids(paths[s], NORM_COLS, set(ps["cand_id"])) for s in (2, 3)],
                    ignore_index=True)
    a = ps[["s1_id"]].merge(s1s, left_on="s1_id", right_on="entity_id", how="left")
    b = ps[["cand_id"]].merge(pls, left_on="cand_id", right_on="entity_id", how="left")
    ok = a["entity_id"].notna().to_numpy() & b["entity_id"].notna().to_numpy()
    a, b = a[ok].reset_index(drop=True), b[ok].reset_index(drop=True)
    pc_country = a["country"].to_numpy()
    generic = fit_generic_vocab(a["name_core"], b["name_core"])
    log(f"  generic vocab size {len(generic)}; computing flags on {len(a):,} pairs", t0)
    flags = noise_flags(a, b, generic)
    rates = rate_table(flags, pc_country)
    b_src = b["entity_id"].str[:2].to_numpy()
    rates_src = pd.DataFrame({f"{c}": rate_table(flags[b_src == c], pc_country[b_src == c])["ALL"]
                              for c in ("S2", "S3")})
    rep += ["## C. Noise operators on matched pairs (train)\n",
            f"Random sample of {len(a):,} ground-truth pairs (S1 side vs pool side). Values are "
            f"**% of pairs**. Generic vocab fitted on this sample: {len(generic)} tokens "
            f"(e.g. {', '.join(sorted(generic)[:25])}).\n",
            md_table(rates.join(rates_src), 2), "",
            "### C2. Address alpha-token Jaccard quantiles (pairs with ≥1 non-empty side)\n",
            md_table(jaccard_quantiles(flags, pc_country)), ""]

    # ---------------- D
    log("D: ambiguity", t0)
    s1n = pq.read_table(paths[1], columns=["entity_id", "country", "name_core"]).to_pandas()
    s1_key = key_hash(s1n["country"], s1n["name_core"])
    share_tab, s1_vc = s1_name_sharing(s1n["country"].to_numpy(), s1_key,
                                       (s1n["name_core"] == "").to_numpy())
    # sample of S1 entities for collision negatives
    idx = np.sort(rng.choice(len(s1n), min(args.s1_sample, len(s1n)), replace=False))
    smp = s1n.iloc[idx][s1n["name_core"].iloc[idx] != ""].reset_index(drop=True)
    smp_key = key_hash(smp["country"], smp["name_core"])
    smp_pairs = pairs[pairs["s1_id"].isin(set(smp["entity_id"]))]
    need = set(smp_pairs["cand_id"])
    pool_counts, pool_rows, cand_keys = [], [], []
    for s in (2, 3):
        t = pq.read_table(paths[s], columns=["entity_id", "country", "name_core"]).to_pandas()
        k = key_hash(t["country"], t["name_core"])
        nonempty = (t["name_core"] != "").to_numpy()
        pool_counts.append(pd.Series(k[nonempty]).value_counts())
        n_s1 = lookup_counts(s1_vc, k)[nonempty]
        pool_rows.append(pd.DataFrame({"country": t["country"].to_numpy()[nonempty], "n_s1": n_s1}))
        m = t["entity_id"].isin(need).to_numpy()
        cand_keys.append(pd.Series(k[m], index=t["entity_id"].to_numpy()[m]))
        del t, k
    pool_vc = pd.concat(pool_counts).groupby(level=0).sum()
    pr = pd.concat(pool_rows, ignore_index=True)
    pool_amb = pr.groupby("country").agg(pool_rows=("n_s1", "size"),
                                         share_name_in_0_s1=("n_s1", lambda x: float((x == 0).mean())),
                                         share_name_in_1_s1=("n_s1", lambda x: float((x == 1).mean())),
                                         share_name_in_2plus_s1=("n_s1", lambda x: float((x >= 2).mean())))
    del pr, pool_counts, pool_rows
    ck = pd.concat(cand_keys)
    sp_ = smp_pairs.assign(ck=ck.reindex(smp_pairs["cand_id"]).to_numpy())
    key_of = pd.Series(smp_key, index=smp["entity_id"].to_numpy())
    sp_["sk"] = key_of.reindex(sp_["s1_id"]).to_numpy()
    same = sp_[sp_["ck"] == sp_["sk"]].groupby("s1_id").size()
    matched_same = same.reindex(smp["entity_id"]).fillna(0).to_numpy()
    coll = collision_negatives(smp_key, smp["country"].to_numpy(), matched_same, pool_vc)
    has_match = smp["entity_id"].isin(set(smp_pairs["s1_id"])).to_numpy()
    exact_hit = float((matched_same[has_match] > 0).mean())
    rep += ["## D. Ambiguity / hard negatives (train)\n",
            "### D1. S1 entities whose (country, name_core) is shared with ≥1 other S1 entity\n",
            md_table(share_tab), "",
            "### D2. Pool records by how many S1 entities carry their exact (country, name_core)\n",
            md_table(pool_amb), "",
            f"### D3. Name-collision negatives for {len(smp):,} sampled S1 entities\n",
            "Pool records in the same country with the *exact same* name_core that are **not** ground-truth "
            f"matches. For comparison, {exact_hit:.3f} of sampled S1 entities with ≥1 match have ≥1 "
            "match with an exactly equal name_core.\n",
            md_table(coll), ""]
    del s1n, s1_key, s1_vc, pool_vc

    # ---------------- E
    log("E: train->test shift (vocab)", t0)
    tr_paths = [paths[s] for s in (1, 2, 3)]
    tr_countries = sorted(pd.unique(samp.loc[samp["split"] == "train", "country"]))
    vocab = {"core": {"ALL": arrow_vocab(tr_paths, "name_core")},
             "addr": {"ALL": arrow_vocab(tr_paths, "addr_alpha")}}
    for c in tr_countries:
        vocab["core"][c] = arrow_vocab(tr_paths, "name_core", c)
        vocab["addr"][c] = arrow_vocab(tr_paths, "addr_alpha", c)
    log("  vocab built", t0)
    rows = []
    for (split, c), g in samp.groupby(["split", "country"]):
        base = {"split": split, "country": c, **profile(g)}
        refs = ["ALL"] + [x for x in tr_countries if x != c] if split == "test" else \
            [x for x in tr_countries if x != c]
        for r in refs:
            tok_c, rec_c = oov_rates(g["name_core"], vocab["core"][r])
            tok_a, _ = oov_rates(g["addr_alpha"], vocab["addr"][r])
            rows.append({**base, "vs_train_vocab": r, "core_tok_oov": tok_c,
                         "core_rec_oov": rec_c, "addr_tok_oov": tok_a})
    shift = pd.DataFrame(rows).set_index(["split", "country", "vs_train_vocab"])
    shift = shift.drop(columns=["rows"])
    rep += ["## E. Train → test shift\n",
            "Profiles on the samples of section A pooled over sources. OOV = share of name_core / addr_alpha "
            "tokens absent from the train vocabulary (`vs_train_vocab`: ALL = full train; a country label = "
            "train records of that country only). Train rows use the *other* train country as vocabulary: a "
            "held-out-country baseline that mimics the France situation.\n",
            md_table(shift), ""]
    for c in sorted(set(samp.loc[samp["split"] == "test", "country"]) - set(tr_countries)):
        g = samp[(samp["split"] == "test") & (samp["country"] == c)]
        rep += [f"### E2. Unseen country `{c}`: top name_core / addr_alpha tokens (test sample)\n",
                md_table(top_tokens(g["name_core"], vocab["core"]["ALL"], 25).join(
                    top_tokens(g["addr_alpha"], vocab["addr"]["ALL"], 25), rsuffix="_addr")), "",
                f"Legal-suffix tokens recognised in `{c}` (share of records): "
                + ", ".join(f"{k} {v:.3f}" for k, v in g["legal"].value_counts(normalize=True).head(10).items())
                + "\n"]
    del vocab

    # ---------------- F
    log("F: blocking keys", t0)
    frac = args.key_frac
    kf = pd.concat([sample_parquet(paths[s], ["country", "name_core", "addr_alpha", "street_key", "postcode"],
                                   int(pq.ParquetFile(paths[s]).metadata.num_rows * frac), args.seed + 10 + s)
                    for s in (2, 3)], ignore_index=True)
    kstats, lookup = key_df_stats(kf, frac)
    del kf
    share = pair_key_sharing(a, b, pc_country, lookup)
    rep += ["## F. Blocking key statistics\n",
            f"### F1. Pool document frequency per key family ({frac:.0%} pool sample per country, df scaled "
            "by 1/frac; blocking drops keys with df > 2000)\n",
            "Families: n = name token, nb = adjacent name-token bigram, nm = sorted initials, a = address "
            "alpha token, ab = house number + street word, na = name token x address token.\n",
            md_table(kstats), "",
            f"### F2. Matched pairs sharing ≥1 key per family ({len(a):,} pairs)\n",
            "`usable` = shared key with estimated df ≤ 2000 (i.e. survives the blocking cut); "
            "`only_this_family*` = recall that would be lost if that family were removed.\n",
            md_table(share, 4), ""]

    rep += ["## G. Implications\n", "_(written by hand after the run)_\n"]
    elapsed = time.time() - t0
    rep += [f"\n---\nRuntime {elapsed / 60:.1f} min; peak memory {peak_mem_mb():,.0f} MB.\n"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep), encoding="utf-8")
    log(f"wrote {out} (peak mem {peak_mem_mb():,.0f} MB)", t0)


if __name__ == "__main__":
    main()
