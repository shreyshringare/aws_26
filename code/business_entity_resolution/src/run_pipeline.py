"""End-to-end pipeline entry point.

Run from code/business_entity_resolution/:
    python -m src.run_pipeline --data ../../dataset --eval-only                   # dev OOF evaluation
    python -m src.run_pipeline --data ../../dataset --eval-only --stop-after blocking
    python -m src.run_pipeline --data ../../dataset --out ../../output            # train -> test outputs
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import blocking, config, context, decide, evaluate, features, io_utils, model, noise, normalize, translit

LOG_FIELDS = ["exp_id", "change", "block_recall", "cands_per_entity", "f05_all", "f05_us", "f05_in",
              "f05_singletons", "f05_matched", "us_to_in", "in_to_us", "t_match", "t_empty",
              "public_lb", "decision"]


def log(msg):
    """Print a timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cache_path(name, ext="parquet"):
    """Path of a cached artifact, namespaced by config.CACHE_TAG."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / f"{config.CACHE_TAG}_{name}.{ext}"


def cached_frame(name, fn):
    """Load cache/<tag>_<name>.parquet if present, else compute fn(), save and return it."""
    p = cache_path(name)
    if p.exists():
        log(f"cache hit {p.name}")
        return pd.read_parquet(p)
    df = fn()
    df.to_parquet(p, index=False)
    return df


def get_translit(data):
    """Learned native-script table (fitted on train ground-truth name pairs, cached)."""
    p = cache_path("translit", "tsv")
    if p.exists():
        return translit.load_table(p)
    d = Path(data) / "train"
    s1 = io_utils.load_tsv(d / "train_source1.tsv")[["entity_id", "business_name"]]
    pool = pd.concat([io_utils.load_tsv(d / f"train_source{k}.tsv")[["entity_id", "business_name"]]
                      for k in (2, 3)], ignore_index=True)
    pairs = io_utils.gt_pairs(io_utils.load_ground_truth(data))
    j = (pairs.merge(s1, left_on="s1_id", right_on="entity_id")
         .merge(pool, left_on="cand_id", right_on="entity_id", suffixes=("_s1", "_pool")))
    table = translit.fit_table(j["business_name_s1"].tolist(), j["business_name_pool"].tolist())
    translit.save_table(table, p)
    log(f"translit table: {len(table):,} tokens")
    return table


def load_norm(data, split, source, table):
    """Normalized frame of one source file, cached."""
    def make():
        raw = io_utils.load_tsv(Path(data) / split / f"{split}_{source}.tsv")
        return normalize.normalize_frame(raw, table, config.N_JOBS)
    return cached_frame(f"norm_{split}_{source}", make)


def load_split_norm(data, split, table):
    """Normalized (S1, pool=S2+S3) frames with contiguous rids and frequency columns."""
    s1 = load_norm(data, split, "source1", table)
    pool = pd.concat([load_norm(data, split, "source2", table),
                      load_norm(data, split, "source3", table)], ignore_index=True)
    s1["rid"] = np.arange(len(s1), dtype=np.int64)
    pool["rid"] = np.arange(len(pool), dtype=np.int64)
    features.add_frequency_columns(s1, pool)
    return s1, pool


def get_generic_vocab(s1n, pooln, pos):
    """Learned generic add-on name tokens (noise channel), cached as JSON."""
    p = cache_path("generic_vocab", "json")
    if p.exists():
        return set(json.loads(p.read_text()))
    a = s1n["name_core"].take(pos["s1_rid"].to_numpy()).tolist()
    b = pooln["name_core"].take(pos["pool_rid"].to_numpy()).tolist()
    vocab = noise.fit_generic_vocab(a, b)
    p.write_text(json.dumps(sorted(vocab)))
    log(f"generic vocab: {len(vocab)} tokens")
    return vocab


def gt_rid_pairs(gt, s1n, pooln):
    """Positive pairs as (s1_rid, pool_rid)."""
    p = io_utils.gt_pairs(gt)
    m1 = pd.Series(s1n["rid"].to_numpy(), index=s1n["entity_id"].to_numpy())
    m2 = pd.Series(pooln["rid"].to_numpy(), index=pooln["entity_id"].to_numpy())
    return pd.DataFrame({"s1_rid": m1.reindex(p["s1_id"].to_numpy()).to_numpy(np.int64),
                         "pool_rid": m2.reindex(p["cand_id"].to_numpy()).to_numpy(np.int64)})


def sample_s1(s1n, frac):
    """Stratified (per country label) sample of S1 rids; frac >= 1 returns all."""
    if frac >= 1:
        return s1n["rid"].to_numpy()
    return s1n.groupby("country", group_keys=False).sample(frac=frac, random_state=config.SEED)["rid"].to_numpy()


def pair_key(s1_rid, pool_rid):
    """Pack (s1_rid, pool_rid) into one int64 for fast set operations."""
    return np.asarray(s1_rid, np.int64) * (1 << 32) + np.asarray(pool_rid, np.int64)


def blocking_summary(cands, pos, n_entities):
    """Pair recall, candidates/entity and expected missed positives per entity."""
    hit = int(np.isin(pair_key(pos["s1_rid"], pos["pool_rid"]),
                      pair_key(cands["s1_rid"], cands["pool_rid"])).sum())
    tot = len(pos)
    return {"block_recall": hit / max(tot, 1), "cands_per_entity": len(cands) / max(n_entities, 1),
            "missed_per_entity": (tot - hit) / max(n_entities, 1)}


class Scorer:
    """Maps selected (s1_rid, pool_rid) rows to id lists and scores them with the exact metric."""

    def __init__(self, s1n, pooln, gt, rids):
        """Bind frames, ground truth and the evaluated S1 rids."""
        self.s1n, self.pooln, self.gt = s1n, pooln, gt
        self.rids = np.asarray(rids)
        self.ids = s1n["entity_id"].take(self.rids).tolist()
        self.country = s1n["country"].take(self.rids).to_numpy()

    def lists(self, sel):
        """{s1 entity_id: [pool entity_ids]} for selected rows."""
        a = self.s1n["entity_id"].take(sel["s1_rid"].to_numpy()).to_numpy()
        b = self.pooln["entity_id"].take(sel["pool_rid"].to_numpy()).to_numpy()
        return io_utils.pairs_to_lists(pd.DataFrame({"s1_id": a, "cand_id": b}))

    def __call__(self, sel, mask=None):
        """Macro F0.5 over evaluated entities (optionally a boolean mask over them)."""
        ids = self.ids if mask is None else [i for i, m in zip(self.ids, mask) if m]
        return evaluate.score(self.lists(sel), self.gt, ids)[0]

    def breakdown(self, sel):
        """All / per country / singletons / matched breakdown."""
        return evaluate.breakdown(self.lists(sel), self.gt, self.ids, self.country)


def transfer_report(feats, y, cols, scorer, dec, extra):
    """Cross-country transfer (France proxy): train stage-1 on one label, score each other label."""
    ctry = scorer.s1n["country"].take(feats["s1_rid"].to_numpy()).to_numpy()
    labels = list(pd.unique(ctry))
    out = {}
    for a in labels:
        for b in labels:
            if a == b:
                continue
            ma, mb = ctry == a, ctry == b
            m = model.fit_full(feats.loc[ma, cols], y[ma], n_estimators=300)
            fb = feats.loc[mb, ["s1_rid", "pool_rid"]].assign(p=m.predict_proba(feats.loc[mb, cols])[:, 1])
            sel = decide.apply_decoder(fb, dec, None, extra)
            out[f"{a}->{b}"] = scorer(sel, mask=scorer.country == b)
    return out


def append_log(row):
    """Append one experiment row to experiments/log.csv."""
    p = config.REPO_ROOT / "experiments" / "log.csv"
    p.parent.mkdir(exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def write_outputs(out_dir, s1t, poolt, candt, selt):
    """Write candidate_pairs.tsv (= exact scored set) and matching_results.tsv in test S1 order."""
    out_dir = Path(out_dir)
    ids = s1t["entity_id"].tolist()

    def lists(df):
        a = s1t["entity_id"].take(df["s1_rid"].to_numpy()).to_numpy()
        b = poolt["entity_id"].take(df["pool_rid"].to_numpy()).to_numpy()
        return io_utils.pairs_to_lists(pd.DataFrame({"s1_id": a, "cand_id": b}))

    io_utils.write_lists(out_dir / "candidate_pairs.tsv", ids, lists(candt),
                         ["source1_entity_id", "candidate_entity_ids"])
    io_utils.write_lists(out_dir / "matching_results.tsv", ids, lists(selt),
                         ["source1_entity_id", "matched_entity_ids"])


def validate(out_dir, data):
    """Run the official validator; returns its exit code."""
    v = config.REPO_ROOT / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(v), "--matching", str(Path(out_dir) / "matching_results.tsv"),
                        "--candidate", str(Path(out_dir) / "candidate_pairs.tsv"),
                        "--test-dir", str(Path(data) / "test")])
    return r.returncode


def parse_args(argv=None):
    """CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../../dataset")
    ap.add_argument("--out", default="../../output")
    ap.add_argument("--eval-only", action="store_true", help="dev OOF evaluation, no test outputs")
    ap.add_argument("--frac", type=float, default=None, help="share of train S1 entities to use")
    ap.add_argument("--stop-after", choices=["blocking", "stage1"], default=None)
    ap.add_argument("--no-stage2", action="store_true")
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--exp-id", default="")
    ap.add_argument("--change", default="")
    return ap.parse_args(argv)


def main(argv=None):
    """Run the pipeline end-to-end (see module docstring)."""
    args = parse_args(argv)
    t0 = time.time()
    table = get_translit(args.data)
    s1n, pooln = load_split_norm(args.data, "train", table)
    gt = io_utils.load_ground_truth(args.data)
    pos_all = gt_rid_pairs(gt, s1n, pooln)

    frac = args.frac if args.frac is not None else (config.DEV_FRAC if args.eval_only else config.TRAIN_FRAC)
    rids = sample_s1(s1n, frac)
    pos = pos_all[np.isin(pos_all["s1_rid"].to_numpy(), rids)]
    log(f"train entities={len(rids):,} positives={len(pos):,}")

    cands = cached_frame(f"cands_train_{frac}", lambda: blocking.generate_candidates(s1n, pooln, rids))
    brep = blocking_summary(cands, pos, len(rids))
    log(f"blocking {brep}")
    row = {"exp_id": args.exp_id, "change": args.change,
           "block_recall": round(brep["block_recall"], 4), "cands_per_entity": round(brep["cands_per_entity"], 1)}
    if args.stop_after == "blocking":
        append_log(row)
        return

    generic = get_generic_vocab(s1n, pooln, pos_all)

    feats = cached_frame(f"feats1_train_{frac}",
                         lambda: features.build_features(cands, s1n, pooln, generic))
    y = np.isin(pair_key(feats["s1_rid"], feats["pool_rid"]),
                pair_key(pos["s1_rid"], pos["pool_rid"])).astype(np.int8)
    groups = feats["s1_rid"].to_numpy()
    cols1 = model.feature_cols(feats)
    log(f"stage1: {len(feats):,} pairs, {len(cols1)} features, pos rate {y.mean():.3f}")
    p1, models1 = model.train_oof(feats[cols1], y, groups)
    feats["p1"] = p1

    cols, pcol, models2 = cols1, "p1", None
    if not (args.no_stage2 or args.stop_after == "stage1"):
        feats = context.add_stage2_features(feats, pooln)
        cols = cols1 + model.feature_cols(feats, prefix="f2_")
        log(f"stage2: {len(cols)} features")
        p2, models2 = model.train_oof(feats[cols], y, groups)
        feats["p2"] = p2
        pcol = "p2"

    scorer = Scorer(s1n, pooln, gt, rids)
    extra = brep["missed_per_entity"]
    scored = feats[["s1_rid", "pool_rid"]].assign(p=feats[pcol].to_numpy())
    calib = decide.fit_calibrator(scored["p"].to_numpy(), y)
    dec, grid = decide.tune(scored, calib, extra, scorer)
    log(f"decoder {dec}")
    sel = decide.apply_decoder(scored, dec, calib, extra)
    bd = scorer.breakdown(sel)
    log(f"OOF breakdown {bd}")
    tr = {} if args.no_transfer else transfer_report(feats, y, cols1, scorer, dec, extra)
    log(f"transfer {tr}")
    row.update({"f05_all": round(bd["all"], 5), "f05_us": round(bd.get("country=US", np.nan), 5),
                "f05_in": round(bd.get("country=India", np.nan), 5),
                "f05_singletons": round(bd["singletons"], 5), "f05_matched": round(bd["matched"], 5),
                "us_to_in": round(tr.get("US->India", np.nan), 5), "in_to_us": round(tr.get("India->US", np.nan), 5),
                "t_match": dec.get("t_match", ""), "t_empty": dec.get("t_empty", "")})
    append_log(row)
    cache_path("decoder", "json").write_text(json.dumps(dec))
    if args.eval_only:
        log(f"done in {time.time() - t0:.0f}s")
        return

    log("fitting full models for test")
    m1 = model.fit_full(feats[cols1], y, n_estimators=model.mean_best_iter(models1))
    m2 = model.fit_full(feats[cols], y, n_estimators=model.mean_best_iter(models2)) if models2 else None
    s1t, poolt = load_split_norm(args.data, "test", table)
    candt = cached_frame("cands_test", lambda: blocking.generate_candidates(s1t, poolt))
    log(f"test candidates {len(candt):,} ({len(candt) / len(s1t):.1f}/entity)")
    ft = cached_frame("feats1_test", lambda: features.build_features(candt, s1t, poolt, generic))
    ft["p1"] = m1.predict_proba(ft[cols1])[:, 1]
    if m2 is not None:
        ft = context.add_stage2_features(ft, poolt)
        ft["p2"] = m2.predict_proba(ft[cols])[:, 1]
    scored_t = ft[["s1_rid", "pool_rid"]].assign(p=ft[pcol].to_numpy())
    selt = decide.apply_decoder(scored_t, dec, calib, extra)
    write_outputs(args.out, s1t, poolt, ft[["s1_rid", "pool_rid"]], selt)
    code = validate(args.out, args.data)
    log(f"validator exit={code}; matched entities={selt['s1_rid'].nunique():,}/{len(s1t):,}; "
        f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
