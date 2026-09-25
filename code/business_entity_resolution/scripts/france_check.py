"""France check (docs/PROMPTS.md Prompt 10): compare France's match behaviour to US/India.

Usage (from code/business_entity_resolution/):
    python scripts/france_check.py
    python scripts/france_check.py --test-dir ../../dataset/test --matching ../../output/matching_results.tsv
    python scripts/france_check.py --scored ../../cache/scored_test.parquet   # adds probability distribution

Read-only analysis tool. It never calls the pipeline, never trains anything, and never writes to
output/ -- it only reads dataset/test/test_source1.tsv (for the country label per S1 entity) and
output/matching_results.tsv (for match counts), and prints:
  1. share of S1 entities with >=1 match, per country label (France vs US vs India vs any others)
  2. mean matches per matched entity, per country label (averaged only over entities that got a
     match, so singletons/empties don't dilute it)
  3. (only if --scored is given) the distribution of the decoder's calibrated probability `p` for
     the pairs that were kept as matches, per country label
  4. a recommendation on whether France looks different enough from US/India to justify spending one
     of the team's 5-per-day / 15-total leaderboard uploads (CLAUDE.md upload budget) on a
     France-specific diagnostic upload (the same submission with France rows emptied, to isolate
     France's contribution to the public LB score)

Data availability note on item 3 (probability distribution):
`matching_results.tsv` only carries the final `matched_entity_ids` -- no probability score. The
calibrated probability for the kept pairs lives in an in-memory dataframe inside
`src/run_pipeline.py` (`scored_t`/`selt`, built around the `score_stream`/`apply_decoder` calls near
the end of `main()`, with columns `s1_rid`, `pool_rid`, `p`) that is never persisted to disk in a
normal run. So this script cannot compute item 3 from `matching_results.tsv` alone. Two ways to feed
it, both supported by this module:
  (a) From a real pipeline run: dump `selt` (the decoder's *kept* pairs, not the full candidate set)
      to a parquet/tsv file with an S1-entity-id column and a probability column, e.g.
          selt.assign(s1_id=s1t["entity_id"].to_numpy()[selt["s1_rid"].to_numpy()]) \\
              [["s1_id", "p"]].to_parquet(cache_path / "scored_test_selected.parquet")
      then pass that path via `--scored`.
  (b) Programmatically: call `probability_stats_by_country(scored_df, country_map)` directly from
      Python (a notebook, or right after `write_outputs()` in run_pipeline.py) with the in-memory
      dataframe -- no file round-trip needed. This is the natural integration point once a real
      pipeline run exists.
If `--scored` is omitted, the script skips item 3 and says so explicitly in the report rather than
guessing or silently leaving it out.

"Meaningfully different" thresholds (see `recommend()`): a >=10-percentage-point gap in match rate,
or a >=25% relative gap in mean-matches-per-matched-entity, between France and the US/India average.
Rationale: the challenge metric (macro F0.5 per S1 entity) is dominated by the empty-vs-non-empty
decision and by matched-set precision/recall, so a double-digit-percentage-point swing in match rate
is large enough to plausibly reflect a systematic blocking/decision problem on the unseen country
label rather than sampling noise on however many France rows the test set has.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.io_utils import load_tsv

ROOT = Path(__file__).resolve().parents[3]  # student_resource/
PKG = ROOT / "code" / "business_entity_resolution"

DEFAULT_TEST_DIR = ROOT / "dataset" / "test"
DEFAULT_MATCHING = ROOT / "output" / "matching_results.tsv"

RATE_DELTA_THRESHOLD = 0.10        # 10 percentage points of match-rate gap
MEAN_MATCHES_REL_THRESHOLD = 0.25  # 25% relative gap in mean-matches-per-matched-entity


def load_country_map(test_source1_path):
    """Return {entity_id: country} for every S1 entity in test_source1.tsv."""
    s1 = load_tsv(test_source1_path)
    return dict(zip(s1["entity_id"], s1["country"]))


def load_match_counts(matching_results_path):
    """Return {entity_id: n_matched_ids} from matching_results.tsv (0 for an empty list)."""
    m = load_tsv(matching_results_path)

    def _n(s):
        return len(s.split(",")) if s else 0

    return dict(zip(m["source1_entity_id"], m["matched_entity_ids"].map(_n)))


def country_stats(country_map, match_counts):
    """Per-country match-rate and mean-matches-per-matched-entity.

    Returns {country: {"n_entities": int, "n_matched": int, "match_rate": float,
    "mean_matches_per_matched": float or None}}. `match_rate` is the share of entities in that
    country with >=1 match. `mean_matches_per_matched` is averaged only over entities that got
    >=1 match (None if none did), so it isn't diluted by singleton/no-match entities.
    """
    by_country = {}
    for eid, country in country_map.items():
        n = match_counts.get(eid, 0)
        d = by_country.setdefault(country, {"n_entities": 0, "n_matched": 0, "matched_counts": []})
        d["n_entities"] += 1
        if n > 0:
            d["n_matched"] += 1
            d["matched_counts"].append(n)
    out = {}
    for country, d in by_country.items():
        mc = d["matched_counts"]
        out[country] = {
            "n_entities": d["n_entities"],
            "n_matched": d["n_matched"],
            "match_rate": (d["n_matched"] / d["n_entities"]) if d["n_entities"] else 0.0,
            "mean_matches_per_matched": (sum(mc) / len(mc)) if mc else None,
        }
    return out


def probability_stats_by_country(scored_df, country_map, id_col=None, prob_col="p"):
    """Distribution of kept-match probabilities per country.

    `scored_df` must have a probability column (`prob_col`, default "p") and an S1 entity-id column
    -- auto-detected from {"s1_id", "source1_entity_id", "entity_id"} unless `id_col` is given.
    One row per kept (S1, candidate) match pair, e.g. `selt` from src/run_pipeline.py with `s1_rid`
    mapped back to `entity_id` strings (see the module docstring). Returns {country: {"n": int,
    "mean": float, "median": float, "p25": float, "p75": float, "min": float, "max": float}},
    skipping any country with zero rows.
    """
    if id_col is None:
        for c in ("s1_id", "source1_entity_id", "entity_id"):
            if c in scored_df.columns:
                id_col = c
                break
        else:
            raise ValueError("scored_df needs an S1 id column: s1_id / source1_entity_id / entity_id")
    df = scored_df[[id_col, prob_col]].copy()
    df["country"] = df[id_col].map(country_map)
    out = {}
    for country, g in df.groupby("country"):
        p = g[prob_col].astype(float)
        if len(p) == 0:
            continue
        out[country] = {
            "n": int(len(p)),
            "mean": float(p.mean()),
            "median": float(p.median()),
            "p25": float(p.quantile(0.25)),
            "p75": float(p.quantile(0.75)),
            "min": float(p.min()),
            "max": float(p.max()),
        }
    return out


def load_scored(path, prob_col="p"):
    """Load an optional scored-pairs artifact (parquet, or tsv/csv) for the probability breakdown.

    Not produced automatically by this script -- see the module docstring for how to dump it from a
    real src/run_pipeline.py run. Coerces `prob_col` to float (tsv/csv round-trips as text).
    """
    path = Path(path)
    if path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    df[prob_col] = df[prob_col].astype(float)
    return df


def recommend(stats, france_label="France", baseline_labels=("US", "India")):
    """Decide whether France's match behaviour looks meaningfully different from US/India.

    Compares France's match_rate and mean_matches_per_matched against the average over whichever of
    `baseline_labels` are present in `stats`. "Meaningfully different" = match-rate gap >=
    RATE_DELTA_THRESHOLD (10 percentage points) OR mean-matches-per-matched relative gap >=
    MEAN_MATCHES_REL_THRESHOLD (25%) -- see module docstring for the rationale. Returns
    {"different": bool, "reasons": [str, ...], "message": <printable recommendation text>}.
    """
    present = [b for b in baseline_labels if b in stats]
    if france_label not in stats or not present:
        return {
            "different": False,
            "reasons": ["insufficient data (missing France or all baseline countries)"],
            "message": "Not enough data to compare France to US/India (one or both are absent from "
                       "the results).",
        }
    fr = stats[france_label]
    base_rate = sum(stats[b]["match_rate"] for b in present) / len(present)
    base_mm_vals = [stats[b]["mean_matches_per_matched"] for b in present
                     if stats[b]["mean_matches_per_matched"] is not None]
    base_mean_matches = (sum(base_mm_vals) / len(base_mm_vals)) if base_mm_vals else None

    reasons = []
    rate_gap = abs(fr["match_rate"] - base_rate)
    if rate_gap >= RATE_DELTA_THRESHOLD:
        reasons.append(
            f"match-rate gap {rate_gap:.1%} >= {RATE_DELTA_THRESHOLD:.0%} threshold "
            f"(France {fr['match_rate']:.1%} vs US/India avg {base_rate:.1%})"
        )
    if fr["mean_matches_per_matched"] is not None and base_mean_matches:
        rel_gap = abs(fr["mean_matches_per_matched"] - base_mean_matches) / base_mean_matches
        if rel_gap >= MEAN_MATCHES_REL_THRESHOLD:
            reasons.append(
                f"mean-matches-per-matched relative gap {rel_gap:.1%} >= "
                f"{MEAN_MATCHES_REL_THRESHOLD:.0%} threshold (France "
                f"{fr['mean_matches_per_matched']:.2f} vs US/India avg {base_mean_matches:.2f})"
            )

    different = bool(reasons)
    if different:
        message = (
            "France looks meaningfully different from US/India:\n  - " + "\n  - ".join(reasons) +
            "\n\nTrade-off (Prompt 10): (a) keep the submission as-is, betting the gap is real and "
            "the model is simply behaving differently (correctly) for an unseen country label; or "
            "(b) spend one leaderboard upload on a France-specific diagnostic -- the same submission "
            "with France rows emptied, to isolate France's contribution to the public LB score. "
            "This costs one of the team's 5-per-day / 15-total uploads (CLAUDE.md upload budget) -- "
            "worth spending only because the gap above looks large enough that confirming it (rather "
            "than guessing) is likely worth the upload."
        )
    else:
        message = (
            "France's match-rate and mean-matches-per-matched are within threshold of the US/India "
            "average -- no red flag. Recommendation: (a) keep the submission as-is; spending an "
            "upload on a France-specific diagnostic is not justified by this gap."
        )
    return {"different": different, "reasons": reasons, "message": message}


def format_report(stats, prob_stats=None):
    """Build the printable France-vs-US/India report string (stats table + recommendation)."""
    lines = ["France check (docs/PROMPTS.md Prompt 10)", "=" * 42, ""]
    lines.append(f"{'country':<10} {'n_entities':>10} {'match_rate':>11} {'mean_matches':>13}")
    for country in sorted(stats):
        s = stats[country]
        mm = f"{s['mean_matches_per_matched']:.2f}" if s["mean_matches_per_matched"] is not None else "n/a"
        lines.append(f"{country:<10} {s['n_entities']:>10} {s['match_rate']:>10.1%} {mm:>13}")
    lines.append("")
    if prob_stats:
        lines.append("Probability distribution of kept matches (calibrated p), per country:")
        for country in sorted(prob_stats):
            p = prob_stats[country]
            lines.append(
                f"  {country}: n={p['n']} mean={p['mean']:.3f} median={p['median']:.3f} "
                f"p25={p['p25']:.3f} p75={p['p75']:.3f} min={p['min']:.3f} max={p['max']:.3f}"
            )
    else:
        lines.append(
            "Probability distribution: skipped (no --scored artifact given). "
            "matching_results.tsv alone carries no probability column; pass --scored pointing at a "
            "dumped scored/selected-pairs file to include this breakdown (see module docstring)."
        )
    lines.append("")
    lines.append(recommend(stats)["message"])
    return "\n".join(lines)


def parse_args(argv=None):
    """CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR), help="dir containing test_source1.tsv")
    ap.add_argument("--matching", default=str(DEFAULT_MATCHING), help="path to matching_results.tsv")
    ap.add_argument("--scored", default=None,
                     help="optional scored/selected-pairs artifact (parquet or tsv/csv) with an S1 "
                          "id column and a probability column, for the item-3 breakdown (see module "
                          "docstring for how to produce it)")
    ap.add_argument("--prob-col", default="p", help="probability column name in --scored (default: p)")
    return ap.parse_args(argv)


def main(argv=None):
    """Read test_source1.tsv + matching_results.tsv (and optionally --scored), print the report."""
    args = parse_args(argv)

    country_map = load_country_map(Path(args.test_dir) / "test_source1.tsv")
    match_counts = load_match_counts(args.matching)
    stats = country_stats(country_map, match_counts)

    prob_stats = None
    if args.scored:
        scored_df = load_scored(args.scored, prob_col=args.prob_col)
        prob_stats = probability_stats_by_country(scored_df, country_map, prob_col=args.prob_col)

    report = format_report(stats, prob_stats)
    print(report)
    return report


if __name__ == "__main__":
    main()
