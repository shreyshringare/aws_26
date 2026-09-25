"""Tests for scripts/france_check.py: per-country match stats, probability breakdown, recommendation."""
import pandas as pd
import pytest

from scripts import france_check


def _write_test_source1(path, rows):
    """Write a tiny test_source1.tsv. `rows` is a list of (entity_id, country)."""
    lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
    lines += [f"{eid}\tName {eid}\tAddr {eid}\t{country}" for eid, country in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_matching(path, rows):
    """Write a tiny matching_results.tsv. `rows` is a list of (entity_id, matched_ids_csv_or_empty)."""
    lines = ["source1_entity_id\tmatched_entity_ids"]
    lines += [f"{eid}\t{matched}" for eid, matched in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_country_map(tmp_path):
    """load_country_map reads entity_id -> country from test_source1.tsv."""
    p = tmp_path / "test_source1.tsv"
    _write_test_source1(p, [("S1-1", "US"), ("S1-2", "France")])
    cm = france_check.load_country_map(p)
    assert cm == {"S1-1": "US", "S1-2": "France"}


def test_load_match_counts(tmp_path):
    """load_match_counts counts comma-separated ids, 0 for an empty second column."""
    p = tmp_path / "matching_results.tsv"
    _write_matching(p, [("S1-1", "S2-1,S2-2"), ("S1-2", ""), ("S1-3", "S3-1")])
    mc = france_check.load_match_counts(p)
    assert mc == {"S1-1": 2, "S1-2": 0, "S1-3": 1}


def test_country_stats_match_rate_and_mean_matches():
    """country_stats computes per-country match rate and mean-matches-per-matched-entity."""
    country_map = {"S1-1": "US", "S1-2": "US", "S1-3": "France", "S1-4": "France"}
    match_counts = {"S1-1": 2, "S1-2": 0, "S1-3": 1, "S1-4": 1}
    stats = france_check.country_stats(country_map, match_counts)
    assert stats["US"]["n_entities"] == 2
    assert stats["US"]["n_matched"] == 1
    assert stats["US"]["match_rate"] == pytest.approx(0.5)
    assert stats["US"]["mean_matches_per_matched"] == pytest.approx(2.0)
    assert stats["France"]["match_rate"] == pytest.approx(1.0)
    assert stats["France"]["mean_matches_per_matched"] == pytest.approx(1.0)


def test_country_stats_no_matches_gives_none_mean():
    """A country where nobody matched gets mean_matches_per_matched=None, not a ZeroDivisionError."""
    country_map = {"S1-1": "France"}
    match_counts = {"S1-1": 0}
    stats = france_check.country_stats(country_map, match_counts)
    assert stats["France"]["match_rate"] == 0.0
    assert stats["France"]["mean_matches_per_matched"] is None


def test_probability_stats_by_country_basic():
    """probability_stats_by_country groups kept-match probabilities by country and summarizes them."""
    country_map = {"S1-1": "US", "S1-2": "US", "S1-3": "France"}
    scored = pd.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
        "p": [0.9, 0.8, 0.7, 0.95],
    })
    stats = france_check.probability_stats_by_country(scored, country_map)
    assert stats["US"]["n"] == 3
    assert stats["US"]["mean"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert stats["France"]["n"] == 1
    assert stats["France"]["mean"] == pytest.approx(0.95)


def test_probability_stats_by_country_id_col_autodetect():
    """The S1 id column is auto-detected from source1_entity_id when s1_id is absent."""
    country_map = {"S1-1": "US"}
    scored = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-1"], "p": [0.5, 0.6]})
    stats = france_check.probability_stats_by_country(scored, country_map)
    assert stats["US"]["n"] == 2


def test_probability_stats_by_country_missing_id_col_raises():
    """A scored_df with no recognizable id column raises a clear ValueError."""
    scored = pd.DataFrame({"weird_col": ["S1-1"], "p": [0.5]})
    with pytest.raises(ValueError):
        france_check.probability_stats_by_country(scored, {"S1-1": "US"})


def test_recommend_france_looks_fine():
    """When France's match rate/mean matches are close to the US/India average, recommend (a) only."""
    stats = {
        "US": {"match_rate": 0.90, "mean_matches_per_matched": 1.5},
        "India": {"match_rate": 0.88, "mean_matches_per_matched": 1.4},
        "France": {"match_rate": 0.89, "mean_matches_per_matched": 1.45},
    }
    rec = france_check.recommend(stats)
    assert rec["different"] is False
    assert "keep the submission as-is" in rec["message"]
    assert "not justified" in rec["message"]
    assert "France rows emptied" not in rec["message"]  # no need to spell out the diagnostic upload


def test_recommend_france_looks_meaningfully_worse():
    """A large match-rate gap between France and US/India triggers the (a)/(b) diagnostic-upload text."""
    stats = {
        "US": {"match_rate": 0.90, "mean_matches_per_matched": 1.5},
        "India": {"match_rate": 0.88, "mean_matches_per_matched": 1.4},
        "France": {"match_rate": 0.40, "mean_matches_per_matched": 1.4},
    }
    rec = france_check.recommend(stats)
    assert rec["different"] is True
    assert rec["reasons"]
    assert "diagnostic" in rec["message"]
    assert "France rows emptied" in rec["message"]
    assert "5-per-day" in rec["message"] or "15-total" in rec["message"]


def test_recommend_missing_france_or_baseline_is_handled():
    """If France or every baseline country is absent from stats, recommend() reports insufficient data."""
    rec = france_check.recommend({"US": {"match_rate": 0.9, "mean_matches_per_matched": 1.2}})
    assert rec["different"] is False
    assert "Not enough data" in rec["message"]


def test_format_report_without_scored_notes_the_skip():
    """format_report() explains why the probability section is skipped when prob_stats is None."""
    stats = {"US": {"n_entities": 2, "n_matched": 1, "match_rate": 0.5, "mean_matches_per_matched": 2.0}}
    report = france_check.format_report(stats, prob_stats=None)
    assert "skipped" in report
    assert "--scored" in report


def test_format_report_with_scored_includes_distribution():
    """format_report() prints the per-country probability summary when prob_stats is given."""
    stats = {"US": {"n_entities": 2, "n_matched": 1, "match_rate": 0.5, "mean_matches_per_matched": 2.0}}
    prob_stats = {"US": {"n": 2, "mean": 0.8, "median": 0.8, "p25": 0.75, "p75": 0.85, "min": 0.7, "max": 0.9}}
    report = france_check.format_report(stats, prob_stats=prob_stats)
    assert "Probability distribution" in report
    assert "mean=0.800" in report


def test_load_scored_tsv_coerces_prob_col(tmp_path):
    """load_scored() reads a tsv artifact and coerces the probability column to float."""
    p = tmp_path / "scored.tsv"
    p.write_text("s1_id\tp\nS1-1\t0.5\nS1-2\t0.75\n", encoding="utf-8")
    df = france_check.load_scored(p)
    assert df["p"].tolist() == pytest.approx([0.5, 0.75])


def test_load_scored_parquet(tmp_path):
    """load_scored() reads a parquet artifact directly (probability column already numeric)."""
    p = tmp_path / "scored.parquet"
    pd.DataFrame({"s1_id": ["S1-1"], "p": [0.42]}).to_parquet(p)
    df = france_check.load_scored(p)
    assert df["p"].tolist() == pytest.approx([0.42])


def test_main_end_to_end_prints_report(tmp_path, capsys):
    """main() reads test_source1.tsv + matching_results.tsv from disk and prints the full report."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_test_source1(test_dir / "test_source1.tsv", [
        ("S1-1", "US"), ("S1-2", "US"), ("S1-3", "India"), ("S1-4", "France"),
    ])
    matching = tmp_path / "matching_results.tsv"
    _write_matching(matching, [
        ("S1-1", "S2-1,S2-2"), ("S1-2", "S2-3"), ("S1-3", "S3-1"), ("S1-4", ""),
    ])
    france_check.main(["--test-dir", str(test_dir), "--matching", str(matching)])
    out = capsys.readouterr().out
    assert "France check" in out
    assert "France" in out and "US" in out and "India" in out
    assert "skipped" in out  # no --scored given


def test_main_with_scored_includes_probability_section(tmp_path, capsys):
    """main() with --scored adds the probability-distribution section to the printed report."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_test_source1(test_dir / "test_source1.tsv", [("S1-1", "US"), ("S1-2", "France")])
    matching = tmp_path / "matching_results.tsv"
    _write_matching(matching, [("S1-1", "S2-1"), ("S1-2", "S3-1")])
    scored = tmp_path / "scored.tsv"
    scored.write_text("s1_id\tp\nS1-1\t0.9\nS1-2\t0.6\n", encoding="utf-8")
    france_check.main(["--test-dir", str(test_dir), "--matching", str(matching), "--scored", str(scored)])
    out = capsys.readouterr().out
    assert "Probability distribution of kept matches" in out
