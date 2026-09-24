"""E00: write an all-empty submission (every S1 row, empty lists) to verify the upload path.

Expected public LB ≈ singleton rate (~0.056). Run from code/business_entity_resolution/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.io_utils import load_tsv, write_lists  # noqa: E402


def main():
    """Write empty matching/candidate files for all test S1 ids."""
    ids = load_tsv(config.REPO_ROOT / "dataset" / "test" / "test_source1.tsv")["entity_id"].tolist()
    out = config.REPO_ROOT / "output"
    write_lists(out / "matching_results.tsv", ids, {}, ["source1_entity_id", "matched_entity_ids"])
    write_lists(out / "candidate_pairs.tsv", ids, {}, ["source1_entity_id", "candidate_entity_ids"])
    print(f"wrote {len(ids):,} rows to {out}")


if __name__ == "__main__":
    main()
