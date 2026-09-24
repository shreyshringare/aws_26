"""Round-trip test: our writer output must pass the official validator."""
import subprocess
import sys

from src import config
from src.io_utils import load_tsv, write_lists

VALIDATOR = config.REPO_ROOT / "utils" / "validate_submission.py"


def _src(path, rows):
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                    + "".join(f"{r}\tN\tA\tUS\n" for r in rows), encoding="utf-8")


def test_writer_passes_validator(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _src(test_dir / "test_source1.tsv", ["S1-1", "S1-2", "S1-3"])
    _src(test_dir / "test_source2.tsv", ["S2-1", "S2-2"])
    _src(test_dir / "test_source3.tsv", ["S3-1"])
    ids = ["S1-1", "S1-2", "S1-3"]
    m, c = tmp_path / "matching_results.tsv", tmp_path / "candidate_pairs.tsv"
    write_lists(m, ids, {"S1-1": ["S2-1", "S3-1", "S2-1"]}, ["source1_entity_id", "matched_entity_ids"])
    write_lists(c, ids, {"S1-1": ["S2-1", "S3-1", "S2-2"], "S1-2": ["S2-2"]},
                ["source1_entity_id", "candidate_entity_ids"])
    df = load_tsv(m)
    assert df["matched_entity_ids"].tolist() == ["S2-1,S3-1", "", ""]
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(m), "--candidate", str(c),
                        "--test-dir", str(test_dir), "--check-ids"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
