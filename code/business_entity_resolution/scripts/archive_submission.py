"""Archive the current output/ files as a leaderboard submission (version history, per guidelines).

Usage (from code/business_entity_resolution/), right after uploading:
    python scripts/archive_submission.py --tag sub-d1-02 --exp-id E03 [--public-lb 0.61]

Writes student_resource/submissions/<tag>/{matching_results.tsv.gz,candidate_pairs.tsv.gz,meta.json}
and creates the git tag <tag> on the current commit. submissions/ is gitignored (large files);
sync it to the team S3 bucket.
"""
import argparse
import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # student_resource/


def _git(*args):
    """Run a git command in the repo root and return stripped stdout."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main():
    """Copy output files (gzipped) plus metadata into submissions/<tag>/ and tag the commit."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--public-lb", default="")
    args = ap.parse_args()
    dest = ROOT / "submissions" / args.tag
    dest.mkdir(parents=True, exist_ok=False)
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        src = ROOT / "output" / name
        if src.exists():
            with open(src, "rb") as fi, gzip.open(dest / f"{name}.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
    meta = {"tag": args.tag, "exp_id": args.exp_id, "public_lb": args.public_lb,
            "commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain")),
            "archived_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    _git("tag", args.tag)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
