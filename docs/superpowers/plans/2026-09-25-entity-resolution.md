# Business Entity Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that, for every test S1 entity, outputs the matching S2/S3 records, maximising macro F0.5 per S1 entity (Amazon ML Challenge 2026).

**Architecture:** Normalize → per-country IDF key blocking (sparse matmul) → stage-1 pair features incl. noise-channel features → LightGBM (GroupKFold OOF) → stage-2 context + triangle-support features → LightGBM → isotonic calibration → global one-to-one → expected-F0.5 subset selection. Every stage caches parquet under `cache/`.

**Tech Stack:** Python ≥ 3.11, pandas 3 (Arrow strings), numpy, scipy.sparse, rapidfuzz, unidecode, joblib, LightGBM 4.7 (MIT), scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-entity-resolution-design.md`. **Rules:** `CLAUDE.md` (read it first, every shift).

**Conventions for every task**
- Working dir for commands: `student_resource/code/business_entity_resolution/` unless stated.
- Tests: `python -m pytest -q tests` (the `-m` form puts the package root on `sys.path`).
- Every function gets a docstring (organiser requirement).
- Never use `country` as a feature. Only as a blocking group key.
- Commit after every task. Update `HANDOFF.md` at the end of every shift.

**Shift map**
| Phase | Tasks | Target shift | Exit criterion |
|---|---|---|---|
| 0 Foundation | 1–4 | S1 | tests green, E00 all-empty file passes validator |
| 1 Normalization | 5–7 | S1–S2 | normalize tests green, translit table fitted |
| 2 Blocking | 8–9 | S2 | dev blocking recall reported (target ≥ 0.97) |
| 3 Stage-1 + first submission | 10–15 | S3 | dev OOF F0.5 printed, validated test outputs, upload #2 |
| 4 Differentiators | 16–18 | D2 | stage-2 + EF0.5 decoder evaluated vs baseline, transfer reported |
| 5 Packaging | 19–21 | D3 | zip built, doc filled, final upload chosen |

---

## File map

| File | Responsibility |
|---|---|
| `src/__init__.py` | package marker |
| `src/config.py` | paths, seeds, hyper-parameters (exists — modified in Task 1) |
| `src/io_utils.py` | TSV load, GT parsing, submission writing (exists) |
| `src/evaluate.py` | exact macro F0.5, breakdowns, blocking report (exists) |
| `src/normalize.py` | hand-written dictionaries + name/address normalization |
| `src/translit.py` | learned native-script → Latin token table |
| `src/blocking.py` | key hashing, IDF sparse index, top-K candidates per country |
| `src/noise.py` | noise-channel features + learned generic-word vocabulary |
| `src/features.py` | stage-1 pair features |
| `src/model.py` | LightGBM OOF / full fit |
| `src/context.py` | stage-2 group, competition and triangle features |
| `src/decide.py` | calibration, one-to-one, threshold + expected-F0.5 decoders, tuning |
| `src/run_pipeline.py` | CLI orchestration, experiment logging, output writing, validation |
| `src/eda.py` | EDA script (moved from `scripts/`) |
| `scripts/make_empty_submission.py` | E00 all-empty submission |
| `scripts/make_zip.py` | final submission zip |
| `tests/…` | pytest per module |
| `conftest.py` | pytest root marker |

---

## Phase 0 — Foundation

### Task 1: Package skeleton, config, requirements

**Files:**
- Create: `src/__init__.py`, `conftest.py`, `tests/__init__.py`
- Modify: `src/config.py` (replace whole file)
- Verify exists: `requirements.txt`, `README.md` (written with this plan)
- Move: `scripts/eda.py` → `src/eda.py`

- [ ] **Step 1: Create package files**

`src/__init__.py`:
```python
"""Business entity resolution pipeline (Amazon ML Challenge 2026)."""
```

`conftest.py`:
```python
"""Pytest root marker: makes `src` importable when running `python -m pytest` here."""
```

`tests/__init__.py`: empty file.

- [ ] **Step 2: Replace `src/config.py`**

```python
"""Central configuration: paths, seeds, sampling, blocking, model and decoding hyper-parameters."""
import os
from pathlib import Path

SEED = 42

PKG_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_DIR.parents[1]  # student_resource/
CACHE_DIR = Path(os.environ.get("ER_CACHE", PKG_DIR / "cache"))
# Bump when normalize/blocking/features change so stale parquet caches are ignored.
CACHE_TAG = os.environ.get("ER_CACHE_TAG", "v1")
N_JOBS = int(os.environ.get("ER_JOBS", max(1, (os.cpu_count() or 2) - 2)))

# --- sampling -----------------------------------------------------------------
DEV_FRAC = 0.10    # share of train S1 entities used by --eval-only dev runs
TRAIN_FRAC = 0.15  # share of train S1 entities used to fit models in full runs

# --- blocking -----------------------------------------------------------------
BLOCK_MAX_DF = 2000   # drop keys whose pool document frequency exceeds this
BLOCK_TOPK = 40       # candidates kept per S1 entity
BLOCK_CHUNK = 4000    # S1 rows per sparse-matmul chunk
BLOCK_W = {"n": 1.0, "nb": 1.0, "nm": 0.5, "a": 0.6, "ab": 1.5, "p": 1.0}

# --- model --------------------------------------------------------------------
N_FOLDS = 5
EARLY_STOP = 50
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.08,
    num_leaves=127,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_estimators=2000,
    verbose=-1,
    random_state=SEED,
    n_jobs=N_JOBS,
)

# --- stage 2 ------------------------------------------------------------------
TRIANGLE_TOPM = 5     # supporters considered per S1 group
TRIANGLE_PMIN = 0.01  # candidates below this stage-1 score get no triangle features

# --- decoding -----------------------------------------------------------------
P_MIN = 0.01
EF_DRAWS = 128
EF_KMAX = 12
T_MATCH = 0.5
T_EMPTY = 0.5
```

- [ ] **Step 3: Move EDA into src**

Run: `git mv scripts/eda.py src/eda.py`

- [ ] **Step 4: Install pinned requirements and verify import**

Run: `python -m pip install -r requirements.txt && python -c "import src.config as c; print(c.REPO_ROOT.name, c.N_JOBS)"`
Expected: `student_resource <int>`

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: package skeleton, config, pinned requirements"
```

### Task 2: Metric tests (evaluate.py already written)

**Files:**
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for the exact challenge metric and blocking report."""
import pytest

from src.evaluate import blocking_report, breakdown, f05_entity, score


def test_problem_statement_example():
    assert f05_entity({"a", "b", "x"}, {"a", "b"}) == pytest.approx(0.7142857, abs=1e-6)


def test_singleton_rules():
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity({"x"}, set()) == 0.0
    assert f05_entity(set(), {"a"}) == 0.0
    assert f05_entity({"x"}, {"a"}) == 0.0


def test_macro_average_and_breakdown():
    gt = {"s1": {"a"}, "s2": set()}
    pred = {"s1": ["a"], "s2": ["z"]}
    mean, vals = score(pred, gt)
    assert mean == 0.5 and list(vals) == [1.0, 0.0]
    bd = breakdown(pred, gt, ["s1", "s2"], ["US", "France"])
    assert bd["country=US"] == 1.0 and bd["country=France"] == 0.0
    assert bd["singletons"] == 0.0 and bd["matched"] == 1.0


def test_blocking_report():
    gt = {"s1": {"a", "b"}, "s2": set()}
    rep = blocking_report({"s1": ["a", "c"], "s2": ["d"]}, gt, ["s1", "s2"])
    assert rep["pair_recall"] == 0.5
    assert rep["entity_full_recall"] == 0.5
    assert rep["cands_per_entity"] == 1.5
```

- [ ] **Step 2: Run**

Run: `python -m pytest -q tests/test_evaluate.py`
Expected: `4 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_evaluate.py && git commit -m "test: exact F0.5 metric and blocking report"
```

### Task 3: Writer + official validator round-trip test

**Files:**
- Test: `tests/test_io.py`

- [ ] **Step 1: Write test**

```python
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
```

- [ ] **Step 2: Run**

Run: `python -m pytest -q tests/test_io.py`
Expected: `1 passed` (if it fails, fix `io_utils.write_lists`; never edit the validator).

- [ ] **Step 3: Commit**

```bash
git add tests/test_io.py && git commit -m "test: writer output passes official validator"
```

### Task 4: E00 all-empty submission (format check, expected LB ≈ 0.056)

**Files:**
- Create: `scripts/make_empty_submission.py`

- [ ] **Step 1: Write script**

```python
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
```

- [ ] **Step 2: Run + validate**

Run: `python scripts/make_empty_submission.py`
Expected: `wrote 1,732,544 rows to ...output`
Run (from `student_resource/`): `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
Expected: `PASS`

- [ ] **Step 3: Upload (shift driver only)** `output/matching_results.tsv` to the portal. Record the public LB in `HANDOFF.md` and `experiments/log.csv` (exp_id `E00`). Tag: `git tag sub-d1-01`.

- [ ] **Step 4: Commit**

```bash
git add scripts/make_empty_submission.py experiments/log.csv HANDOFF.md && git commit -m "feat: E00 all-empty submission" && git tag sub-d1-01
```

---

## Phase 1 — Normalization

### Task 5: normalize.py

**Files:**
- Create: `src/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for name/address normalization dictionaries and rules."""
import pandas as pd

from src.normalize import normalize_address, normalize_frame, normalize_name, split_raw


def test_name_legal_split_and_dots():
    norm, core, legal, first = normalize_name("Pvt. EFS Print Ventures Ltd.")
    assert core == "efs print ventures"
    assert legal == "pvt ltd" and first == 1
    assert normalize_name("Green & Auclair L.L.C.")[1:3] == ("green auclair", "llc")
    assert normalize_name("E.U.R.L. Maison")[2] == "eurl"


def test_name_junk_accents_and_web():
    assert normalize_name("-- Holloway Peak Inc Seafood")[1] == "holloway peak seafood"
    assert normalize_name("Norah Pletcher Báin Care")[0] == "norah pletcher bain care"
    assert normalize_name("wilfordhancock.com")[1] == "wilfordhancock"


def test_name_translit_table_used():
    table = {"प्राइवेट": "private", "लिमिटेड": "limited"}
    assert normalize_name("राम प्राइवेट लिमिटेड", table)[2] == "pvt ltd"


def test_address_canon_states_numbers():
    a = normalize_address("3029 Veterans Parkway, City Of Marshfield, Wisconsin")
    b = normalize_address("3029 Veterans Pkwy, City Of Marshfield, WI")
    assert a[0] == b[0]
    addr_norm, alpha, nums, postcode, street, land = a
    assert nums == "3029" and street == "3029_veterans" and postcode == ""
    assert "wi" in alpha.split()


def test_address_french_and_landmark_and_pin():
    assert normalize_address("63 R. DE DIEPPE, LILLE")[0] == normalize_address("63 Rue de Dieppe, Lille")[0]
    a = normalize_address("12 MG Road, Near SBI ATM, Pune 411001")
    assert a[5] == "near sbi atm" and a[3] == "411001" and "sbi" not in a[0]


def test_split_raw_keeps_devanagari_words():
    assert split_raw("मॉडर्न फाउंडेशन") == ["मॉडर्न", "फाउंडेशन"]


def test_normalize_frame_columns():
    df = pd.DataFrame({"entity_id": ["S3-1"], "business_name": ["मॉडर्न X"],
                       "business_address": [""], "country": ["India"]})
    out = normalize_frame(df, None, n_jobs=1)
    assert out.loc[0, "is_s3"] == 1 and out.loc[0, "script_flag"] == 1
    assert out.loc[0, "addr_norm"] == "" and out.loc[0, "rid"] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest -q tests/test_normalize.py`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.normalize'`)

- [ ] **Step 3: Implement `src/normalize.py`**

```python
"""Country-agnostic text normalization for business names and addresses.

Every dictionary in this module is hand-written (no external data) and is listed in
the methodology document. Canonical forms are short abbreviations so that "Street",
"St" and French "Saint" collapse to the same token on both sides of a pair.
"""
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from unidecode import unidecode

LEGAL = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "llc": "llc", "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "llp": "llp", "lp": "lp", "plc": "plc",
    "pllc": "pllc", "pc": "pc", "opc": "opc", "sarl": "sarl", "sas": "sas",
    "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
    "cie": "cie", "compagnie": "cie",
}
NAME_STOP = {"and", "the", "of", "de", "du", "des", "la", "le", "les", "et", "d", "l",
             "www", "com", "net", "org"}
JUNK = {"null", "none", "nan"}

ADDR_CANON = {
    "street": "st", "saint": "st", "road": "rd", "avenue": "av", "ave": "av",
    "boulevard": "bd", "blvd": "bd", "drive": "dr", "lane": "ln", "court": "ct",
    "place": "pl", "parkway": "pkwy", "highway": "hwy", "circle": "cir", "trail": "trl",
    "terrace": "ter", "square": "sq", "apartment": "apt", "suite": "ste", "sainte": "ste",
    "floor": "fl", "building": "bldg", "north": "n", "south": "s", "east": "e", "west": "w",
    "rue": "r", "allee": "all", "impasse": "imp", "chemin": "ch", "route": "rte",
    "faubourg": "fbg", "sector": "sec",
}
ADDR_DROP = {"no", "hno", "h", "door", "unit", "number", "num", "null", "none", "nan", "po", "box"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy",
}

_ABBR_DOT = re.compile(r"(?<=\b[^\W\d_])\.")  # dot after a single letter: L.L.C. -> LLC
_LANDMARK = re.compile(r"^\s*(near|nr|opp|opposite|behind|beside|besides|next to|in front of|adjacent to)\b",
                       re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_NON_LATIN = r"[^\x00-ɏ]"


def _sep_table():
    """Translation table mapping BMP punctuation, symbols, separators and control chars to a space."""
    table = {}
    for cp in range(0x10000):
        cat = unicodedata.category(chr(cp))
        if cat[0] in "PSZ" or cat in ("Cc", "Cf"):
            table[cp] = " "
    return table


_SEP = _sep_table()


def split_raw(text):
    """NFKC + lowercase + punctuation split, before transliteration (native-script tokens kept intact)."""
    if not text:
        return []
    s = unicodedata.normalize("NFKC", text).lower().replace("&", " and ")
    return _ABBR_DOT.sub("", s).translate(_SEP).split()


@lru_cache(maxsize=1 << 20)
def to_ascii(tok):
    """Transliterate one token to lowercase ASCII alphanumerics (may yield several tokens)."""
    return re.sub(r"[^a-z0-9]+", " ", unidecode(tok).lower()).strip()


def tokens(text, table=None):
    """Normalized ASCII tokens: learned transliteration table first, unidecode fallback, junk removed."""
    out = []
    for tok in split_raw(text):
        if not tok.isascii():
            tok = (table.get(tok) if table else None) or to_ascii(tok)
            out.extend(tok.split())
        else:
            out.append(tok)
    return [t for t in out if t not in JUNK]


def normalize_name(text, table=None):
    """Return (name_norm, name_core, legal, legal_first) for a business name."""
    toks = tokens(text, table)
    legal = [LEGAL[t] for t in toks if t in LEGAL]
    core = [t for t in toks if t not in LEGAL and t not in NAME_STOP]
    if not core:
        core = [t for t in toks if t not in LEGAL] or toks
    first = int(bool(toks) and toks[0] in LEGAL)
    return " ".join(toks), " ".join(core), " ".join(dict.fromkeys(legal)), first


def _map_states(toks):
    """Replace US state full names (one or two words) by their 2-letter codes."""
    out, i = [], 0
    while i < len(toks):
        two = " ".join(toks[i:i + 2])
        if i + 1 < len(toks) and two in US_STATES:
            out.append(US_STATES[two])
            i += 2
        else:
            out.append(US_STATES.get(toks[i], toks[i]))
            i += 1
    return out


def normalize_address(text, table=None):
    """Return (addr_norm, addr_alpha, nums, postcode, street_key, landmark) for an address."""
    keep, land = [], []
    for seg in (text or "").split(","):
        (land if _LANDMARK.match(seg) else keep).append(seg)
    toks = [t for t in tokens(" , ".join(keep), table) if t not in ADDR_DROP]
    toks = [ADDR_CANON.get(t, t) for t in _map_states(toks)]
    nums = [d for t in toks for d in _DIGITS.findall(t)]
    alpha = [t for t in toks if not any(c.isdigit() for c in t)]
    postcode = ""
    for i, d in enumerate(nums):
        if len(d) == 6 or (len(d) == 5 and i > 0):
            postcode = d
    street = ""
    for i, t in enumerate(toks):
        if any(c.isdigit() for c in t):
            nxt = next((u for u in toks[i + 1:] if u.isalpha() and len(u) > 1), "")
            if nxt:
                street = f"{_DIGITS.findall(t)[0]}_{nxt}"
                break
    landmark = " ".join(tokens(" ".join(land), table))
    return " ".join(toks), " ".join(alpha), " ".join(nums), postcode, street, landmark


def _norm_chunk(names, addrs, table):
    """Normalize one chunk of names and addresses (runs in a worker process)."""
    return [normalize_name(x, table) for x in names], [normalize_address(x, table) for x in addrs]


def normalize_frame(df, table=None, n_jobs=1, chunk=100_000):
    """Normalize a raw source frame into the `norm` contract (see spec §4); rid = row position."""
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_norm_chunk)(names[i:i + chunk], addrs[i:i + chunk], table)
        for i in range(0, len(names), chunk)
    )
    n = pd.DataFrame([r for p in parts for r in p[0]],
                     columns=["name_norm", "name_core", "legal", "legal_first"])
    a = pd.DataFrame([r for p in parts for r in p[1]],
                     columns=["addr_norm", "addr_alpha", "nums", "postcode", "street_key", "landmark"])
    out = pd.concat([n, a], axis=1)
    out.insert(0, "rid", np.arange(len(df), dtype=np.int64))
    out.insert(1, "entity_id", df["entity_id"].to_numpy())
    out.insert(2, "country", df["country"].to_numpy())
    out["name_raw"] = df["business_name"].to_numpy()
    out["addr_raw"] = df["business_address"].to_numpy()
    out["legal_first"] = out["legal_first"].astype(np.int8)
    out["is_s3"] = df["entity_id"].str.startswith("S3-").to_numpy().astype(np.int8)
    raw = df["business_name"] + " " + df["business_address"]
    out["script_flag"] = raw.str.contains(_NON_LATIN, regex=True).to_numpy().astype(np.int8)
    return out
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/test_normalize.py`
Expected: `7 passed`. If one fails, fix the dictionary/rule (not the test), unless the test contradicts the spec.

- [ ] **Step 5: Commit**

```bash
git add src/normalize.py tests/test_normalize.py && git commit -m "feat: name/address normalization with hand-written dictionaries"
```

### Task 6: translit.py (learned native-script table)

**Files:**
- Create: `src/translit.py`
- Test: `tests/test_translit.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for the learned transliteration table."""
from src.translit import fit_table, load_table, save_table


def test_fit_table_majority_alignment(tmp_path):
    s1 = ["Ram Marketing Private Limited"] * 3 + ["Shyam Traders Private Limited"] * 3
    pool = ["राम मार्केटिंग प्राइवेट लिमिटेड"] * 3 + ["श्याम ट्रेडर्स प्राइवेट लिमिटेड"] * 3
    table = fit_table(s1, pool, min_support=3, min_share=0.6)
    assert table["प्राइवेट"] == "private" and table["लिमिटेड"] == "limited"
    assert table["राम"] == "ram"
    p = tmp_path / "t.tsv"
    save_table(table, p)
    assert load_table(p) == table


def test_fit_table_skips_unequal_lengths():
    assert fit_table(["A B"] * 5, ["राम"] * 5) == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest -q tests/test_translit.py`
Expected: FAIL (`No module named 'src.translit'`)

- [ ] **Step 3: Implement `src/translit.py`**

```python
"""Learned native-script -> Latin token table, fitted on train matched name pairs only.

No external transliteration data is used: the table is estimated from positionally
aligned tokens of (S1 Latin name, pool native-script name) ground-truth pairs.
"""
from collections import Counter, defaultdict

from .normalize import normalize_name, split_raw


def fit_table(s1_names, pool_names, min_support=3, min_share=0.6):
    """Fit {native token: latin token} from matched name pairs with equal token counts."""
    counts = defaultdict(Counter)
    for a, b in zip(s1_names, pool_names):
        if not b or b.isascii():
            continue
        ta = normalize_name(a)[0].split()
        tb = split_raw(b)
        if len(ta) != len(tb):
            continue
        for native, latin in zip(tb, ta):
            if not native.isascii():
                counts[native][latin] += 1
    table = {}
    for tok, c in counts.items():
        best, n = c.most_common(1)[0]
        if n >= min_support and n / sum(c.values()) >= min_share:
            table[tok] = best
    return table


def save_table(table, path):
    """Write the table as a two-column UTF-8 TSV (native, latin)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k in sorted(table):
            f.write(f"{k}\t{table[k]}\n")


def load_table(path):
    """Read a table written by `save_table`."""
    with open(path, encoding="utf-8") as f:
        return dict(line.rstrip("\n").split("\t", 1) for line in f if line.strip())
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/test_translit.py`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/translit.py tests/test_translit.py && git commit -m "feat: learned native-script transliteration table"
```

### Task 7: Normalization smoke on real data

- [ ] **Step 1: Run on a 50k-row slice and eyeball**

Run:
```bash
python -c "
from src import io_utils, normalize, config
df = io_utils.load_tsv(config.REPO_ROOT/'dataset/train/train_source2.tsv').head(50000)
n = normalize.normalize_frame(df, None, n_jobs=4)
print(n.sample(15, random_state=1)[['name_raw','name_core','legal','addr_norm','street_key','postcode']].to_string())
print('script share', n.script_flag.mean(), 'empty core', (n.name_core=='').mean())"
```
Expected: `script share` ≈ 0.15–0.2, `empty core` < 0.001, and readable rows. Add any missing abbreviations you see to the `normalize.py` dictionaries (and a test line), then re-run the tests.

- [ ] **Step 2: Commit** any dictionary additions.

```bash
git add -A && git commit -m "feat: normalization dictionary fixes from real-data smoke"
```

---

## Phase 2 — Blocking

### Task 8: blocking.py

**Files:**
- Create: `src/blocking.py`
- Test: `tests/test_blocking.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for key generation and per-country IDF blocking."""
import pandas as pd

from src.blocking import generate_candidates, record_keys
from src.normalize import normalize_frame


def _frame(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def test_record_keys_families():
    keys = record_keys("peak highland fast", "veterans pkwy marshfield wi", "3029_veterans", "")
    fams = {f for f, _ in keys}
    assert fams == {"n", "nb", "nm", "a", "ab"}
    assert ("nm", "fhp") in keys and ("ab", "3029_veterans") in keys


def test_generate_candidates_finds_match_and_respects_country():
    s1 = normalize_frame(_frame([
        ["S1-1", "Peak Highland Fast Inc", "3029 Veterans Parkway, Marshfield, WI", "US"],
        ["S1-2", "Zephay Labs Inc", "2621 Cotten Road, Tyler, TX", "US"],
        ["S1-3", "Team Ecole", "175 Boulevard Roosevelt, Bordeaux", "France"],
    ]), None)
    pool = normalize_frame(_frame([
        ["S3-9", "Peak Highland Fast", "3029 Veterans Pkwy, Marshfield, Wisconsin", "US"],
        ["S2-7", "Zephay Labs", "2621 Cotten Rd, Tyler, Texas", "US"],
        ["S2-8", "Team Ecole SARL", "175 BD ROOSEVELT, BORDEAUX", "France"],
        ["S2-5", "Team Ecole", "175 Boulevard Roosevelt, Bordeaux", "US"],
    ]), None)
    c = generate_candidates(s1, pool, topk=2, max_df=10, chunk=2, n_jobs=1)
    top = c[c.block_rank == 0].set_index("s1_rid")["pool_rid"].to_dict()
    assert top == {0: 0, 1: 1, 2: 2}
    fr = c[c.s1_rid == 2]
    assert set(fr.pool_rid) == {2}  # US record S2-5 never offered to a France S1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest -q tests/test_blocking.py`
Expected: FAIL (`No module named 'src.blocking'`)

- [ ] **Step 3: Implement `src/blocking.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/test_blocking.py`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/blocking.py tests/test_blocking.py && git commit -m "feat: per-country IDF key blocking via sparse matmul"
```

### Task 9: Pipeline skeleton up to blocking + dev recall run

**Files:**
- Create: `src/run_pipeline.py` (full version; later tasks only use more of it)

- [ ] **Step 1: Write `src/run_pipeline.py`**

```python
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
    generic = get_generic_vocab(s1n, pooln, pos_all)

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
```

Note: `candidate_pairs.tsv` is written from `ft` (the exact rows the model scored), so matches ⊆ candidates by construction.

- [ ] **Step 2: Create stub modules so imports resolve** (replaced in Phase 3/4)

`src/features.py`, `src/noise.py`, `src/model.py`, `src/context.py`, `src/decide.py`: each gets one docstring line for now, e.g. `"""Stage-1 pair features (implemented in Task 11)."""`. Add to `features.py`:
```python
"""Stage-1 pair features (full implementation in Task 11)."""
import numpy as np


def add_frequency_columns(s1n, pooln):
    """Add log pool-frequency of name_core to both frames (how common a business name is)."""
    vc = pooln["name_core"].value_counts()
    pooln["name_freq"] = np.log1p(pooln["name_core"].map(vc).to_numpy(np.float32))
    s1n["name_freq"] = np.log1p(s1n["name_core"].map(vc).fillna(0).to_numpy(np.float32))
```

- [ ] **Step 3: Dev blocking run (laptop or SageMaker)**

Run: `python -m src.run_pipeline --data ../../dataset --eval-only --stop-after blocking --exp-id E01 --change "blocking v1"`
Expected output: `blocking {'block_recall': ..., 'cands_per_entity': ~40, ...}`. The first run normalizes all train files (≈ 5–15 min, cached after that).
If the laptop runs out of memory: run on SageMaker (README §SageMaker).

- [ ] **Step 4: Tune blocking until recall ≥ 0.97** (one change per run, one log row each; delete `cache/*cands_train*` between runs)

Levers, in order: `BLOCK_TOPK` 40→60; `BLOCK_MAX_DF` 2000→5000; `BLOCK_W["a"]` 0.6→1.0; `BLOCK_W["ab"]` 1.5→2.5. Diagnose misses with:
```bash
python -c "
import pandas as pd, numpy as np
from src import run_pipeline as rp, io_utils, config
t = rp.get_translit('../../dataset'); s1, pool = rp.load_split_norm('../../dataset','train',t)
gt = io_utils.load_ground_truth('../../dataset'); pos = rp.gt_rid_pairs(gt, s1, pool)
c = pd.read_parquet(rp.cache_path('cands_train_0.1'))
pos = pos[pos.s1_rid.isin(c.s1_rid.unique())]
miss = pos[~np.isin(rp.pair_key(pos.s1_rid,pos.pool_rid), rp.pair_key(c.s1_rid,c.pool_rid))].sample(20, random_state=0)
for a,b in zip(miss.s1_rid, miss.pool_rid): print(s1.name_raw[a],'|',s1.addr_raw[a],'\n   ->',pool.name_raw[b],'|',pool.addr_raw[b])"
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: pipeline skeleton through blocking; dev recall logged"
```

---

## Phase 3 — Stage-1 model and first real submission

### Task 10: noise.py (differentiator #2)

**Files:**
- Create: `src/noise.py` (replace stub)
- Test: `tests/test_noise.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for noise-channel features."""
from src.noise import NOISE_FEATS, digit_deletion, fit_generic_vocab, noise_row

ROW = ("name_core", "legal", "legal_first", "nums", "postcode", "addr_alpha",
       "name_norm", "name_raw", "addr_norm", "addr_raw")


def _row(**kw):
    base = dict(name_core="", legal="", legal_first=0, nums="", postcode="", addr_alpha="",
                name_norm="", name_raw="", addr_norm="", addr_raw="")
    base.update(kw)
    return tuple(base[k] for k in ROW)


def test_digit_deletion():
    assert digit_deletion("3424", "424") == 1
    assert digit_deletion("1201", "120") == 1
    assert digit_deletion("136", "36") == 1
    assert digit_deletion("3424", "3424") == 0
    assert digit_deletion("3424", "12") == 0


def test_generic_vocab():
    s1 = ["davis society care", "peak fast", "blue ocean"] * 100
    pool = ["davis society", "peak fast services", "blue ocean"] * 100
    v = fit_generic_vocab(s1, pool, min_count=50, min_share=0.5)
    assert v == {"care", "services"}


def test_noise_row_values():
    a = _row(name_core="valley holdings", legal="inc", legal_first=0, nums="719",
             addr_alpha="mountain rd addison vt", name_norm="valley holdings inc",
             name_raw="Valley Holdings Inc", addr_norm="719 mountain rd addison vt", addr_raw="719 Mountain Road")
    b = _row(name_core="vlley holdings services", legal="inc", legal_first=1, nums="19",
             addr_alpha="mountain rd addison vt", name_norm="inc vlley holdings services",
             name_raw="Inc Vlley Holdings Services", addr_norm="19 mountain rd addison vt", addr_raw="19 Mountain Rd")
    vals = dict(zip(NOISE_FEATS, noise_row(a, b, {"services"})))
    assert vals["f_nz_num_digit_del"] == 1
    assert vals["f_nz_legal_moved"] == 1
    assert vals["f_nz_edit1_tokens"] == 1
    assert vals["f_nz_name_generic_only"] == 0  # 'vlley'/'valley' also differ
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest -q tests/test_noise.py`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement `src/noise.py`**

```python
"""Noise-channel features (differentiator #2): explain *why* two records differ.

Each feature detects one perturbation operator observed in the training pairs
(house-number digit deletion, generic word add/drop, legal suffix moved, one-edit
typos, accent-only or abbreviation-only differences, trade-name swaps).
Row tuples follow ROW_COLS order (see features.ROW_COLS).
"""
from collections import Counter

from rapidfuzz.distance import DamerauLevenshtein

NOISE_FEATS = [
    "f_nz_num_digit_del", "f_nz_num_missing_one", "f_nz_name_generic_only", "f_nz_name_core_equal",
    "f_nz_legal_moved", "f_nz_edit1_tokens", "f_nz_addr_only", "f_nz_accent_only", "f_nz_addr_norm_only",
]


def digit_deletion(a, b):
    """1 if the shorter number string equals the longer with exactly one digit removed."""
    if abs(len(a) - len(b)) != 1:
        return 0
    long_, short = (a, b) if len(a) > len(b) else (b, a)
    return int(any(long_[:i] + long_[i + 1:] == short for i in range(len(long_))))


def fit_generic_vocab(s1_cores, pool_cores, min_count=100, min_share=0.5):
    """Learn 'generic add-on' tokens: tokens that mostly occur in the symmetric difference of matched names."""
    diff, tot = Counter(), Counter()
    for a, b in zip(s1_cores, pool_cores):
        sa, sb = set(a.split()), set(b.split())
        tot.update(sa | sb)
        diff.update(sa ^ sb)
    return {t for t, n in diff.items() if n >= min_count and n / tot[t] >= min_share}


def _jac(a, b):
    """Jaccard similarity of two sets (0 when both empty)."""
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _edit1(ca, cb):
    """Count tokens of ca-cb (len >= 3) within Damerau-Levenshtein distance 1 of a token in cb-ca."""
    only_b = [u for u in cb - ca if len(u) >= 3]
    return sum(1 for t in ca - cb if len(t) >= 3
               and any(DamerauLevenshtein.distance(t, u, score_cutoff=1) <= 1 for u in only_b))


def noise_row(a, b, generic):
    """Noise-channel feature values for one pair of ROW_COLS tuples (order = NOISE_FEATS)."""
    ca, cb = set(a[0].split()), set(b[0].split())
    na, nb = a[3].split(), b[3].split()
    diff = ca ^ cb
    return [
        digit_deletion(na[0], nb[0]) if na and nb else 0,
        int(bool(na) != bool(nb)),
        int(bool(diff) and diff <= generic),
        int(bool(ca) and ca == cb),
        int(bool(a[1]) and a[1] == b[1] and a[2] != b[2]),
        _edit1(ca, cb),
        int(not (ca & cb) and _jac(set(a[5].split()), set(b[5].split())) >= 0.6),
        int(a[6] == b[6] and a[7].lower() != b[7].lower()),
        int(bool(a[8]) and a[8] == b[8] and a[9].lower() != b[9].lower()),
    ]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/test_noise.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/noise.py tests/test_noise.py && git commit -m "feat: noise-channel features and learned generic vocabulary"
```

### Task 11: features.py

**Files:**
- Modify: `src/features.py` (replace stub; keep `add_frequency_columns`)
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for stage-1 pair features."""
import numpy as np
import pandas as pd

from src.features import add_frequency_columns, build_features
from src.normalize import normalize_frame


def _f(rows):
    return normalize_frame(pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]))


def test_build_features_shapes_and_values():
    s1 = _f([["S1-1", "Valley Holdings Inc", "719 Mountain Road, Addison, VT", "US"]])
    pool = _f([["S3-1", "Vlley Holdings", "719 Mountain Rd, Addison, Vermont", "US"],
               ["S2-2", "Other Thing LLC", "1 Main St, Austin, TX", "US"]])
    add_frequency_columns(s1, pool)
    cands = pd.DataFrame({"s1_rid": [0, 0], "pool_rid": [0, 1],
                          "block_score": np.float32([9, 1]), "block_rank": np.int16([0, 1])})
    f = build_features(cands, s1, pool, set(), n_jobs=1)
    assert list(f["pool_rid"]) == [0, 1]
    assert f.loc[0, "f_addr_jac"] == 1.0 and f.loc[0, "f_num1_eq"] == 1.0
    assert f.loc[0, "f_nz_edit1_tokens"] == 1 and f.loc[0, "f_is_s3"] == 1
    assert f.loc[0, "f_name_core_tset"] > f.loc[1, "f_name_core_tset"]
    assert not any("country" in c for c in f.columns)
    assert all(f[c].dtype == np.float32 for c in f.columns if c.startswith("f_"))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest -q tests/test_features.py`
Expected: FAIL (`ImportError: cannot import name 'build_features'`)

- [ ] **Step 3: Implement `src/features.py`**

```python
"""Stage-1 pair features: string similarities, set overlaps, record statistics and noise features.

Country is deliberately absent: every feature is computed identically for any label.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from . import config
from .noise import NOISE_FEATS, noise_row

ROW_COLS = ["name_core", "legal", "legal_first", "nums", "postcode", "addr_alpha",
            "name_norm", "name_raw", "addr_norm", "addr_raw"]
FUZZ = [
    ("name_norm", "ratio", fuzz.ratio), ("name_norm", "tset", fuzz.token_set_ratio),
    ("name_norm", "tsort", fuzz.token_sort_ratio), ("name_norm", "partial", fuzz.partial_ratio),
    ("name_norm", "jw", JaroWinkler.normalized_similarity),
    ("name_core", "ratio", fuzz.ratio), ("name_core", "tset", fuzz.token_set_ratio),
    ("name_core", "jw", JaroWinkler.normalized_similarity),
    ("addr_norm", "ratio", fuzz.ratio), ("addr_norm", "tset", fuzz.token_set_ratio),
    ("addr_norm", "partial", fuzz.partial_ratio), ("addr_alpha", "tset", fuzz.token_set_ratio),
]
PY_FEATS = ["f_name_jac", "f_addr_jac", "f_nums_jac", "f_num1_eq", "f_pc_eq", "f_pc_missing",
            "f_legal_code", "f_core_len_diff"] + NOISE_FEATS


def add_frequency_columns(s1n, pooln):
    """Add log pool-frequency of name_core to both frames (how common a business name is)."""
    vc = pooln["name_core"].value_counts()
    pooln["name_freq"] = np.log1p(pooln["name_core"].map(vc).to_numpy(np.float32))
    s1n["name_freq"] = np.log1p(s1n["name_core"].map(vc).fillna(0).to_numpy(np.float32))


def _jac(a, b):
    """Jaccard similarity of two sets (0 when both empty)."""
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _legal_code(a, b):
    """0 both empty, 1 equal, 2 exactly one empty, 3 both present but different."""
    if not a and not b:
        return 0
    if a == b:
        return 1
    return 2 if (not a or not b) else 3


def _py_chunk(A, B, generic):
    """Set/number/legal features plus noise features for a chunk of row-tuple pairs."""
    out = np.zeros((len(A), len(PY_FEATS)), np.float32)
    for i, (a, b) in enumerate(zip(A, B)):
        ca, cb = set(a[0].split()), set(b[0].split())
        na, nb = a[3].split(), b[3].split()
        out[i, 0] = _jac(ca, cb)
        out[i, 1] = _jac(set(a[5].split()), set(b[5].split()))
        out[i, 2] = _jac(set(na), set(nb))
        out[i, 3] = float(na[0] == nb[0]) if (na and nb) else -1.0
        out[i, 4] = float(bool(a[4]) and a[4] == b[4])
        out[i, 5] = float(not a[4] or not b[4])
        out[i, 6] = _legal_code(a[1], b[1])
        out[i, 7] = abs(len(ca) - len(cb))
        out[i, 8:] = noise_row(a, b, generic)
    return out


def _take(frame, col, idx):
    """Positional gather of one column as a Python list."""
    return frame[col].take(idx).tolist()


def _features_chunk(c, s1n, pooln, generic, n_jobs):
    """All stage-1 features for one chunk of candidate pairs."""
    ia, ib = c["s1_rid"].to_numpy(), c["pool_rid"].to_numpy()
    out = {"s1_rid": ia, "pool_rid": ib}
    for col, name, scorer in FUZZ:
        out[f"f_{col}_{name}"] = process.cpdist(_take(s1n, col, ia), _take(pooln, col, ib), scorer=scorer,
                                                workers=n_jobs, dtype=np.float32)
    A = list(zip(*[_take(s1n, k, ia) for k in ROW_COLS]))
    B = list(zip(*[_take(pooln, k, ib) for k in ROW_COLS]))
    step = max(1, len(A) // (n_jobs * 4) + 1)
    parts = Parallel(n_jobs=n_jobs)(delayed(_py_chunk)(A[i:i + step], B[i:i + step], generic)
                                    for i in range(0, len(A), step))
    py = np.vstack(parts) if parts else np.zeros((0, len(PY_FEATS)), np.float32)
    for j, name in enumerate(PY_FEATS):
        out[name] = py[:, j]
    out["f_block_score"] = c["block_score"].to_numpy(np.float32)
    out["f_block_rank"] = c["block_rank"].to_numpy(np.float32)
    out["f_is_s3"] = pooln["is_s3"].to_numpy()[ib].astype(np.float32)
    for side, frame, idx in (("a", s1n, ia), ("b", pooln, ib)):
        out[f"f_freq_{side}"] = frame["name_freq"].to_numpy()[idx].astype(np.float32)
        out[f"f_script_{side}"] = frame["script_flag"].to_numpy()[idx].astype(np.float32)
        out[f"f_addr_empty_{side}"] = (frame["addr_norm"].take(idx) == "").to_numpy().astype(np.float32)
        out[f"f_land_{side}"] = (frame["landmark"].take(idx) != "").to_numpy().astype(np.float32)
        out[f"f_ntok_{side}"] = frame["name_core"].take(idx).str.count(" ").to_numpy().astype(np.float32) + 1
    return pd.DataFrame(out)


def build_features(cands, s1n, pooln, generic, n_jobs=None, chunk=2_000_000):
    """Stage-1 features for every candidate pair, in candidate order (float32 f_* columns)."""
    n_jobs = n_jobs or config.N_JOBS
    parts = [_features_chunk(cands.iloc[i:i + chunk], s1n, pooln, generic, n_jobs)
             for i in range(0, len(cands), chunk)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/test_features.py`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py && git commit -m "feat: stage-1 pair features"
```

### Task 12: model.py

**Files:**
- Modify: `src/model.py` (replace stub)
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for GroupKFold LightGBM training."""
import numpy as np
import pandas as pd

from src.model import feature_cols, fit_full, mean_best_iter, train_oof


def test_train_oof_groups_and_signal():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"f_a": rng.random(n).astype(np.float32), "f_b": rng.random(n).astype(np.float32),
                      "s1_rid": np.repeat(np.arange(n // 10), 10)})
    y = (X["f_a"] > 0.7).astype(np.int8).to_numpy()
    cols = feature_cols(X)
    assert cols == ["f_a", "f_b"]
    oof, models = train_oof(X[cols], y, X["s1_rid"].to_numpy(),
                            params=dict(objective="binary", n_estimators=50, verbose=-1, random_state=0), n_folds=3)
    assert len(models) == 3 and oof.shape == (n,)
    assert ((oof > 0.5) == y).mean() > 0.95
    m = fit_full(X[cols], y, n_estimators=mean_best_iter(models))
    assert m.predict_proba(X[cols])[:, 1].shape == (n,)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest -q tests/test_model.py`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement `src/model.py`**

```python
"""LightGBM models: GroupKFold out-of-fold training (grouped by S1 entity) and full refits."""
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from . import config


def feature_cols(df, prefix="f_"):
    """Model input columns: those starting with `prefix` (never ids, labels or country)."""
    return [c for c in df.columns if c.startswith(prefix)]


def train_oof(X, y, groups, params=None, n_folds=None):
    """Fit one LightGBM per GroupKFold fold; return (oof_pred float32, fitted models)."""
    params = dict(config.LGB_PARAMS if params is None else params)
    n_folds = n_folds or config.N_FOLDS
    y = np.asarray(y)
    oof = np.zeros(len(y), np.float32)
    models = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(config.EARLY_STOP, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
        print(f"  fold {k}: best_iter={m.best_iteration_}", flush=True)
    return oof, models


def mean_best_iter(models):
    """Average best iteration across fold models (+10% for the larger full-data refit)."""
    its = [m.best_iteration_ or m.n_estimators for m in models]
    return max(10, int(np.mean(its) * 1.1))


def fit_full(X, y, n_estimators, params=None):
    """Refit on all rows with a fixed number of trees (used for test inference and transfer)."""
    p = dict(config.LGB_PARAMS if params is None else params)
    p["n_estimators"] = n_estimators
    m = lgb.LGBMClassifier(**p)
    m.fit(X, np.asarray(y))
    return m
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/test_model.py`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py && git commit -m "feat: GroupKFold LightGBM OOF and full refit"
```

### Task 13: decide.py (calibration, one-to-one, threshold + expected-F0.5 decoders, tuning)

**Files:**
- Modify: `src/decide.py` (replace stub)
- Test: `tests/test_decide.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for the decision layer."""
import numpy as np
import pandas as pd

from src.decide import apply_decoder, one_to_one, select_expected_f05, select_threshold


def _df(rows):
    return pd.DataFrame(rows, columns=["s1_rid", "pool_rid", "p"])


def test_one_to_one_keeps_best_group():
    d = one_to_one(_df([[0, 7, 0.9], [1, 7, 0.6], [1, 8, 0.8]]))
    assert sorted(map(tuple, d[["s1_rid", "pool_rid"]].to_numpy())) == [(0, 7), (1, 8)]


def test_threshold_with_empty_gate():
    d = _df([[0, 1, 0.9], [0, 2, 0.4], [1, 3, 0.45]])
    s = select_threshold(d, t_match=0.4, t_empty=0.5)
    assert sorted(s["pool_rid"]) == [1, 2]


def test_expected_f05_prefers_confident_sets_and_empty():
    d = _df([[0, 1, 0.97], [0, 2, 0.95], [0, 3, 0.05], [1, 4, 0.08], [2, 5, 0.9]])
    s = select_expected_f05(d, extra=0.0, p_min=0.01, kmax=12, n_draws=256, seed=0)
    assert sorted(s["pool_rid"]) == [1, 2, 5]


def test_apply_decoder_threshold_roundtrip():
    d = _df([[0, 1, 0.9], [1, 1, 0.8], [1, 2, 0.2]])
    s = apply_decoder(d, {"method": "threshold", "t_match": 0.5, "t_empty": 0.5, "tau": 1.0}, None, 0.0)
    assert list(map(tuple, s[["s1_rid", "pool_rid"]].to_numpy())) == [(0, 1)]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest -q tests/test_decide.py`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement `src/decide.py`**

```python
"""Decision layer (differentiator #1b): calibration, global one-to-one, expected-F0.5 subset selection.

Per S1 entity the metric is F0.5 = 1.25*TP / (k + 0.25*G). Instead of a fixed threshold we pick,
for every entity, the k (possibly 0) that maximises the expected F0.5 under the calibrated
probabilities. A thresholded decoder is kept as the baseline; tuning keeps whichever scores higher.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config


def fit_calibrator(p, y):
    """Isotonic calibration of OOF scores against labels."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(p, np.float64), np.asarray(y, np.float64))
    return iso


def temper(p, tau):
    """Sharpen (tau < 1) or soften (tau > 1) probabilities in logit space."""
    if tau == 1.0:
        return p
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / tau))


def one_to_one(df):
    """Keep each pool record only in the S1 group where its probability is highest."""
    if len(df) == 0:
        return df
    return df.loc[df.groupby("pool_rid")["p"].idxmax()]


def select_threshold(df, t_match, t_empty):
    """Baseline: keep p >= t_match, but predict empty when the group's best p < t_empty."""
    best = df.groupby("s1_rid")["p"].transform("max")
    return df[(df["p"] >= t_match) & (best >= t_empty)]


def select_expected_f05(df, extra, p_min=None, kmax=None, n_draws=None, seed=None, batch=50_000):
    """Per S1 group choose top-k (k >= 0) maximising Monte-Carlo expected F0.5.

    `extra` = expected true matches outside the scored candidates (missed by blocking).
    """
    p_min = config.P_MIN if p_min is None else p_min
    kmax = kmax or config.EF_KMAX
    n_draws = n_draws or config.EF_DRAWS
    seed = config.SEED if seed is None else seed
    tot = df.groupby("s1_rid")["p"].sum()
    d = df[df["p"] >= p_min].sort_values(["s1_rid", "p"], ascending=[True, False])
    pos = d.groupby("s1_rid").cumcount().to_numpy()
    d, pos = d[pos < kmax], pos[pos < kmax]
    if len(d) == 0:
        return d[["s1_rid", "pool_rid", "p"]]
    gid, groups = pd.factorize(d["s1_rid"], sort=False)
    P = np.zeros((len(groups), kmax), np.float32)
    P[gid, pos] = d["p"].to_numpy(np.float32)
    ext = np.maximum(tot.reindex(groups).to_numpy(np.float32) - P.sum(1), 0) + np.float32(extra)
    rng = np.random.default_rng(seed)
    kk = np.arange(1, kmax + 1, dtype=np.float32)
    ks = np.zeros(len(groups), np.int16)
    for s in range(0, len(groups), batch):
        pb, eb = P[s:s + batch], ext[s:s + batch]
        hits = rng.random((len(pb), n_draws, kmax), dtype=np.float32) < pb[:, None, :]
        tp = np.cumsum(hits, axis=2, dtype=np.float32)
        g = tp[:, :, -1] + eb[:, None]
        ef = (1.25 * tp / (kk + 0.25 * g[:, :, None])).mean(1)
        ef[pb == 0] = -1.0
        e0 = np.prod(1 - pb, axis=1) * np.exp(-eb)
        ks[s:s + batch] = np.where(e0 >= ef.max(1), 0, ef.argmax(1) + 1)
    return d[pos < ks[gid]][["s1_rid", "pool_rid", "p"]]


def apply_decoder(df, dec, calibrator, extra):
    """Calibrate -> temper -> global one-to-one -> chosen subset selection."""
    d = df[["s1_rid", "pool_rid", "p"]].copy()
    if calibrator is not None:
        d["p"] = calibrator.predict(d["p"].to_numpy(np.float64)).astype(np.float32)
    d["p"] = temper(d["p"].to_numpy(), dec.get("tau", 1.0))
    d = one_to_one(d)
    if dec["method"] == "threshold":
        return select_threshold(d, dec["t_match"], dec["t_empty"])
    return select_expected_f05(d, extra * dec.get("lam", 1.0))


def tune(df, calibrator, extra, scorer):
    """Grid-search both decoders on OOF scores with the exact metric; return (best params, all results)."""
    grid = []
    for tm in (0.3, 0.4, 0.5, 0.6, 0.7):
        for te in (0.3, 0.4, 0.5, 0.6, 0.7):
            grid.append({"method": "threshold", "t_match": tm, "t_empty": te, "tau": 1.0})
    for tau in (0.8, 1.0, 1.25):
        for lam in (0.5, 1.0, 2.0):
            grid.append({"method": "ef05", "tau": tau, "lam": lam})
    results = []
    for dec in grid:
        s = scorer(apply_decoder(df, dec, calibrator, extra))
        results.append((s, dec))
        print(f"  {dec} -> {s:.5f}", flush=True)
    best = max(results, key=lambda r: r[0])
    return dict(best[1], oof_f05=round(best[0], 5)), results
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/test_decide.py`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/decide.py tests/test_decide.py && git commit -m "feat: decision layer with expected-F0.5 decoder"
```

### Task 14: Stage-1 dev evaluation (stage 2 off)

- [ ] **Step 1: Make `context.py` importable** (stub until Task 16):
```python
"""Stage-2 context and triangle features (implemented in Task 16)."""
```

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest -q tests`
Expected: all pass.

- [ ] **Step 3: Dev OOF run**

Run: `python -m src.run_pipeline --data ../../dataset --eval-only --no-stage2 --exp-id E02 --change "stage1 lgbm + tuned decoder"`
Expected: the log prints `decoder {...}`, `OOF breakdown {'all': ...}` and `transfer {'US->India': ..., 'India->US': ...}`, and appends a row to `experiments/log.csv`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "exp: E02 stage-1 dev OOF"
```

### Task 15: First real submission

- [ ] **Step 1: Full run** (SageMaker recommended; laptop only if memory allows)

Run: `python -m src.run_pipeline --data ../../dataset --out ../../output --no-stage2 --exp-id E03 --change "first full stage1 submission"`
Expected: ends with `validator exit=0`.

- [ ] **Step 2: Upload** `output/matching_results.tsv` (shift driver only). Record the public LB in `experiments/log.csv` (`public_lb` column for E03) and `HANDOFF.md`.

- [ ] **Step 3: Commit + tag**

```bash
git add -A && git commit -m "sub: E03 stage-1 full submission" && git tag sub-d1-02
```

---

## Phase 4 — Differentiators

### Task 16: context.py (stage-2 group, competition, triangle support)

**Files:**
- Modify: `src/context.py` (replace stub)
- Test: `tests/test_context.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for stage-2 context and triangle features."""
import numpy as np
import pandas as pd

from src.context import add_stage2_features
from src.normalize import normalize_frame


def test_stage2_features():
    pool = normalize_frame(pd.DataFrame([
        ["S2-1", "Blue Ocean Ltd", "5 Sea Rd, Goa", "India"],
        ["S3-1", "Blue Ocean", "5 Sea Road, Goa", "India"],
        ["S3-2", "Red Rock", "9 Hill Rd, Pune", "India"],
    ], columns=["entity_id", "business_name", "business_address", "country"]))
    feats = pd.DataFrame({"s1_rid": [0, 0, 0, 1], "pool_rid": [0, 1, 2, 2],
                          "p1": np.float32([0.9, 0.6, 0.3, 0.8])})
    out = add_stage2_features(feats, pool, topm=5, pmin=0.01)
    assert list(out["pool_rid"]) == [0, 1, 2, 2]
    r = out.set_index(["s1_rid", "pool_rid"])
    assert r.loc[(0, 0), "f2_rank"] == 1 and r.loc[(0, 2), "f2_rank"] == 3
    assert r.loc[(0, 2), "f2_best_other"] == np.float32(0.8)
    assert r.loc[(0, 1), "f2_support"] > r.loc[(0, 2), "f2_support"]
    assert r.loc[(0, 1), "f2_sim_top_other_src"] > 0.8
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest -q tests/test_context.py`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement `src/context.py`**

```python
"""Stage-2 features from stage-1 scores (differentiator #1a).

Group context: how a candidate ranks inside its S1 group.
Competition: how strongly the same pool record is claimed by another S1 (one-to-one signal).
Triangle support: true S2 and S3 matches of one entity also resemble each other, so a
candidate similar to the group's other high-scoring candidates gets support.
On train these are built from OUT-OF-FOLD p1 only.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from . import config


def group_context(df):
    """Rank, gap to max, confident count and size inside each S1 group; competition across S1s."""
    g = df.groupby("s1_rid")["p1"]
    df["f2_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    df["f2_gap_max"] = (g.transform("max") - df["p1"]).astype(np.float32)
    df["f2_n_hi"] = (df["p1"] > 0.5).groupby(df["s1_rid"]).transform("sum").astype(np.float32)
    df["f2_gsize"] = g.transform("size").astype(np.float32)
    o = df[["pool_rid", "p1"]].sort_values(["pool_rid", "p1"], ascending=[True, False])
    r = o.groupby("pool_rid").cumcount()
    first = o.loc[r == 0].set_index("pool_rid")["p1"]
    second = o.loc[r == 1].set_index("pool_rid")["p1"]
    is_top = pd.Series(False, index=df.index)
    is_top.loc[o.index[(r == 0).to_numpy()]] = True
    f = first.reindex(df["pool_rid"]).to_numpy()
    s = second.reindex(df["pool_rid"]).fillna(0).to_numpy()
    df["f2_best_other"] = np.where(is_top.to_numpy(), s, f).astype(np.float32)
    df["f2_gap_other"] = (df["p1"] - df["f2_best_other"]).astype(np.float32)
    df["f2_n_claims"] = df.groupby("pool_rid")["p1"].transform("size").astype(np.float32)
    return df


def triangle_features(df, pooln, topm, pmin):
    """Support of each candidate from the group's top-m other candidates (name+address similarity)."""
    d = df[["s1_rid", "pool_rid", "p1"]]
    top = d.sort_values(["s1_rid", "p1"], ascending=[True, False])
    top = top[top.groupby("s1_rid").cumcount() < topm].rename(columns={"pool_rid": "j", "p1": "pj"})
    act = d.loc[d["p1"] >= pmin, ["s1_rid", "pool_rid"]].rename(columns={"pool_rid": "i"})
    pr = act.merge(top, on="s1_rid")
    pr = pr[pr["i"] != pr["j"]].reset_index(drop=True)
    cols = ["f2_support", "f2_n_supporters", "f2_max_sim_top", "f2_sim_top_other_src"]
    if len(pr) == 0:
        return df.assign(**{c: np.float32(0) for c in cols})
    i, j = pr["i"].to_numpy(), pr["j"].to_numpy()
    sim = (process.cpdist(pooln["name_norm"].take(i).tolist(), pooln["name_norm"].take(j).tolist(),
                          scorer=fuzz.token_set_ratio, workers=config.N_JOBS, dtype=np.float32)
           + process.cpdist(pooln["addr_norm"].take(i).tolist(), pooln["addr_norm"].take(j).tolist(),
                            scorer=fuzz.token_set_ratio, workers=config.N_JOBS, dtype=np.float32)) / 200.0
    s3 = pooln["is_s3"].to_numpy()
    pr["sim"] = sim
    pr["w"] = pr["pj"].to_numpy() * sim
    pr["sup"] = ((pr["pj"] > 0.5) & (pr["sim"] > 0.8)).astype(np.float32)
    g = pr.groupby(["s1_rid", "i"])
    agg = pd.DataFrame({"f2_support": g["w"].sum(), "f2_n_supporters": g["sup"].sum(),
                        "f2_max_sim_top": g["sim"].max()})
    other = pr[s3[i] != s3[j]].sort_values(["s1_rid", "i", "pj"], ascending=[True, True, False])
    first_other = other.drop_duplicates(["s1_rid", "i"]).set_index(["s1_rid", "i"])["sim"]
    agg["f2_sim_top_other_src"] = first_other.reindex(agg.index).fillna(-1.0)
    agg = agg.reset_index().rename(columns={"i": "pool_rid"})
    out = df.merge(agg, on=["s1_rid", "pool_rid"], how="left")
    out["f2_sim_top_other_src"] = out["f2_sim_top_other_src"].fillna(-1.0)
    out[cols] = out[cols].fillna(0.0).astype(np.float32)
    return out


def add_stage2_features(df, pooln, topm=None, pmin=None):
    """All stage-2 features; preserves row order of `df` (which must contain p1)."""
    topm = topm or config.TRIANGLE_TOPM
    pmin = config.TRIANGLE_PMIN if pmin is None else pmin
    df = group_context(df.reset_index(drop=True))
    return triangle_features(df, pooln, topm, pmin)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/test_context.py`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/context.py tests/test_context.py && git commit -m "feat: stage-2 context, competition and triangle-support features"
```

### Task 17: Stage-2 dev evaluation vs stage-1

- [ ] **Step 1: Run**

Run: `python -m src.run_pipeline --data ../../dataset --eval-only --exp-id E04 --change "stage2 context+triangle"`
Expected: a log row for E04. Compare it to E02: keep stage 2 only if `f05_all` improves **and** neither transfer column drops by more than 0.002.

- [ ] **Step 2: Check which decoder won** (`decoder {...}` line). Record in `HANDOFF.md`: ef05 vs threshold score, best tau/lam.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "exp: E04 stage-2 dev OOF"
```

### Task 18: Day-2 iteration loop (repeat per idea; one change per experiment)

Each iteration: change one thing → dev run with a new `--exp-id` → keep or revert → commit. Candidate ideas in priority order (each is a concrete change):
1. `BLOCK_TOPK` 40 → 30 if recall barely changes (faster, tighter `candidate_pairs.tsv`).
2. `LGB_PARAMS["num_leaves"]` 127 → 255, `min_child_samples` 50 → 100.
3. `TRIANGLE_TOPM` 5 → 8.
4. Extend `decide.tune` grid: `t_match` in 0.35–0.65 by 0.05 around the best.
5. Normalization dictionary additions from error analysis (below). Bump `ER_CACHE_TAG` (e.g. `v2`) so caches rebuild.

Error analysis command (run after any dev run; prints the worst entities):
```bash
python -c "
import json, numpy as np, pandas as pd
from src import run_pipeline as rp
print(open('../../experiments/log.csv').read()[-800:])"
```
For deeper analysis, load `cache/<tag>_feats1_train_0.1.parquet`, join it with the norm frames and inspect false positives (`p` high, label 0) and false negatives. Write the findings into `HANDOFF.md` → they feed doc §5.

- [ ] **Full-run + upload** only when the dev F0.5 gain ≥ 0.002 (or to test France behaviour). Tag `sub-d2-NN`.

---

## Phase 5 — Packaging

### Task 19: make_zip.py

**Files:**
- Create: `scripts/make_zip.py`

- [ ] **Step 1: Write script**

```python
"""Build <team>_submission.zip with the exact structure required by the problem statement.

Layout:
  output/{matching_results.tsv,candidate_pairs.tsv}
  code/business_entity_resolution/{src/,README.md,requirements.txt}
  Documentation_template.md
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # student_resource/
PKG = ROOT / "code" / "business_entity_resolution"


def main():
    """Create the zip in student_resource/ and list its contents."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    args = ap.parse_args()
    z = ROOT / f"{args.team}_submission.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            zf.write(ROOT / "output" / f, f"output/{f}")
        for p in sorted((PKG / "src").rglob("*.py")):
            zf.write(p, f"code/business_entity_resolution/src/{p.relative_to(PKG / 'src').as_posix()}")
        for f in ("README.md", "requirements.txt"):
            zf.write(PKG / f, f"code/business_entity_resolution/{f}")
        zf.write(ROOT / "Documentation_template.md", "Documentation_template.md")
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            print(n)
    print(f"-> {z} ({z.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run** `python scripts/make_zip.py --team <TEAM>`. Expected: a listing that shows exactly the layout in the docstring.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_zip.py && git commit -m "feat: submission zip builder"
```

### Task 20: Methodology doc (`Documentation_template.md`)

- [ ] **Step 1: Fill every template section**, keeping the core (sections 1–6) ≤ 2 pages. Source for each section:
  - §1 Executive summary: the pipeline + the two differentiators, one sentence each.
  - §2.1: EDA table from spec §2.
  - §2.2: Approach type = "Blocking + Classifier + collective metric-aware decoding". Core innovation = triangle support + expected-F0.5 decoding + noise-channel features.
  - §3: key families + weights (`config.BLOCK_W`), `BLOCK_MAX_DF`, `BLOCK_TOPK`; total candidate pairs on test (`wc -l`-style count from the run log); final dev blocking recall (log.csv).
  - §4: feature groups from `features.PY_FEATS`, `FUZZ`, `noise.NOISE_FEATS`, `f2_*`; model = LightGBM two-stage; threshold selection = expected-F0.5 vs threshold grid on OOF (winning params from `cache/<tag>_decoder.json`).
  - §5: best dev OOF breakdown + transfer numbers (log.csv); FP/FN patterns from HANDOFF.md.
  - §6: conclusion.
  - Appendix A: module map (File map table above) + reproduce command. Appendix B: full `experiments/log.csv` table, the hand-written dictionaries (list names: `LEGAL`, `NAME_STOP`, `ADDR_CANON`, `ADDR_DROP`, `US_STATES`), and a description of the learned artifacts (translit table, generic vocabulary), stating both are fitted on train only.
- [ ] **Step 2: Commit** `git add Documentation_template.md && git commit -m "docs: methodology"`

### Task 21: Final selection and zip (D3 18:00–23:00 IST)

- [ ] Choose the final submission by dev OOF + transfer (not public LB). Re-run the full pipeline for that commit if `output/` is not from it.
- [ ] Validator `PASS`. Upload the final `matching_results.tsv`. Tag `sub-d3-final`.
- [ ] `python scripts/make_zip.py --team <TEAM>`, unzip into a temp dir and check the layout. Submit the zip.
- [ ] Update `HANDOFF.md` with the final state.
