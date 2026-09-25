# Business Entity Resolution — Amazon ML Challenge 2026

For every Source-1 business, find all Source-2/Source-3 records of the same business.
The pipeline is: normalize → per-country IDF key blocking → LightGBM pair model (two stages) →
global one-to-one + expected-F0.5 decoding. Methodology: `Documentation_template.md`
(zip root).

## Environment
- Python ≥ 3.11 (pandas 3). CPU only. No network access is used by the pipeline.
- Full runs: ~100 GB RAM recommended (SageMaker `ml.r5.4xlarge`). The dev loop (`--eval-only`)
  runs on a 16 GB laptop.

```bash
cd code/business_entity_resolution
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data layout (relative to this folder)
```
../../dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
../../dataset/test/test_source{1,2,3}.tsv
```

## Reproduce end-to-end
```bash
# 1. tests
python -m pytest -q tests

# 2. local evaluation only (10% of train S1, 5-fold OOF, prints breakdown + transfer)
python -m src.run_pipeline --data ../../dataset --eval-only

# 3. full: train on train split, predict test, write + validate outputs
python -m src.run_pipeline --data ../../dataset --out ../../output
```
Step 3 writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs
`utils/validate_submission.py`. `candidate_pairs.tsv` is exactly the set of pairs the model scored.

Useful flags: `--stop-after blocking|stage1`, `--no-stage2`, `--no-transfer`, `--frac 0.05`,
`--exp-id E07 --change "text"` (appends a row to `experiments/log.csv`).

## Run + check locally on Windows
PowerShell, from `code/business_entity_resolution/`:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# tests
python -m pytest -q tests

# fast dev loop (no test-set run, prints OOF F0.5 + blocking recall)
python -m src.run_pipeline --data ..\..\dataset --eval-only

# full run — cap joblib worker processes; loky hits WinError 1450
# ("insufficient system resources") above ~6-8 worker processes on
# most laptops. LightGBM threads (ER_LGB_JOBS) are independent and can
# stay higher since they don't spawn OS processes.
$env:ER_JOBS = "4"
$env:ER_LGB_JOBS = "8"
python -m src.run_pipeline --data ..\..\dataset --out ..\..\output
```
Git Bash / WSL: same commands with `source .venv/Scripts/activate` and `export ER_JOBS=4
ER_LGB_JOBS=8` instead of the `$env:` lines.

Check the output is submission-ready (run from `student_resource/`, not this folder):
```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
`PASS` (exit 0) means safe to upload; anything else lists the exact rows/columns to fix.

If a run crashes partway through, re-running resumes from cached parquet under `cache/`
(normalize/blocking/features already computed are not redone) — see Caching below.

## Notebook (local or SageMaker)
`notebooks/train.ipynb` is a cell-by-cell version of the same pipeline — it imports `src/run_pipeline`'s
helpers directly, so it produces identical results to the CLI and shares the same parquet cache. Edit the
parameters cell (`EVAL_ONLY`, `FRAC`, `NO_STAGE2`, `ER_JOBS`, …) and Run All. Works unmodified after a
`git clone` onto a SageMaker notebook instance — it locates `src/` by walking up from its own directory.

## Caching
Intermediates are cached as parquet in `cache/` (override with `ER_CACHE=/path`). The cache is
namespaced by `ER_CACHE_TAG` (default `v1`). **After changing normalize, blocking or features,
set a new tag** (`export ER_CACHE_TAG=v2`) or delete `cache/`. Parallelism: `ER_JOBS` (default = cores − 2).

## Pipeline stages (`src/`)
| Module | Role |
|---|---|
| `normalize.py` | hand-written dictionaries (legal suffixes, street abbreviations, US states), name/address cleaning |
| `translit.py` | native-script → Latin token table learned from train pairs |
| `blocking.py` | per-country IDF-weighted key matching (sparse matmul), top-K candidates |
| `features.py`, `noise.py` | stage-1 similarity features + noise-channel features |
| `model.py` | LightGBM, GroupKFold by S1 entity, full refit |
| `context.py` | stage-2 group/competition/triangle-support features from OOF scores |
| `decide.py` | isotonic calibration, global one-to-one, expected-F0.5 subset selection |
| `evaluate.py` | exact macro F0.5 metric |
| `run_pipeline.py` | orchestration, experiment log, output writing, validation |
| `eda.py` | exploratory analysis script |

## Compliance
- Uses only files under `dataset/`. No external data, APIs or geocoding.
- Models: LightGBM (MIT licence), a few MB, far below the 8B-parameter limit.
- `country` is used only to group records for blocking (open label set, France included), never as a model feature.
- Seed 42 everywhere; versions pinned in `requirements.txt`.

## SageMaker (team runs)
1. Console → SageMaker AI → Notebook instances → Create: `ml.r5.4xlarge`, 100 GB volume,
   region us-east-1, default IAM role. Set a billing alert first. **Stop the instance when idle.**
2. In a terminal on the instance:
   ```bash
   conda create -y -n er python=3.12 && source activate er
   git clone <private repo url> && cd <repo>/code/business_entity_resolution
   pip install -r requirements.txt
   aws s3 cp s3://<bucket>/dataset.zip ../../ && (cd ../.. && unzip -q dataset.zip)
   ```
3. Long runs: `nohup python -m src.run_pipeline --data ../../dataset --out ../../output > run.log 2>&1 &`
   then `tail -f run.log`.
4. **Hand over between accounts (each teammate spends their own $200):** push code to git,
   `aws s3 sync cache/ s3://<shared-bucket>/cache/` and share it via a bucket policy or
   `aws s3 presign` (see the AWS guide). Keep every account in **us-east-1** to avoid transfer
   charges. Use default S3 encryption only (a custom KMS key breaks cross-account reads).
