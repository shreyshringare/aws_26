# CLAUDE.md — Amazon ML Challenge 2026: Business Entity Resolution

## Project in one paragraph
Three sources of noisy business records (name, address, country) with no shared key. Source 1 (S1) is deduplicated. For every S1 entity, output all Source 2 / Source 3 (S2/S3) records that are the same real-world business (zero, one or many). Scored by macro F0.5 per S1 entity. Challenge window: 25 Sep 2026 00:00 IST to 27 Sep 2026 23:59 IST.

## Hard rules — never violate
* No external data or lookups. No geocoding, business registries, ER APIs, web scraping, or downloaded datasets. Only files under `dataset/`. Hand-written normalization dictionaries (abbreviations, legal suffixes) are allowed and must be listed in the methodology doc.
* Models: MIT or Apache 2.0 licence, at most 8B parameters. Llama and Gemma are NOT allowed. OK: LightGBM, scikit-learn, XLM-R, DeBERTa-v3, multilingual-e5, MiniLM, Qwen2.5 (Apache), Mistral-7B. Record licence + size of any model added.
* Country is an open set. Test data contains France, which is absent from training. Never hard-code, filter, or one-hot on `{"US", "India"}`. Never use country as a model feature. Country may only be used as a blocking key (group by whatever labels exist).
* Every test S1 entity must appear in outputs, France included.
* All files are tab-separated. Always `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`.

## Layout (paths relative to `student_resource/`)
```
dataset/train/   train_source{1,2,3}.tsv, train_ground_truth.tsv
dataset/test/    test_source{1,2,3}.tsv
utils/validate_submission.py      official format validator (do not edit)
output/          matching_results.tsv, candidate_pairs.tsv
experiments/log.csv               one row per experiment
code/business_entity_resolution/
  src/config.py        paths, seeds, k values, thresholds
  src/io_utils.py      load TSVs, write outputs, read_lists
  src/normalize.py     name/address/postcode normalization
  src/blocking.py      candidate generation + blocking_report
  src/features.py      pair features
  src/model.py         GroupKFold LightGBM, OOF predictions, two-stage context features
  src/decide.py        threshold, empty gate, one-to-one assignment, threshold tuning
  src/evaluate.py      exact macro F0.5 scorer + breakdowns
  src/run_pipeline.py  end-to-end entry point
  tests/               pytest
  README.md, requirements.txt
Documentation_template.md         methodology write-up
```
This CLAUDE.md lives in `student_resource/` and is not part of the submission zip.

## Commands
```bash
# setup
cd code/business_entity_resolution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full pipeline (train -> test outputs)
python -m src.run_pipeline --data ../../dataset --out ../../output

# local evaluation only (OOF on train, prints breakdown)
python -m src.run_pipeline --data ../../dataset --eval-only

# tests
pytest -q tests

# official validator (run from student_resource/)
python3 utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Output format (rejection if wrong)
* `matching_results.tsv`: header `source1_entity_id\tmatched_entity_ids`
* `candidate_pairs.tsv`: header `source1_entity_id\tcandidate_entity_ids`
* One row per test S1 entity, no duplicate S1 rows, order = test_source1 order.
* ID lists comma-separated, no quotes, no spaces, no duplicates, only S2-/S3- IDs that exist in the test files. Empty list = empty second column.
* Matches must be a subset of candidates.
* `candidate_pairs.tsv` = the exact set the model scored (last filtering stage), not an earlier looser pass. Keep it tight: it is audited for recall and reduction ratio.

## Metric (implement exactly; never substitute pairwise F1)
Per S1 entity, then mean over all S1 entities:
* true empty & pred empty -> 1.0; true empty & pred non-empty -> 0.0
* true non-empty & (pred empty or tp == 0) -> 0.0
* else P = tp/|pred|, R = tp/|true|, F0.5 = 1.25PR / (0.25*P + R)
Sanity check: pred {a,b,x}, true {a,b} -> 0.714.

## Pipeline design
1. Normalize: unidecode, lowercase, `&`->`and`, punctuation->space, split legal suffix into its own column, expand address abbreviations (careful: `st` can mean saint in French), extract postcode and number tokens.
2. Block: per country label, union of char 2-4 gram TF-IDF kNN on name and on address, shared postcode (small groups only), rare shared tokens. Target blocking recall >= 0.97 with tens (not hundreds) of candidates per entity. Always report recall and candidates/entity after changes.
3. Features: rapidfuzz ratio / token_set / partial / Jaro-Winkler, token Jaccard, TF-IDF cosine for name and address; postcode equal/missing; number-token Jaccard; legal-suffix agreement; name frequency; is_s3; length diff.
4. Model: LightGBM, stage 1 on pair features; stage 2 adds context features (rank in S1 group, gap to group best, gap to candidate's best other S1, group size) computed from stage-1 OUT-OF-FOLD scores.
5. Decide: optional one-to-one (each S2/S3 to its best S1, only if EDA confirms), match threshold, empty gate (predict empty if best prob < t_empty). Tune on OOF with the exact scorer; prefer flat regions of the score surface.

## Validation rules
* Split by S1 entity with GroupKFold (5 folds). Never split by pair.
* Held-out entities search the full train S2/S3 pool.
* Tune thresholds and build context features only from OOF predictions.
* Every evaluation prints: all / per country / singletons / matched, plus blocking recall and candidates/entity.
* Also report cross-country transfer: train US -> score India, and India -> US. This is the proxy for France; do not accept changes that improve overall but hurt transfer.

## Working conventions
* One change per experiment. Append a row to `experiments/log.csv`: exp_id, change, block_recall, cands_per_entity, f05_all, f05_us, f05_in, f05_singletons, f05_matched, us_to_in, in_to_us, t_match, t_empty, public_lb, decision.
* Git: commit before every experiment; tag every leaderboard upload `sub-dN-MM`.
* Fixed seeds everywhere (config.SEED = 42). Pin versions in requirements.txt.
* Every function gets a docstring (required by the organizers).
* Cache expensive intermediates (normalized frames, candidates, features) as parquet under `code/business_entity_resolution/cache/` (gitignored, regenerable).
* Keep the pipeline runnable end-to-end from raw data with one command at all times.
* Prefer vectorized rapidfuzz (`process.cpdist`) or joblib if feature extraction is slow.

## Submission budget
5 uploads/day, 15 total. Public LB = subset; final rank = private LB.
* Always run the validator before uploading; never spend an upload on a format error.
* Upload only on a meaningful local gain, or to test France behaviour (e.g. same file with France rows emptied vs. predicted).
* Pick the final submission by local OOF + cross-country robustness, not public LB.

## Do not
* Do not add network calls or external datasets to the pipeline.
* Do not tune on in-fold predictions or leak labels into features.
* Do not replace LightGBM with an LLM; neural models are optional extra features only.
* Do not edit `utils/validate_submission.py` or anything in `dataset/`.
* Do not change output ordering or headers.

## Current status (update as you go)
* [ ] EDA answers: singleton rate = ?, one-to-one holds = ?, cross-country matches = ?
* [ ] E00 all-empty submission: public LB = ?
* [ ] Fuzzy baseline: local = ?, public = ?
* [ ] LightGBM stage 1 / stage 2
* [ ] Decision layer tuned
* [ ] Packaging + README + methodology doc
