# Amazon ML Challenge 2026 — Business Entity Resolution

Shareable summary of where this project stands. For full detail see `CLAUDE.md` (rules/conventions), `HANDOFF.md` (shift log), `docs/superpowers/plans/2026-09-25-entity-resolution.md` (task checklist), `experiments/log.csv` (every experiment).

## The problem

Three noisy business-record sources (name, address, country), no shared key:
- **Source 1 (S1)** — deduplicated, the "canonical" list.
- **Source 2 / Source 3 (S2/S3)** — noisy records that may or may not refer to the same real business.

For every S1 entity, output every S2/S3 record that's the same real-world business (could be zero, one, or many). Scored per-entity with **macro F0.5** (precision weighted higher than recall).

Challenge window: 25–27 Sep 2026 (IST). Submission budget: 5 uploads/day, 15 total.

## Hard constraints

- No external data, APIs, geocoding, or web scraping — only files under `dataset/`.
- Models must be MIT/Apache-2.0, ≤8B params. No Llama/Gemma. LightGBM is the workhorse; small transformer encoders (XLM-R, MiniLM, e5) are allowed as feature sources, not required.
- **Country is an open set.** Test data includes France, which never appears in training. Country can only be used as a *blocking key* — never as a model feature, never hard-coded.
- Every test S1 entity must appear in the output, including France.

## Pipeline architecture

```
normalize -> block -> feature-extract -> LightGBM (stage 1) -> context features (stage 2) -> decide -> outputs
```

1. **Normalize** (`src/normalize.py`) — unidecode, lowercase, punctuation cleanup, legal-suffix split, address abbreviation expansion, postcode/number extraction.
2. **Block** (`src/blocking.py`) — per-country candidate generation via char n-gram TF-IDF kNN (name + address) + shared postcode + rare shared tokens. Goal: recall ≥0.97 with tens of candidates per entity, not hundreds.
3. **Features** (`src/features.py`) — fuzzy string similarity (ratio/token_set/partial/Jaro-Winkler), token Jaccard, TF-IDF cosine, postcode/number agreement, legal-suffix match, name frequency, etc.
4. **Model** (`src/model.py`) — LightGBM with GroupKFold (split by S1 entity, never by pair). Stage 1 scores pairs; stage 2 adds context features (rank within group, gap to best competitor, group size) computed from stage-1 out-of-fold scores.
5. **Decide** (`src/decide.py`) — calibration, optional one-to-one constraint, match/empty thresholds tuned on OOF predictions using the *exact* competition scorer.

Every change is validated on: overall F0.5, per-country F0.5, singleton F0.5, matched-only F0.5, **and cross-country transfer (train US → score India, and vice versa)** — this is the proxy for how the model will generalize to unseen France data. A change that helps overall but hurts transfer is rejected.

## Where things stand (as of exp E06, latest)

| Metric | Value |
|---|---|
| Dev OOF F0.5 (all) | **0.96817** (up from E03 baseline 0.94642 — +2.18pt this session) |
| Blocking recall / candidates per entity | 0.9523 / 42.6 |
| US→India transfer | 0.89623 |
| India→US transfer | 0.94474 |
| Best public leaderboard score | 0.942 (exp E03, one upload spent) |
| Chosen decoder | threshold-based, t_match=0.7, t_empty=0.3 (may be re-tuned since E05 briefly used an expected-F0.5 decoder) |
| Target | 0.985+ (stretch, from a 0.942 starting point) |

**E06** (latest experiment) combined: alias splitting (dba/aka/fka/formerly/nee → separate `name_alias` column, best-of-name-vs-alias scoring), French `etablissements/ets` legal-token handling, `bis/ter` street-suffix handling, plus new retrieval-evidence/ambiguity features. It improved F0.5, singleton F0.5 (+0.89pt), *and* both transfer directions simultaneously — a rare no-tradeoff win.

## Known open items

- Stage 2 (context features) has hit a snag on the full SageMaker run: `models2` comes back empty, so it silently falls back to stage-1-only scoring. Root cause not yet traced (see `HANDOFF.md` blockers).
- S3 cache-sharing bucket (`aws-hack-sagemaker`) exists but isn't writable yet — the notebook instance's IAM role needs an S3 policy attached.
- Only 1 of 5 daily uploads used so far; the plan is to spend more once stage 2 is confirmed working end-to-end and transfer numbers look solid.
- Packaging (README, methodology doc, zip) is not done yet — last task in the plan.

## Team logistics

- Each teammate has their own AWS account + $200 credits (region `us-east-1`). Full pipeline runs need ~100GB RAM (`ml.r5.4xlarge` instance).
- Shift-based handoff: `HANDOFF.md` is the relay log — read it, work the next unchecked task in the plan, commit after each change, update `HANDOFF.md` before leaving.
- Every experiment gets a row in `experiments/log.csv` (recall, cands/entity, all F0.5 breakdowns, transfer, thresholds, public LB, decision/reasoning).

## Repo layout

```
dataset/                     train/test TSVs (source1/2/3 + ground truth)
output/                      matching_results.tsv, candidate_pairs.tsv (submission files)
experiments/log.csv          one row per experiment, full history
code/business_entity_resolution/
  src/                       normalize, block, features, model, decide, evaluate, run_pipeline
  tests/                     pytest suite
docs/superpowers/            design spec + implementation plan (task checklist)
CLAUDE.md                    full rules/conventions (source of truth)
HANDOFF.md                   shift relay log
```
