# HANDOFF — shift relay log

Read `CLAUDE.md` first, then this file. Update the **Current state** block and add a shift entry
before you leave. Plan: `docs/superpowers/plans/2026-09-25-entity-resolution.md`.

## Current state (overwrite every shift)
| Item | Value |
|---|---|
| Last updated | 2026-09-26 (post-E06; Prompts 8A-8H code landed, full SageMaker run is next) |
| Next plan task | Full SageMaker run with stage 2 + Exp J2 + all Prompt 8A-8H code enabled (Prompt 9), then re-tune decoder for real (Prompt 8A) and re-measure blocking recall at full scale (Prompt 8F) |
| Best dev OOF F0.5 | **0.96817 (exp E06**, normalization v2 + retrieval-evidence features; up from E05's 0.96519 / E04's 0.96274 / E03's 0.94642) |
| Dev blocking recall / cands per entity | 0.9523 / 42.6 (E06, dev-sample only — full-scale re-check still open, Prompt 8F) |
| Transfer US→IN / IN→US | **0.89623 / 0.94474** (E06 — best transfer numbers yet, improved alongside f05_all) |
| Best public LB (exp id) | 0.942 (exp E03, tag sub-d1-01) — still the only upload spent; E04/E05/E06's gains are NOT on the leaderboard yet |
| Uploads used today (of 5) | D1: 1 · D2: 0 (session) |
| Chosen decoder + params | ef05, tau=0.8, lam=0.0 (cache/v3_decoder.json, oof_f05=0.96817 — matches E06 exactly, so this WAS re-tuned for E06 already; earlier docs claiming a stale t_match=0.7/0.3 were themselves stale, corrected this session) |
| Cache tag in use | v3 |
| Running jobs (where, command, log file) | none |
| Code landed this session (not yet run at scale) | Exp J1 (`context.cluster_features`, cross-source cluster features), Exp J2 (`decide.correlated_hits` + `comp_col`/`joint` wiring through `tune`/`apply_decoder`/`run_pipeline.main`), Exp HN1 hook (`sampling.training_rows(hard_fp=...)`), segment-specific decoding (`decide.tune_segments`/`apply_segmented_decoder`), France vocabulary-coverage empty gate (`decide.select_threshold_vocab_aware`). All TDD, all unit-tested (103 tests pass), none exercised on real data yet — local machine has ~1.9GB free RAM, unsafe for a full stage-1+stage-2 retrain (see docs/RESEARCH_JOINT_S2S3.md, docs/NEXT_STEPS.md items 11-15). Plan: exercise all of it in one full SageMaker run rather than locally. |
| SageMaker instance state (account owner) | not confirmed running this session — check before assuming it's up |
| Shared S3 bucket for cache/artifacts | `aws-hack-sagemaker` created, **still not confirmed writable** — check IAM role policy before relying on it |
| Known bugs / blockers | **None currently open.** (Corrected this session: "models2 empty" was never a bug — the one full run to date deliberately passed `--no-stage2`; don't pass it on the next full run.) RAM is tight on the local Windows machine (as low as ~1.9GB free observed this session) — this is why the new J1/J2/HN1/segment/vocab-gate code hasn't been measured locally; use SageMaker instead. |
| Target | 0.942 (public LB) → 0.985+ requested. Dev OOF now at 0.96817 (E06). See `docs/NEXT_STEPS.md` items 11-16 for the ranked path forward. |

## Credits tracker (each teammate has their own AWS account)
| Member | AWS plan (Free/Paid) | Credits left | Instance quota ok? | Notes |
|---|---|---|---|---|
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

## Shift schedule (IST)
| Shift | Time | Driver | Focus |
|---|---|---|---|
| S1 | D1 06–12 |  | Tasks 1–7: foundation, E00 upload, normalization |
| S2 | D1 12–18 |  | Tasks 8–9: blocking, recall ≥ 0.97 |
| S3 | D1 18–24 |  | Tasks 10–15: stage-1 model, first real upload |
| S4 | D1 24–D2 06 |  | full-data run on SageMaker, translit/dictionary checks |
| S5–S8 | D2 |  | Tasks 16–18: stage 2, decoder, iteration, 3–5 uploads |
| S9 | D3 06–12 |  | last improvements; **feature freeze 14:00** |
| S10 | D3 12–18 |  | final full run, final selection (OOF + transfer) |
| S11 | D3 18–24 |  | Tasks 19–21: zip, methodology doc, final upload |

## Upload checklist (shift driver only; guidelines: max 5/day, keep version history)
- [ ] Validator PASS (from `student_resource/`)
- [ ] Upload `output/matching_results.tsv` in the portal (one login per person, desktop only)
- [ ] `python scripts/archive_submission.py --tag sub-dN-MM --exp-id EXX --public-lb <score>` (archives files + meta, git-tags the commit)
- [ ] Record `public_lb` in `experiments/log.csv` and in the Current state table; sync `submissions/` to S3

## End-of-shift checklist
- [ ] Code committed and pushed; experiment rows appended to `experiments/log.csv`
- [ ] Any long job started *before* leaving (note it above)
- [ ] Cache synced to the shared S3 bucket if the next driver uses another account
- [ ] SageMaker instance stopped if nothing is running
- [ ] Current state table updated; shift entry written below

## Queued work derived from EDA v2 (docs/EDA.md §G)
1. **Data picking** (new `src/sampling.py`): dev/train entities stratified by country × match-count bin (0 / 1–2 / 3–4 / 5+); training rows = all positives + all negatives in top-10 block ranks + 25% of the rest (weight 4); OOF still scores all rows. Also: skip the validator call when `utils/` is absent (self-contained zip).
2. **Normalization v2** (bump cache tag to v2): split aliases (`dba|aka|fka|formerly|formerly known as|nee|doing business as`) into a `name_alias` column and score the best of name vs alias; French `etablissements/ets` as a legal-like token; street key skips `bis/ter`; generic vocab requires ≥ 3-char tokens; seed French generic words (ecole, amicale, groupe, association…) only if learned vocab misses them.
3. **Features**: drop postcode features (dead: ≤ 1.9% coverage); add address-shape features (addr token count, landmark flag already exists); alias-aware name similarity.
4. **France risk**: 45% of France name tokens unseen in train (the US↔IN proxy only shows 28–36%) → prefer char-level similarity; rarity/frequency stats computed on the test pool per country (already the case).

## Shift entries (newest first)
### 2026-09-26 — post-E06 differentiator code (Prompts 8A-8H), deferred to SageMaker for measurement
- Ranked the remaining differentiation opportunities (joint S2/S3 cluster consistency, hard-negative
  mining, blocking recall push, France hardening, segment thresholds, ensembling) and wrote
  `docs/RESEARCH_JOINT_S2S3.md` researching the top-ranked item in depth. Turned the list into
  `docs/PROMPTS.md` Prompts 8A-8H and `docs/NEXT_STEPS.md` items 11-16.
- Corrected a stale/incorrect claim from earlier docs: "models2 empty" was never an unresolved bug
  (see `docs/NEXT_STEPS.md` Warning 13, `docs/PROMPTS.md` Prompt 8B, `SOLUTION_DEEPDIVE.md` §14) —
  the one full run to date just deliberately passed `--no-stage2`.
- Local machine RAM was too tight (~1.9GB free) to safely retrain stage 1+2 for a real Prompt 8A
  decoder re-tune, so pivoted per user direction: implement all the code (TDD) locally, defer actual
  measurement to a single comprehensive SageMaker full run.
- Landed, all TDD with unit tests (103 pass): Exp J1 cluster features (`context.cluster_features`),
  Exp J2 correlated decision-time decoder (`decide.correlated_hits`, wired through
  `tune`/`apply_decoder`/`run_pipeline.main` via `comp_col`/`dec["joint"]`, backward-compatible —
  omitting `comp_col` reproduces prior behavior exactly), Exp HN1 hard-negative mining hook
  (`sampling.training_rows(hard_fp=...)`, also fixed a latent `ZeroDivisionError` at
  `easy_rate=0.0`), segment-specific decoding (`decide.tune_segments`/`apply_segmented_decoder`),
  France vocabulary-coverage empty gate (`decide.select_threshold_vocab_aware`).
- **None of the above has been measured on real data yet.** Next driver: run the full pipeline on
  SageMaker with stage 2 enabled (not `--no-stage2`) so Exp J2 actually engages, then work through
  Prompt 8A (decoder re-tune, now automatic since `decide.tune` always runs) and Prompt 8F (blocking
  recall at full scale) using that same run's outputs before deciding whether to spend an upload.

### 2026-09-25 — MVP run (stage-1 only)
- Ran `train.ipynb` end-to-end on `ml.r5.4xlarge`: normalized, blocked (B03 multi-channel,
  recall 0.9185, 42.5 cands/entity), scored 74,992,735 test pairs, decoded, wrote outputs.
- Validator PASS: 1,732,544/1,732,544 S1 rows (122,399 empty, 1,610,145 non-empty). Not yet
  run with `--check-ids`.
- Local OOF f05_all = 0.94642 (matches exp E03 exactly — this run's `models2` came back empty,
  so `m2` was `None` and scoring fell back to stage-1-only; stage-2 context features were not
  applied). Numbers not yet logged as their own row in `experiments/log.csv`.
- Saved `m1_stage1.txt` + `{tag}_decoder.json` to instance-local `cache/` (no `m2_stage2.txt`
  since m2 is None). S3 push to `aws-hack-sagemaker` fails on missing IAM creds — see blockers.
- No LB upload made — outputs and models exist locally on the instance only.
- Next: (1) find why `models2` is empty and fix stage-2 training, (2) fix notebook instance IAM
  role for S3, (3) log this run properly in `experiments/log.csv`, (4) compute US↔IN transfer,
  (5) decide whether to spend first real upload.

### 2026-09-25 — planning
- Done: EDA (singletons 5.6%, one-to-one exact, 0 cross-country pairs, France = 15% of test S1),
  design spec, implementation plan, CLAUDE.md, README, requirements.
- Provisional code already present: `src/config.py`, `src/io_utils.py`, `src/evaluate.py`.
- Next: plan Task 1.
