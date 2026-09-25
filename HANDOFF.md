# HANDOFF — shift relay log

Read `CLAUDE.md` first, then this file. Update the **Current state** block and add a shift entry
before you leave. Plan: `docs/superpowers/plans/2026-09-25-entity-resolution.md`.

## Current state (overwrite every shift)
| Item | Value |
|---|---|
| Last updated | 2026-09-25 (post-first-upload) |
| Next plan task | Merge `worktree-agent-a0b1edbe6ec982855` (Prompt 3) — `src/context.py` on feat/pipeline is a stub, blocks real stage 2 |
| Best dev OOF F0.5 | 0.94642 (exp E03, stage-1 only — `NO_STAGE2=True`, stage-2 code not merged yet) |
| Dev blocking recall / cands per entity | 0.9185 / 42.5 |
| Transfer US→IN / IN→US | not computed (nan in log.csv — still owed per CLAUDE.md) |
| Best public LB (exp id) | **0.942 (exp E03, tag sub-d1-01)** — first real upload |
| Uploads used today (of 5) | D1: 1 · D2: 0 · D3: 0 |
| Chosen decoder + params | threshold, t_match=0.7, t_empty=0.3 |
| Cache tag in use | v1 |
| Running jobs (where, command, log file) | none (`mvp_run.log` / `mvp_run_err.log` at repo root from last background run) |
| SageMaker instance state (account owner) | running — `ml.r5.4xlarge`, notebook `train.ipynb` executed through section 11 + local model/decoder save |
| Shared S3 bucket for cache/artifacts | `aws-hack-sagemaker` created, **not yet writable** — notebook instance IAM role has no S3 permissions (`aws s3 cp` fails "Unable to locate credentials"); needs policy attach + instance stop/start |
| Known bugs / blockers | (1) `src/context.py` on feat/pipeline is a docstring stub — real stage-2 code only on unmerged `worktree-agent-a0b1edbe6ec982855` (commit `61efe0b`); do not flip `NO_STAGE2=False` until merged. (2) S3 push blocked on IAM role (see above). |
| Target | 0.942 → 0.985+ requested. See `docs/NEXT_STEPS.md` ranked plan; realistic ceiling is blocking-recall-bound, treat 0.985 as "push as far as possible." |

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
