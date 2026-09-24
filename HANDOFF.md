# HANDOFF — shift relay log

Read `CLAUDE.md` first, then this file. Update the **Current state** block and add a shift entry
before you leave. Plan: `docs/superpowers/plans/2026-09-25-entity-resolution.md`.

## Current state (overwrite every shift)
| Item | Value |
|---|---|
| Last updated | 2026-09-25 (planning shift) |
| Next plan task | Task 1 (Phase 0 foundation) |
| Best dev OOF F0.5 | — |
| Dev blocking recall / cands per entity | — |
| Transfer US→IN / IN→US | — |
| Best public LB (exp id) | — |
| Uploads used today (of 5) | D1: 0 · D2: 0 · D3: 0 |
| Chosen decoder + params | — |
| Cache tag in use | v1 |
| Running jobs (where, command, log file) | none |
| SageMaker instance state (account owner) | not created |
| Shared S3 bucket for cache/artifacts | not created |
| Known bugs / blockers | none |

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
### 2026-09-25 — planning
- Done: EDA (singletons 5.6%, one-to-one exact, 0 cross-country pairs, France = 15% of test S1),
  design spec, implementation plan, CLAUDE.md, README, requirements.
- Provisional code already present: `src/config.py`, `src/io_utils.py`, `src/evaluate.py`.
- Next: plan Task 1.
