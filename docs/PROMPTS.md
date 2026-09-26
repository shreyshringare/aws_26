# Step-by-step prompts (copy-paste into Claude Code)

Run these in order. Each prompt is self-contained: paste it into a Claude Code session opened at
`D:\AWS_HACK\student_resource` (or the SageMaker checkout). Do one prompt per session or per shift.
Before starting, check `HANDOFF.md` → "Current state" to see which step is next.

State at the time of writing (26 Sep, updated through Prompt 8 / exp E06 done, Prompts 8A-8H added):
- **`docs/NEXT_STEPS.md` is the live priority tracker — read it first, this header is a snapshot.**
  It has a "Status update" note at the top of its ranked-order section showing exactly which items are
  done vs pending, plus its own Warnings/Leads sections. This PROMPTS.md header will drift; NEXT_STEPS
  is where completions get marked.
- **Prompt 1 (MVP)**: done. Prompt 2 (upload): done — first real submission, tag `sub-d1-01`, exp E03,
  public LB **0.942**, recorded in `experiments/log.csv` and `HANDOFF.md`. No upload spent since —
  E04/E05/E06's dev gains are not yet reflected on the leaderboard.
- **Prompt 3 (merge stage-2 branch)**: done. **Prompt 4 (blocking)**: done, B05 kept, recall 0.9534.
  **Prompt 5 (E04 stage-1 dev eval)**: done, 0.96274 all. **Prompt 6 (E05 stage-2)**: done and kept —
  0.96519 all, beats E04 on every metric. **Prompt 7/7B (error taxonomy + peer-review features)**: done,
  `docs/ERRORS.md` written. **Prompt 8 (normalization v2)**: done — **exp E06 is the current best**,
  0.96817 all / 0.89623 US→India / 0.94474 India→US, improving singletons and both transfer directions
  simultaneously (see `experiments/log.csv`).
- **New since Prompt 8: `docs/RESEARCH_JOINT_S2S3.md`** researches the highest-ranked remaining
  differentiator (joint S2↔S3 cluster consistency, `docs/IDEAS.md` B.6) and **Prompts 8A-8H** below turn
  the full post-E06 priority list (`docs/NEXT_STEPS.md` items 11-16) into runnable prompts, in
  recommended order: 8A (cheap decoder re-tune) → 8B (fix the stage-2-silently-disabled bug, prerequisite
  for a real full run) → 8C/8D (joint S2↔S3, the top differentiator) → 8E (hard-negative mining) → 8F
  (blocking recall full-data re-check) → 8G (France hardening) → 8H (segment thresholds). Do these before
  Prompt 9's full run so the full run actually reflects the best model, not E06 alone.
- **Known open blocker (do not skip):** the last full SageMaker run had `models2` come back empty even
  with stage 2 intended on — test inference silently fell back to stage-1-only. This is a DIFFERENT
  situation from the old, intentional `NO_STAGE2=True` MVP case; see Prompt 8B and
  `docs/NEXT_STEPS.md` Warning 13. Fix this before Prompt 9's full run or the upload won't reflect E05/
  E06's stage-2 gains at all.
- **`output/` currently holds the empty E00 baseline, not a real model run** (verified: 0 non-empty rows
  out of 1,732,544) — see `SOLUTION_DEEPDIVE.md` §12. Regenerate before trusting it for Prompt 9 or a
  zip build.
- Own-EDA (`docs/EDA.md` §G) and teammate-EDA (`docs/EDA_TEAMMATE.md`) findings have both been
  cross-checked directly against the source — see `docs/NEXT_STEPS.md` items 9 and 10, now resolved.
  Alias splitting and French `etablissements`/`bis`/`ter` landed in E06. Still open: near-miss number
  flag (distinct from `f_nz_num_conflict` — don't conflate), NaN-vs-0 for missing-address features (may
  now be partially covered by E06's retrieval-evidence/ambiguity features — verify before re-doing).
- Models + decoder saved locally on the SageMaker instance (`cache/m1_stage1.txt`,
  `cache/{tag}_decoder.json`) via notebook section 12. **Not yet in S3** — the notebook instance's IAM
  role has no S3 permissions (`aws s3 cp` → "Unable to locate credentials"); needs a policy attach +
  instance stop/start before `aws-hack-sagemaker` is usable. This does NOT block portal upload or
  `archive_submission.py` — only the S3 backup/sharing step.
- The first full test run crashed in blocking (WinError 1450, too many worker processes). Fix committed
  in `0acf8f7` (`ER_LGB_JOBS` separate from `ER_JOBS`). System RAM is generally tight on this machine
  (as low as 1.1GB free observed during blocking runs) — the specified `ER_JOBS=4` in prompts below may
  need lowering to 2 depending on what else is running; watch `Get-Process python` and free memory
  before launching a heavy prompt.
- **Portal rules (guidelines):** each teammate uses only their own login on one laptop or desktop (no simultaneous logins, no mobile). Max 5 uploads per team per day, and every upload is archived (version history is used for shortlisting). Ask questions via the organisers' Google Form; they won't make decisions for us.
- ⚠️ **Only one session may run the pipeline at a time.** A full run needs most of the RAM; two parallel runs both crash or thrash badly. Check with `Get-Process python` first.

## ▶ Start here (visible progress path)
1. **Prompt 0** (session start) → shows where things stand.
2. **Prompt N0** (dashboard notebook) → already built; re-run it after every step to *see* progress
   (score chart, blocking recall, live run log, uploads).
3. Prompts 1-8 are done (MVP through normalization v2, exp E06 = current best, 0.96817 dev OOF). Don't
   re-run them; read `docs/NEXT_STEPS.md`'s status note if you need the history.
4. **Next up: Prompts 8A → 8H, in order** — the post-E06 differentiator list
   (`docs/NEXT_STEPS.md` items 11-16, researched in `docs/RESEARCH_JOINT_S2S3.md` for the top item).
   Start with **8A** (cheap decoder re-tune) and **8B** (fix the stage-2-silently-disabled bug — a real
   prerequisite, not optional) before the heavier **8C/8D** (joint S2↔S3 cluster consistency, the
   highest-ranked new differentiator), then **8E-8H**.
5. Only after 8A-8H (or as many as time allows, in that priority order): **Prompt 9** (full SageMaker run
   → submission #2, with a real upload this time reflecting all the dev gains since E03) → **Prompt 10**
   (France check) → **Prompt 11** (push to GitHub) → **Prompts 12-13** (Day 3 packaging + final
   selection) → **Prompt 14** (end of shift, every shift).

Where progress shows up: `notebooks/00_dashboard.ipynb` (charts), `experiments/log.csv` (every run), `HANDOFF.md` (current state), `submissions/` (every upload), `git log`.

**Notebooks — rules:** notebooks live in `student_resource/notebooks/`. They only *drive and visualise*. They call `python -m src.run_pipeline …` as a subprocess (or read its log), import read-only helpers from `src/`, and never contain pipeline logic (the zip must reproduce from `src/` with one command). Clear heavy outputs before committing. On SageMaker, start long runs with `nohup` from a notebook cell and tail the log in another cell, so a browser disconnect doesn't kill the run.

---

## Prompt N0 — Progress dashboard notebooks (build once, reuse every shift)
```
Create student_resource/notebooks/ with three thin notebooks (no pipeline logic inside; use src/ and
subprocess only; keep each under ~15 cells; matplotlib only, no new dependencies except the notebook
runtime):
1. 00_dashboard.ipynb — reads experiments/log.csv and submissions/*/meta.json and shows:
   a) table of all experiments; b) line chart of f05_all (plus US / India / singletons) by exp_id;
   c) bar chart of block_recall and cands_per_entity by exp_id; d) public LB per upload with uploads
   used per day (limit 5); e) the last 15 lines of the newest *run*.log in student_resource/;
   f) python processes running (psutil if available, else skip).
2. 01_run_mvp.ipynb — one cell per MVP step (1A pre-flight checks, 1B start the run in the background
   with nohup/subprocess.Popen writing ../mvp_run.log, a cell that tails the log, 1C validator + QA
   summary from docs/QA_*.md). Each cell prints clear PASS/FAIL lines.
3. 02_errors.ipynb — loads cached dev features + OOF scores (tag set in the first cell) and shows
   false positives / false negatives side by side with raw records, filterable by country and error type.
Install the notebook runtime locally if missing (pip install notebook ipykernel) and add it to a
separate requirements-dev.txt (not requirements.txt, which stays the pinned pipeline environment).
Run 00_dashboard top to bottom once to prove it works with the current log (E01–E03). Clear outputs,
then commit.
```

---

## Prompt 0 — Session start (use at the beginning of every shift)
```
Read CLAUDE.md, then HANDOFF.md, then docs/STRATEGY.md §3 and docs/IDEAS.md Part A.
Run `git status`, `git log --oneline -10` and `python -m pytest -q tests` (from code/business_entity_resolution).
Summarise in 10 lines: current best dev score, blocking recall, uploads used today,
uncommitted changes, running jobs, and the next step in docs/PROMPTS.md. Don't change anything yet.
```

## MVP phase: from working code to a file ready for the portal (Prompts 1A–1D)
The MVP is the stage-1 pipeline that is already built and measured (dev OOF F0.5 0.94642, exp E03). These
four prompts only run it, check it and package it. **No model or feature changes until the first upload is
done.**

**Status: effectively done, via a different path.** The actual run happened in
`notebooks/train.ipynb` on the SageMaker instance (not this CLI sequence) — output exists, validator
PASSED (1,732,544/1,732,544 rows). These prompts remain valid if you need to reproduce via CLI, but skip
to Prompt 1C to check whether `docs/QA_E03r.md` exists yet (it likely doesn't — the notebook path didn't
write it), then Prompt 1D/Prompt 2 for the actual upload.

### Prompt 1A — Pre-flight check
```
Goal: make sure the machine and repo are ready for the full MVP test run. Change nothing except commits.
1. PowerShell: Get-Process python. If any pipeline process is running, STOP and tell me.
2. git status / git log -5 on feat/pipeline: confirm commit 0acf8f7 (ER_LGB_JOBS fix) is present and
   experiments/log.csv holds the E03 row; commit any pending log/doc changes (not code).
3. python -m pytest -q tests (from code/business_entity_resolution): all must pass.
4. Confirm the caches exist in code/business_entity_resolution/cache: v1_translit.tsv, v1_norm_train_*,
   v1_norm_test_source{1,2,3}.parquet, v1_cands_train_0.1.parquet, v1_feats1_train_0.1.parquet,
   v1_generic_vocab.json.
5. Report free RAM and free disk.
Done when: all checks are green, or you report exactly what isn't.
```

### Prompt 1B — Run the MVP on the full test set
```
Goal: produce output/matching_results.tsv and output/candidate_pairs.tsv from the MVP (stage 1 only).
Context: the first full run crashed in test blocking with WinError 1450 because ER_JOBS=10 spawned too
many worker processes; fixed by 0acf8f7 (ER_LGB_JOBS separate). Test normalization + train features are
cached (tag v1): the run retrains LightGBM OOF (~25 min), refits, blocks the test set (~40 min),
scores test pairs in streamed chunks, decodes, writes outputs and runs the validator.
1. From code/business_entity_resolution start in the background, logging to ../../mvp_run.log:
   ER_JOBS=4 ER_LGB_JOBS=12 python -u -m src.run_pipeline --data ../../dataset --out ../../output
   --frac 0.1 --no-stage2 --no-transfer --exp-id E03r --change "MVP full run"
2. Watch the log for stage lines and for Traceback / MemoryError / WinError. Report each stage as it finishes.
3. If it crashes in blocking or scoring: lower ER_JOBS to 2 and rerun (caches resume). If it crashes with
   MemoryError in scoring: rerun with the chunk argument of score_stream lowered to 1_000_000 (smallest
   code change possible, commit it).
Don't change the model, features or decoder. Don't upload.
Done when: the log shows "validator exit=0" and the run finishes.
```

### Prompt 1C — Sanity QA of the predictions (before anyone uploads)
```
Goal: prove output/matching_results.tsv is correct and sensible. Read-only except a QA report file.
Check and report:
1. Official validator with ID checks, from student_resource/:
   python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
   (if it runs out of memory, rerun without --candidate). Must print PASS.
2. Row count = number of rows in dataset/test/test_source1.tsv (1,732,544), same order, header exact.
3. Every matched id is in candidate_pairs for the same S1 row; no S1-/duplicate ids; no id assigned to
   two different S1 entities (one-to-one).
4. Per country (US, India, France): share of S1 with ≥1 match, mean matches per matched entity, and the
   candidates/entity. Compare with the train dev expectation (≈94% with ≥1 match, mean ≈3.5). Flag
   anything far off, especially France.
5. Print 10 random matched examples per country (S1 name|address → matched names|addresses) and 5 S1
   with empty predictions per country, so a human can eyeball them.
Write the results to docs/QA_E03r.md and commit it.
Done when: validator PASS + checks 2–3 clean. If anything fails, stop and explain; don't patch outputs by hand.
```

### Prompt 1D — Package the upload
```
Goal: everything ready so a human can upload in one step, with full version history.
1. Confirm docs/QA_E03r.md shows PASS and clean checks.
2. Update HANDOFF.md → Current state: best dev F0.5 0.9464 (E03), blocking recall 0.9185,
   decoder threshold t_match=0.7, cache tag v1, "ready to upload: output/matching_results.tsv (E03r)".
3. Commit (code + docs only; output/ is gitignored).
4. Print the exact next steps for the human:
   a) upload output/matching_results.tsv in the portal (one login per person, desktop only; counts toward 5/day)
   b) then run Prompt 2 with the public score.
Don't upload anything yourself.
```

---- ⬆ MVP complete: **human uploads `output/matching_results.tsv` to the portal** ⬆ ----

## Prompt 2 — Record the upload (human uploads first)
```
I uploaded output/matching_results.tsv to the portal; status = <SCORED or error text>, public LB = <SCORE>.
If the status is not SCORED, stop and diagnose the format against the problem statement (don't archive).
Run: python code/business_entity_resolution/scripts/archive_submission.py --tag sub-d1-01 --exp-id E03 --public-lb <SCORE>
Put public_lb into the E03 row of experiments/log.csv, update HANDOFF.md (Current state, uploads used today),
and commit.
Note: archive_submission.py only copies output/ into submissions/<tag>/ and git-tags the commit — it does
NOT push to S3 itself (its docstring says "sync it to the team S3 bucket" as a separate manual step).
The S3 IAM-role blocker on the SageMaker instance does not block this prompt.
```

## Prompt 3 — Merge the measured-later improvements (now a hard prerequisite for stage 2)
```
Merge branch worktree-agent-a0b1edbe6ec982855 into feat/pipeline. It contains:
na name×address cross keys + EDA weights (drop postcode family); num_conflict and acronym features
(postcode features removed, stricter generic vocab); src/sampling.py (stratified S1 sample by
country × match-count bin, hard-negative training rows with weights); stage 2 in src/context.py;
wider decoder grids.
Resolve conflicts, keeping the LGB_JOBS config change from feat/pipeline. Run the full test suite.
Commit. Then remove the worktree (git worktree remove) once merged. Don't run the pipeline.

Confirmed by checking both branches directly: feat/pipeline's src/context.py is currently a one-line
docstring stub with no implementation; the worktree branch (commit 61efe0b, "feat: stage-2 context,
competition and triangle-support features") has the real group_context/triangle-support code. Until this
merges, do NOT set NO_STAGE2=False in train.ipynb or run Prompt 6 — context.add_stage2_features would
call an unimplemented function.
```

## Prompt 4 — Blocking experiments (recall ceiling)
```
Measure blocking changes on the dev sample, one change per run.
Use ER_CACHE_TAG=v2 (v2_norm_* caches already exist); delete v2_cands_train_* between runs.
Command (from code/business_entity_resolution):
ER_CACHE_TAG=v2 ER_JOBS=4 python -m src.run_pipeline --data ../../dataset --eval-only --stop-after blocking --exp-id <ID> --change "<text>"
Runs: B04 = EDA weights only (disable the na family); B05 = + na cross keys; B06 = best one + channel k +10
on the channel that gains most.
Report a table: exp, recall, cands/entity, runtime, peak RAM. Keep the best config with cands/entity ≤ 60
and commit it. Then run the miss diagnosis (plan Task 9 Step 4) and summarise the remaining miss types.

Reference point (docs/EDA_TEAMMATE.md §10, EXACT full-train, deterministic union not our IDF-weighted
kNN so not directly comparable — but a sanity anchor): exact name+address+postal = 31.9% recall at ~11
cands/S1; + rare address token = 78.7% at 231; + rare name token = 90.5% at 2,075; everything = 96.3% at
11,231. Our sparse IDF union already beats this efficiency (B03: 91.85% at 42.5) — the target is closing
the last few points of recall without the candidate-count blowup their deterministic approach hits.
```

## Prompt 4B — Character n-gram retrieval channel (peer review; only if recall is still < 0.95 after Prompt 4)
```
Add a fourth blocking channel "char": character 3–4-gram TF-IDF of name_core (sklearn
TfidfVectorizer(analyzer="char_wb", ngram_range=(3,4), min_df=2, dtype=float32)), fitted per country
on that country's pool. Query S1 in batches (sparse dot, top-k per row, k=15) so memory stays bounded;
union with the existing channels and pass its rank/score as extra columns.
Measure on the dev sample (ER_CACHE_TAG=v2, --stop-after blocking) as B07 vs the best of Prompt 4.
Report recall, cands/entity, runtime and peak RAM. Keep it only if recall ↑ ≥ 0.5 pt at cands/entity ≤ 70
and runtime stays under ~2× (the full test set is ~8× the dev sample).
```

## Prompt 5 — Stage-1 dev evaluation with new features and data picking
```
With the best blocking config committed, run the stage-1 dev evaluation including transfer:
ER_CACHE_TAG=v2 ER_JOBS=4 ER_LGB_JOBS=12 python -m src.run_pipeline --data ../../dataset --eval-only --no-stage2 --exp-id E04 --change "stage1 + contradiction/acronym feats + data picking"
Compare with E03 (0.94642 all / 0.95648 US / 0.93133 IN / 0.95802 singletons).
Report the breakdown, decoder chosen, US->India and India->US transfer, training time vs E03, and the
top 15 feature importances.
Also test the one-to-one assumption (docs/IDEAS.md B.8): score the best decoder with and without
decide.one_to_one on the same OOF scores and report both (keep hard one-to-one only if it wins). Keep a change only if f05_all improves and transfer doesn't drop by more
than 0.002; otherwise say which part to revert.

Two concrete feature gaps found against docs/EDA_TEAMMATE.md §5/§7 (checked, not yet in features.py):
1. Near-miss number flag — current features have numeric-token Jaccard (f_py col 2) and first-number-
   equal (col 3), but no "numbers present on both sides, close but not equal" flag. This is the
   dominant distractor pattern in their hard_negatives sample ("511 Yacht Drive" vs "513 Yacht Drive",
   "Sn 2101" vs "Sn 2108") — small numeric edit distance, not exact match, not totally different.
   NOT the same as noise.py's existing f_nz_num_conflict (numbers present, disagree, no distance
   notion) -- that already exists and does not cover this, confirmed by reading noise.py directly.
2. Address-missing features are 0, not NaN, when one side has no address (fuzz.ratio("", x) == 0).
   f_addr_empty_{a,b} flags exist so the model can partly compensate, but the raw similarity columns
   still read as "totally different" rather than "no data" — test whether making them NaN (LightGBM
   handles NaN natively) changes anything before assuming the existing flag already covers it.
Add both as E04b if time allows; report whether either moves f05_all.
```

## Prompt 6 — Stage 2 (triangle support) dev evaluation
```
Prerequisite: Prompt 3 must be merged first (src/context.py is a stub on feat/pipeline until then).
Run the stage-2 dev evaluation:
ER_CACHE_TAG=v2 ER_JOBS=4 ER_LGB_JOBS=12 python -m src.run_pipeline --data ../../dataset --eval-only --exp-id E05 --change "stage2 context+competition+triangle"
Compare with E04. Report the breakdown, decoder (threshold vs ef05, with params), transfer, and the
importance of the f2_* features. Recommend keep/drop for stage 2.
```

## Prompt 7 — Error analysis taxonomy
```
Using the latest dev run caches (feats + OOF scores for tag v2), sample 200 errors: 100 false positives
(predicted match, not true) and 100 false negatives (true match missed, including blocking misses).
Show both raw records for each. Classify each into: blocking miss / native script / abbreviation or
alias (dba/aka) / address normalization / name collision (same name, different business) /
number conflict / threshold / other.
Write docs/ERRORS.md with the counts, 3 examples per class, and one concrete fix per class, ranked by
expected gain. Add the fixes to docs/IDEAS.md as queued items. Don't implement anything.
```

## Prompt 7B — Peer-review features (docs/IDEAS.md B.8, ⏭ items)
```
Implement (TDD, docstrings, country never a feature):
1. Retrieval-evidence features: per-channel rank for name/addr/all (missing = K+1) and n_channels
   (how many channels retrieved the pair) — requires blocking to output channel ranks; keep existing columns.
2. Ambiguity/density features: pool count of the exact normalized name_core (have), count sharing the
   first core token, count sharing the street_key; S1-side count of other S1 with the same name_core.
3. A fast dev preset: --frac 0.02 stratified (~40k entities) for quick screening.
Screen on the fast preset vs the previous best (same preset), then confirm the winner on the 10% dev
as E0x. Report both.
```

## Prompt 8 — Normalization v2 (from EDA + error analysis)
```
Confirmed by directly reading src/normalize.py (not just the docs): alias splitting and French
etablissements/ets/bis/ter are genuinely NOT implemented yet -- docs/EDA.md §G7/G10 (our own EDA)
both call for them and both are still open, docs/NEXT_STEPS.md item 10 has the full code-level audit.
The one "ter" match in normalize.py is the unrelated street-type abbreviation ("terrace"->"ter"), not
the French house-number suffix -- don't mistake one for covering the other.

Implement normalization v2 (TDD, docstrings, no external data):
- Split aliases (dba|aka|fka|formerly|formerly known as|nee|doing business as) into a new name_alias
  column; name features use the best of name vs alias.
- French: etablissements/ets as a legal-like token; street key skips bis/ter/quater.
- Any top fixes from docs/ERRORS.md that belong in normalization.
Bump ER_CACHE_TAG to v3 (re-normalization needed, ~10 min). Rerun the dev evaluation (best config) as
E06 and compare with the previous best. Commit.

Resolved by docs/EDA_TEAMMATE.md §9 (their own normalization-collision study, 35k-sample SAMPLE
evidence, not a disagreement to litigate further): conservative normalization (NFKC + casefold +
punctuation/whitespace only) for blocking IDENTITY KEYS, abbreviation-expansion and token-sorting for
FEATURES only — their collision groups grow raw 226 → conservative 389 → abbrev-expanded 429 → token-
sorted 436. Our blocking.py currently uses abbreviation-normalized name_core/legal-split tokens directly
as key families (n, nb, nm, np, nc, n2), which is the opposite of their recommendation. Their IDF
weighting mitigation is a different fix for the same problem, not verified at our recall level — this
v2 pass is also the moment to run the one cheap check: drop the abbreviation-normalized families from
the blocking union (keep them as scoring features only) and compare recall/cands-per-entity against the
Prompt 4 best. Report both configs; keep whichever wins on recall at bounded cands/entity.
```

## Prompt 8A — Re-tune the decoder against E06's features (cheap, do first)
```
Context: the recorded decoder (threshold, t_match=0.7, t_empty=0.3) predates E06's normalization-v2 and
retrieval-evidence/ambiguity features. E05 separately found ef05 (tau=0.8, lam=0.0) beat threshold on
stage-2-only scores. Neither has been checked against E06's actual OOF scores yet.
Using the cached E06 OOF scores (retrain stage 1 + stage 2 if not already cached for the current tag),
run decide.tune() and report the full grid result, not just the winner. Compare the winning decoder's OOF
breakdown (all/US/India/singletons/matched) and US<->India transfer against E06's committed numbers
(0.96817 all, 0.89623 US->India, 0.94474 India->US). Update cache/<tag>_decoder.json and
experiments/log.csv only if the winner changed from what's currently recorded; note E06 either way.
Don't touch features, model, or blocking. This is read-only re-tuning.
```

## Prompt 8B — CORRECTED: no bug exists here; run stage 2 for real on the SageMaker full run
```
Correction (found while starting Prompt 8A): the previous wording of this prompt described
"models2 empty" as an unexplained bug. It is NOT a bug. docs/NEXT_STEPS.md's own section 0 (predates
this correction) already diagnosed it: train.ipynb's cell 5bc14283 launched the MVP run with an explicit
`--no-stage2` flag, on purpose (the MVP path). E05/E06's dev OOF numbers (0.96519/0.96817) already prove
stage 2 trains and scores correctly when enabled -- that's literally how those numbers were produced.
What's actually true: no FULL (non-eval-only, real test-set) run has ever been done with stage 2 enabled
-- every eval-only dev run enabled it, the one full run to date deliberately didn't. This is an unexercised
path, not a broken one.
Action: nothing to fix. When Prompt 9's full SageMaker run happens, do NOT pass --no-stage2 (the default
already trains stage 2) and confirm in the log that "stage2: N features" and a stage-2 fold loop appear
before trusting the run. As light hardening (optional, cheap): add a test asserting model.train_oof never
silently returns an empty models list for a normal (non-degenerate) input, so a real future regression
here would fail loudly instead of quietly falling back to stage-1-only.
```

## Prompt 8C — Joint S2/S3 cluster features (Exp J1; docs/RESEARCH_JOINT_S2S3.md §3 Option A)
```
Read docs/RESEARCH_JOINT_S2S3.md in full first -- it has the design, don't re-derive it.
Implement (TDD, docstrings, OOF-scores only, country never a feature):
1. In src/context.py, add a function that builds a cross-source (is_s3 differs) candidate-similarity
   graph per S1 group, reusing triangle_features' existing name+address blended token_set_ratio metric
   and its sim > 0.8 threshold, then computes connected components (union-find; group sizes are ~40 so
   this is cheap).
2. Add per-candidate features: f2_comp_size, f2_comp_n_sources (1 or 2), f2_comp_mean_p, f2_comp_max_other_p
   (best OTHER member's p in the same component). Wire into add_stage2_features.
3. Retrain stage 2 on the current best config (E06 blocking/features) as exp J1. Compare OOF breakdown
   (all/US/India/singletons/matched) and transfer against E06. Dump feature importances
   (ER_DUMP_IMPORTANCE=1) and report specifically whether f2_comp_* rank above or below the existing
   f2_sim_top_other_src -- this tells us if the new features are redundant with what triangle support
   already gives the model.
Keep only if f05_all improves and transfer doesn't regress beyond the ~0.002 tolerance. Log as J1 in
experiments/log.csv either way (a null result here is a legitimate, useful outcome per the research doc).
```

## Prompt 8D — Joint S2/S3 correlated decision (Exp J2; docs/RESEARCH_JOINT_S2S3.md §3 Option B)
```
Prerequisite: Prompt 8C (J1) done and logged, whatever its outcome.
Read docs/RESEARCH_JOINT_S2S3.md §3 Option B and §4 in full before starting.
Implement in src/decide.py: a new correlated-Monte-Carlo mode for select_expected_f05 that replaces the
independent `hits = rng.random(...) < pb` draw with the two-level draw described there (shared per-cluster
latent z_c ~ Bernoulli(max p in cluster), member draws conditioned on z_c that exactly preserve each
candidate's own marginal p_i). Reuse the same connected-components structure from Prompt 8C (built once
per group from OOF p1/p2, never in-fold -- state this explicitly in the commit message and the
experiments/log.csv change note). Singleton components (no cross-source corroboration) must produce
bit-for-bit identical sampling to the current independent-draw code -- write a test asserting this.
Add `joint: True/False` as a new dimension in decide.tune's grid, exactly parallel to how `one_to_one`
is already gridded. Run as exp J2: report the full grid, the winner, OOF breakdown, and US<->India
transfer vs the best of J1/E06. Keep only if it wins on OOF and doesn't regress transfer beyond ~0.002.
```

## Prompt 8E — Hard-negative mining loop (docs/NEXT_STEPS.md item 12)
```
After Prompt 8C (J1) lands (use whichever stage-2 config won), pull the model's own high-confidence OOF
false positives: feats[(feats[pcol] >= t_match) & (y == 0)] -- same selection notebooks/02_errors.ipynb
already uses for its false-positive view, `pcol` being whichever of p1/p2 the current best config scores
with. Feed this pool into sampling.training_rows as an always-kept, upweighted negative bucket (don't
replace the existing hard/easy negative split, add to it) for one retraining pass, exp id "HN1".
Report: OOF breakdown (all/US/India/singletons/matched) vs the pre-mining baseline, with particular
attention to precision on the matched slice (not just f05_all, since this is a precision-targeted change).
Re-check US<->India transfer explicitly -- flag if hard negatives mined from US/India errors look like
they might be overfitting to US/India-specific false-positive patterns. One iteration only; don't loop
without a fresh HANDOFF.md/log entry per iteration.
```

## Prompt 8F — Blocking recall full-data re-measurement (docs/NEXT_STEPS.md item 13)
```
Context: B05/E06's 0.9523 recall was measured on the dev sample (config.DEV_FRAC). Some of the historical
comparison numbers (docs/EDA_TEAMMATE.md §13) are flagged UNVERIFIED/pilot-scale and acknowledged
optimistic -- don't assume 0.9523 holds at full scale without checking.
Use blocking.recall_at_ks (already built in src/blocking.py, diagnostic-only, not on the default pipeline
path) at the full train S1/pool (frac=1.0 or as large as memory allows on the current machine) instead of
the dev sample. Report pair_recall and cands_per_entity at the current K values plus a couple above/below
for context. If full-scale recall is meaningfully below 0.9523, revisit Prompt 4B's character 3-4-gram
channel (never built -- B07 was never run). If it holds, say so explicitly and close this item out.
```

## Prompt 8G — France-specific robustness hardening (docs/NEXT_STEPS.md item 14)
```
Hard rule: nothing here may read or branch on the `country` column as a model feature -- only vocabulary-
coverage / token-overlap signals computed identically for every record, per CLAUDE.md.
1. Run scripts/france_check.py against the current output/matching_results.tsv (regenerate output/ first
   if it's the empty E00 baseline -- check before trusting it, see SOLUTION_DEEPDIVE.md §12) and report
   its France-vs-US/India recommendation.
2. Test whether leaning stage-2 scoring more on character-level similarity (f_name_norm_jw and rapidfuzz
   char-level scores) relative to token-overlap scores, specifically for pairs where name-token overlap is
   low, changes the OOF breakdown -- implement as a feature interaction or model input, not a
   country-conditioned rule.
3. Test a stricter empty-gate keyed on low name/first-token frequency (existing f_freq_a/f_first_tok_freq_a
   features, a vocabulary-coverage proxy) rather than the flat t_empty threshold.
Report OOF breakdown, transfer, and (if run) the france_check.py comparison for each sub-change. Keep only
changes that improve or hold OOF F0.5 while not regressing transfer.
```

## Prompt 8H — Segment-specific thresholds (docs/NEXT_STEPS.md item 15)
```
Prerequisite: Prompts 8C-8E should be resolved first -- this sharpens whatever the best decoder is at that
point, sequence last among the new differentiator work.
Grid decide.tune's threshold/tau/lam separately per segment, defined ONLY by existing country-agnostic
columns: is_s3, f_addr_empty_a/_b, f_script_a/_b (never country). Report each segment's local-best OOF
F0.5 vs the single global decoder's score on that same segment slice. Keep a segment-specific parameter
only where the local grid beats the global decoder by enough to justify the added decode-path complexity
(state the bar you used). Report the combined OOF breakdown and transfer for the final segmented decoder
vs the single global one.
```

## Prompt 9 — Full run on SageMaker → submission #2
```
On the SageMaker notebook instance (ml.r5.4xlarge, us-east-1): git pull feat/pipeline, pip install -r
requirements.txt, fetch the dataset from S3.
Run the full pipeline with the best dev configuration (stage 2 on if it won), training on TRAIN_FRAC
of train S1:
ER_JOBS=12 ER_LGB_JOBS=16 nohup python -m src.run_pipeline --data ../../dataset --out ../../output --exp-id E07 --change "<best config>" > run.log 2>&1 &
Watch run.log. On "validator exit=0", copy output/*.tsv back (S3), then a human uploads.
Report the dev breakdown from the log, test candidates/entity, and the matched-entity share per country
(France included).
```

## Prompt 10 — France check (spend one upload only if worth it)
```
Compare the France rows of output/matching_results.tsv with US and India: share of entities with
≥1 match, mean matches per entity, and the probability distribution of chosen matches. If France looks
very different (e.g. far fewer matches), propose: (a) keep as is, (b) a France-specific diagnostic
upload with France rows emptied, to measure France's contribution on the public LB. Explain the
trade-off. Don't upload.
```

## Prompt 11 — Push to GitHub
```
Check that https://github.com/shreyshringare/AWS_HACK_TRIP is PRIVATE (python urllib against
api.github.com/repos/shreyshringare/AWS_HACK_TRIP must fail or show private=true). If it's public,
STOP and tell me. If it's private: add it as remote `origin` (if missing), run git status, make sure
dataset/, output/, cache/, submissions/ and *.log are ignored, then push feat/pipeline and main.
Report what was pushed.
```

## Prompt 12 — Packaging (Day 3, after feature freeze 14:00 IST)
```
Follow plan Tasks 19–21 (docs/superpowers/plans/2026-09-25-entity-resolution.md):
1. Write scripts/make_zip.py exactly as in Task 19; build <TEAM>_submission.zip; list its contents.
2. Fill Documentation_template.md (Task 20): core sections ≤ 2 pages (guidelines), details in the
   Appendix. Cite numbers from experiments/log.csv, docs/EDA.md and docs/ERRORS.md. List every
   hand-written dictionary and the learned artifacts (translit table, generic vocab), and state that
   no external data is used and LightGBM is MIT.
3. Unzip into a temp dir and confirm the layout matches the problem statement exactly, and that
   `python -m src.run_pipeline --help` works from the unzipped code/business_entity_resolution.
Commit everything.
```

## Prompt 13 — Final selection (Day 3 evening)
```
From experiments/log.csv and submissions/*/meta.json, pick the final submission. The guidelines say
shortlisting uses BOTH leaderboards and the final rank uses the private one (the unseen part of test),
so weigh: dev OOF F0.5 (largest, most reliable sample), public LB (a subset of the same test
distribution, France included), and US<->India transfer robustness. Prefer a candidate strong on all
three over one that only wins the public LB by a small margin (overfitting risk).
Explain the choice in 5 lines. Make sure output/ holds exactly
that submission's files (restore from submissions/<tag>/ if needed), rerun the validator, rebuild the
zip, and update HANDOFF.md. A human does the final upload and zip submission before 23:59 IST.
```

## Prompt 14 — End of shift (use at the end of every shift)
```
Update HANDOFF.md: Current state table (best dev F0.5, blocking recall, transfer, best public LB,
uploads used today, decoder + params, cache tag, running jobs, instance state), and a new shift entry
(done / next prompt number / blockers). Commit and push (if the remote is private). List anything
still running and whether the SageMaker instance can be stopped.
```
