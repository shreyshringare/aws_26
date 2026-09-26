# Next Steps — combined plan (current: 0.96817 dev OOF / exp E06, 0.942 public LB → target 0.98)

Merges `HANDOFF.md` current state + `docs/IMPROVEMENT_PLAN.md` teammate-EDA findings into one ranked
action list. One change per experiment, log every run in `experiments/log.csv` per `CLAUDE.md`.

**Reality check on the target:** 0.98 macro F0.5 is very high for noisy fuzzy-matched business records —
treat it as "push as far as possible," not a guaranteed number. The honest ceiling depends on blocking
recall (you cannot score what you never shortlist) and on how separable the hard-negative collision
groups actually are. Items 1–3 below are DONE (see the status note before the ranked-order section) — the
real remaining path from 0.96817 upward is items 11–16 (the post-E06 differentiator list). Items 4–10
are refinements layered on top, several already folded into E06.

**Status since this doc was first written:** blocking recall is at 0.9523 (B05/E06, `na` cross keys),
stage 2 (triangle support) is real and kept (E05), normalization v2 (alias split + French
etablissements/bis/ter) landed as E06 — current best, 0.96817 dev OOF, improves singletons and both
transfer directions simultaneously. Public LB is still the pre-E04 E03 number (0.942, one upload spent);
no upload has been spent on E04/E05/E06 yet. **Known blocker carried over from `HANDOFF.md`: the last
full SageMaker run had `models2` come back empty (stage 2 silently disabled at test time) — this must be
fixed before a fresh full run/upload, independent of the new items below.**

---

## 0. Correction to a prior "bug" — read this first

`m2 is None` in the last run is **not a bug**. `train.ipynb` cell 2 has `NO_STAGE2 = True` set on
purpose (MVP path). The 0.94642 score you have is **stage-1 only** — the whole stage-2 group/competition/
triangle-support pass (one of the three differentiators in the design, `docs/IDEAS.md` A.3.1) has never
run on a full model yet. This is the single biggest untapped lever, not a fix — see item 2.

---

## 1. Close the blocking recall gap (highest priority — caps everything downstream)

**Current:** 91.85% recall at 42.5 cands/entity (exp B03). Target ≥ 0.97 per `CLAUDE.md`.
Every true match blocking misses is a match stage-1/stage-2/decision can never recover — this is the
hard ceiling on final F0.5.

**Action** — already speced in `docs/PROMPTS.md` Prompt 4, run in order, one change per run:
```bash
cd code/business_entity_resolution
ER_CACHE_TAG=v2 ER_JOBS=4 python -m src.run_pipeline --data ../../dataset --eval-only \
  --stop-after blocking --exp-id B04 --change "EDA weights only (disable na family)"
ER_CACHE_TAG=v2 ER_JOBS=4 python -m src.run_pipeline --data ../../dataset --eval-only \
  --stop-after blocking --exp-id B05 --change "+ na cross keys"
ER_CACHE_TAG=v2 ER_JOBS=4 python -m src.run_pipeline --data ../../dataset --eval-only \
  --stop-after blocking --exp-id B06 --change "best one + channel k +10 on best-gaining channel"
```
Keep the best config with cands/entity ≤ 60. If recall still < 0.95 after B04–B06, add the character
3–4-gram TF-IDF channel (Prompt 4B, `docs/PROMPTS.md:161`) as B07 — only keep it if recall gains ≥ 0.5pt
at cands/entity ≤ 70 and runtime stays under ~2x.

**Teammate's independent EDA confirms the architecture, not a new blocking method** (full detail now in
`docs/EDA_TEAMMATE.md` §10): their plain deterministic union needs ~11,231 cands/entity for 96.3% recall
(exact name+addr+postal seed = 31.9% @ ~11 cands; + rare address token = 78.7% @ 231; + rare name token =
90.5% @ 2,075) vs our 42.5 for 91.85%. Don't adopt their blocking strategy, just their evidence the
ceiling is reachable — and use their per-channel table as a sanity check on which signal (rare address
token > rare name token > exact channels) carries the most recall per candidate, in case B04-B06 weight
tuning should lean the same direction.

---

## 2. Turn stage 2 on and run the real dev evaluation

**Action:**
```bash
cd code/business_entity_resolution
ER_CACHE_TAG=v2 ER_JOBS=4 ER_LGB_JOBS=12 python -m src.run_pipeline --data ../../dataset --eval-only \
  --exp-id E04 --change "stage1 + stage2 (NO_STAGE2=False) on best blocking config"
```
(Or in the notebook: set `NO_STAGE2 = False` in cell 2, re-run from cell 16.) Compare against E03
(0.94642 all / 0.95648 US / 0.93133 IN / 0.95802 singletons). Report the full breakdown, decoder chosen,
US→India and India→US transfer (currently uncomputed — `NO_TRANSFER` flag), training time vs E03, top 15
feature importances. Keep the change only if f05_all improves **and** transfer doesn't drop > 0.002.

Also test the one-to-one assumption (`docs/IDEAS.md` B.8): score the best decoder with and without
`decide.one_to_one` on the same OOF scores, keep hard one-to-one only if it wins.

---

## 3. Re-tune the decision layer once stage 2 is real

Current decoder (threshold, t_match=0.7, t_empty=0.3) was tuned against stage-1-only scores. Once E04
(stage-1 + stage-2) exists, re-run `decide.tune()` — grid already covers both threshold and EF05
(expected-F0.5 Monte Carlo) decoders; EF05 is the more metric-aware option and may win once stage-2 scores
are more separated. Don't assume the old params still apply.

---

## 4. Ingest teammate's hard-negative / collision examples

**Finding:** hard negatives average name similarity 0.872 — close to median positive similarity
(0.883 S1-S2 / 0.922 S1-S3), meaning a name-only threshold would false-merge constantly.
`hard_negatives.csv` / `false_collision_examples.csv` are concrete already-computed failure cases, not yet
in this repo.

**Action:**
- Get the teammate to push `hard_negatives.csv`, `easy_negatives.csv`, `false_collision_examples.csv`,
  `positive_pairs.tsv.gz` to `s3://aws-hack-sagemaker/eda-teammate/` (bucket exists; confirm their IAM
  role can write to it first — we hit the same "Unable to locate credentials" issue on this shared
  bucket, check role policy before assuming their push works).
- Diff those hard negatives against what the stratified sampling in the unmerged
  `worktree-agent-a0b1edbe6ec982855` branch (`src/sampling.py`) actually selects for training. Gap found →
  concrete experiment (e.g. "oversample name-similarity > 0.85, non-match pairs").

---

## 5. Fix concrete collision groups found in teammate's EDA

Named generic/franchise collisions (`docs/EDA_TEAMMATE.md` §9): "physical therapy" (1,042 candidate
records), "family center" (459), "meridian" (409), "eye group" (378). ("pediatric group (354)" from the
original summary isn't in the full write-up's list — treat as unconfirmed until re-checked.)

**Action:** check each against the learned generic vocab (`cache/v1_generic_vocab.json` /
`get_generic_vocab` output). Any not already flagged as generic add-on tokens → normalization/feature gap,
fix before next stage-1 dev run. Fold into the error-analysis taxonomy (`docs/PROMPTS.md` Prompt 7) as
real "name collision, different business" test cases.

---

## 6. Normalization v2 (bump `ER_CACHE_TAG=v2`)

- **Alias split**: `dba|aka|fka|formerly|formerly known as|nee|doing business as` → separate
  `name_alias` column, score best-of name vs alias similarity.
- **French**: `etablissements/ets` as legal-like token; street key skip `bis/ter`; seed French generic
  words (ecole, amicale, groupe, association…) only if learned vocab misses them.
- **Generic vocab**: require ≥ 3-char tokens.
- **Features**: drop postcode features (≤ 1.9% coverage, dead weight); add address-shape features (token
  count; landmark flag already exists); alias-aware name similarity.

**Note a tension, don't silently resolve it:** "drop postcode features" above is from our own
`docs/EDA.md` (postcode *feature* coverage ≤1.9%, dead as a scoring signal). Teammate's
`docs/EDA_TEAMMATE.md` §10.1 measures postal/PIN as a *blocking channel* — 4.8% recall at 3.2
cands/entity, "cheap, small gain" — a different question (candidate generation vs pair scoring). Dropping
the postcode *feature* and keeping (or not) a postcode *blocking channel* are independent decisions;
don't let one justify skipping the other without checking blocking recall impact separately.

Why this matters for the 0.98 push specifically: France is 15% of test S1, absent from train, and 45% of
its name tokens are unseen (vs 28–36% in the US↔IN transfer proxy) — the proxy is optimistic for France.
Char-level / alias-aware similarity is the mitigation.

---

## 7. Normalization-as-identity-key — now resolved, has a concrete experiment

Was an open disagreement; `docs/EDA_TEAMMATE.md` §9 (their own normalization-collision study) resolves
it with methodology, not just a headline number: raw 226 → conservative-normalized 389 → abbreviation-
expanded 429 → token-sorted 436 collision groups. Their explicit recommendation: conservative
normalization (NFKC + casefold + punctuation/whitespace) for blocking *identity keys*, abbreviation
expansion / token-sorting for *scoring features only*. Our `blocking.py` currently does the opposite —
abbreviation-normalized `name_core`/`legal`-split tokens are blocking key families (`n`, `nb`, `nm`,
`np`, `nc`, `n2`) directly, mitigated instead by IDF weighting + `BLOCK_MAX_DF=2000`.

**Action (folded into Prompt 8, no separate sync needed now):** after the Prompt 4 blocking work lands,
run the blocking union with abbreviation-normalized families removed from the key set (keep them as
scoring features only) and compare recall/cands-per-entity against the Prompt-4 best. Keep whichever
wins. Still worth telling the teammate we're testing their exact recommendation.

---

## 8. Singleton false-positive risk — decision-layer design constraint

`docs/EDA_TEAMMATE.md` §8 (EXACT, all 123,247 train singletons): 37.8% of singletons have an *exact*
same-name candidate in the same country; only 3.0% have an exact same-address candidate. An exact name
match is explicitly **not** evidence of identity for a no-match entity.

**Action:** when re-tuning the decoder after stage 2 lands (item 3), check the singleton slice
specifically — the empty gate should fire on absence of address/number support, not on name similarity
alone. This is a constraint on `decide.py`'s design, not just a feature idea.

---

## 9. Two concrete feature gaps (checked against our actual `features.py`, not just their numbers)

Cross-checked teammate's §5/§7 recommendations against what's implemented. Most already exist (`is_s3`,
legal-form agreement, name IDF, numeric Jaccard, address-empty flags) — two real gaps:

1. **Near-miss number flag** — not implemented. Their dominant distractor pattern (`hard_negatives.csv`)
   is small numeric edit distance ("511 Yacht Drive" vs "513 Yacht Drive"), which current features don't
   distinguish from "numbers agree" or "no numbers present." **Not the same as `f_nz_num_conflict`**
   (already in `noise.py` — numbers present, disagree, no distance notion): near-miss needs the actual
   edit distance between numeric tokens, conflict just needs "not equal." Don't skip this thinking
   conflict already covers it.
2. **Address-missing scores 0, not NaN**, in similarity features when one side has no address. The
   `f_addr_empty_{a,b}` flags exist so the model can partly compensate, but worth testing whether NaN
   (LightGBM handles it natively) changes anything.

**Action:** optional E04b step alongside item 2/3 (`docs/PROMPTS.md` Prompt 5) — add both, report
whether either moves f05_all before committing.

---

## 10. Our own `docs/EDA.md` §G cross-checked against code (not just teammate's EDA)

Same audit as item 9, but against our own EDA's 11 implications (`docs/EDA.md` §G) instead of the
teammate's. Verified by reading the actual source, not assuming from the doc:

**Already implemented, confirmed in code:**
- k-bucket dev/OOF stratification (`src/sampling.py::match_bin`) — G1
- Global one-to-one + competition feature (`decide.one_to_one`, `context.py`), hard-negative training
  rows via `sampling.py::training_rows` — G2
- Address agreement + name-frequency features — G3
- All noise-channel features (`f_nz_*`: digit deletion, num_missing_one, generic-only, legal-moved,
  edit1, addr-only, accent-only, addr-norm-only, num_conflict, acronym) — G4
- Address-shape features instead of country (`f_land`, `f_script`, `f_ntok`, `f_addr_empty`) — G5
- Postcode dropped entirely (no `p` blocking family, no postcode feature group) — G6
- Blocking family reweighting (`a`/`ab` up, `nm` down) matches `config.BLOCK_W` exactly — G8
- Name-frequency/IDF computed per-split pool (test pool when scoring test, not baked from train
  vocab) — `add_frequency_columns` takes `pooln` as a parameter, not a fixed train reference — G9

**Still gaps — same two Prompt 8 already targets, now confirmed as real (not already quietly done):**
- **Alias splitting** (`dba`/`aka`/`fka`/`formerly`) — not in `normalize.py`. Generic-vocab min-length-3
  filter *is* done, the alias-column split isn't — G7.
- **French `etablissements/ets` legal prefix, `bis`/`ter` house-number skip** — not in `normalize.py`.
  (A `"ter"` match exists but it's the unrelated street-type abbreviation "terrace"→"ter", not the
  French house-number suffix — don't mistake one for covering the other.) — G10.

**Action:** no new work beyond what Prompt 8 already plans — this just confirms Prompt 8's scope is
accurate and these two items haven't silently landed elsewhere in the codebase.

---

## 11. Joint S2↔S3 cluster consistency at decision time (`docs/IDEAS.md` B.6, 🧪 — biggest unexplored lever)

**Finding:** stage-2 triangle support (`context.py`) already computes cross-source corroboration as
*features* into the stage-2 model, but the decoder (`decide.select_expected_f05`) still samples each
candidate's inclusion as an **independent** Bernoulli draw — the correlation between mutually-supporting
S2/S3 candidates never reaches the actual decision. Full design researched and written up in
`docs/RESEARCH_JOINT_S2S3.md` — don't re-derive it, read that doc first.

**Action — two experiments, in order (one-change-per-experiment discipline):**
1. **Exp J1** (cheap, low-risk): add cluster-level features to `context.py` — connected components of a
   cross-source (`is_s3` differs) candidate-similarity graph (reuse the `sim > 0.8` threshold triangle
   features already use), then `f2_comp_size` / `f2_comp_n_sources` / `f2_comp_mean_p` /
   `f2_comp_max_other_p` per candidate. Retrain stage 2, compare against E06 on OOF F0.5 breakdown +
   transfer, and check `ER_DUMP_IMPORTANCE` to see whether this is redundant with the existing
   `f2_sim_top_other_src` feature (a real possibility — measure, don't assume).
2. **Exp J2** (the actual decision-time mechanism, gated on J1 not regressing anything): replace
   `select_expected_f05`'s independent Monte Carlo draw with the two-level correlated draw in
   `docs/RESEARCH_JOINT_S2S3.md` §3 Option B (shared per-cluster latent `z_c`, member draws that
   preserve each candidate's own marginal `p_i` exactly). Singleton components degrade to *identical*
   current behavior by construction, so this can't uniquely hurt singletons/uncorroborated entities. Grid
   it into `decide.tune` as a new dimension (`joint: True/False`) exactly like `one_to_one` already is;
   keep only if it beats the current best decoder on OOF **and** doesn't regress US↔India transfer beyond
   the ~0.002 tolerance already precedented for E05.

**Why it's ranked highest:** directly extends an already-built differentiator (triangle support) into
territory most fuzzy-match competitors won't touch at all (decision-time joint reasoning, not just
pairwise scoring). Evidence it should help: mean 3.46 matches/entity spread across both S2 and S3
(`docs/EDA.md` §B), and concrete `docs/ERRORS.md` cases (the 420/433 vs 100/113 number-conflict
inconsistency, the `Costa Pediatrics` ambiguous threshold case) that look exactly like missing
corroboration signal at decode time.

## 12. Hard-negative mining loop (metric-native precision lever)

**Idea:** F0.5 weights precision 2× over recall. The most metric-native way to improve is to make the
model's own confident mistakes harder to make again — retrain with the model's high-confidence OOF false
positives fed back in as additional (weighted) hard negatives, rather than relying on generic feature
engineering to fix precision indirectly.

**Action:** after an OOF run, pull `feats[(feats["p1"]/"p2"] >= t_match) & (y == 0)]` (same selection
`02_errors.ipynb` already uses for its false-positive view) as the hard-negative pool. Feed it back into
`sampling.training_rows` as an always-kept (never subsampled) negative bucket, upweighted, for the next
training pass. One iteration first (don't loop indefinitely — each iteration needs its own
`experiments/log.csv` row and OOF re-check); measure precision specifically (not just f05_all) on the
matched slice, and re-check transfer — hard negatives mined from US/India OOF errors could plausibly
overfit to US/India-specific false-positive patterns that don't generalize to France, so this is exactly
the kind of change the transfer-robustness rule exists to catch.

## 13a. NEW (organizer email, 36h mark): candidate-set size is now its own ranking axis

Amazon ML Challenge 2026 organizer update (received during the challenge, not in the original
problem statement CLAUDE.md was written from): `candidate_pairs.tsv` and the blocking code that
produces it are reviewed for final ranking **independent of the public/private leaderboard F0.5
score**. "The approach that generates a smaller candidate set per Source 1 entity will be ranked
higher in the final evaluation." This is a real change to the optimization target for blocking:
previously cands/entity was only a soft bound ("tens, not hundreds"); it's now explicitly scored.

**Current state**: E06 blocking = 0.9523 recall at 42.6 cands/entity (`BLOCK_CHANNELS`: name top-20,
addr top-20, all top-30, unioned). Never been tuned for *minimum* candidate count at a given
recall floor -- only bounded from above.

**Open question, needs a human call, not a blind retune**: how much recall are we willing to trade
for a smaller candidate set, given this is now separately rewarded but the exact scoring weight
Amazon puts on it vs. the F0.5 score is unknown? Candidates for a future prompt once this is
decided: sweep per-channel top-k downward (e.g. 20/20/30 -> 12/12/18) and report the
recall-vs-cands/entity curve so the team can pick a point on that curve deliberately, rather than
just re-running the existing recall-maximizing tune.

## 13. Blocking recall → 0.97 on full data (still the single largest measured lever)

**Status:** B05/E06 sits at 0.9523 recall (dev sample, `FRAC` per `config.DEV_FRAC`/`TRAIN_FRAC`), short
of the 0.97 `CLAUDE.md` target. The pilot numbers behind some of this (see `docs/EDA_TEAMMATE.md` §13
UNVERIFIED tags) were measured against a small distractor pool and are acknowledged optimistic — a
full-data re-measurement is needed before concluding recall is "close enough."

**Action:** re-run `blocking.recall_at_ks` (already built, diagnostic-only, not on the default path) at
full `TRAIN_FRAC`/on the full train pool rather than the dev sample, to get a recall number not subject
to the small-distractor-pool optimism bias. If it's meaningfully lower than 0.9523 at full scale, revisit
Prompt 4B's character 3-4-gram channel (still not built — B07 was never run, current recall cleared the
0.95 skip threshold on the dev sample only).

## 14. France-specific robustness hardening

**Finding:** France is 15% of test S1, completely absent from train, and 45% of its name tokens are
unseen in train vocabulary (vs 28-36% in the US↔India transfer proxy that's supposed to stand in for it)
— the proxy is measurably optimistic for France specifically. Since *no* competitor trained on France
either, a team that specifically hardens this segment gains disproportionately: everyone is weak here,
but not everyone will have bothered to check.

**Action (three concrete, country-never-a-feature-compliant options):**
1. Char-level similarity weighting: when name-token overlap is low (a proxy for high OOV rate, computable
   without reading country), lean the decision more on `f_name_norm_jw`/rapidfuzz character-level scores
   relative to token-overlap scores — test as a feature interaction or a stage-2 adjustment, not a
   country-conditioned rule.
2. A stricter empty-gate for candidates with low name-token coverage against the *pool's* vocabulary
   (computable per-record via `first_tok_freq`/`name_freq`, already existing features) — again a
   vocabulary-coverage signal, not a country lookup.
3. Adversarial train-vs-test-France validation: since France can't be held out during training the way
   US/India can (it's not in train at all), the best available check is `scripts/france_check.py`'s
   match-rate / mean-matches-per-matched comparison against US/India on live output — run it after every
   significant model change, not just once before an upload.

**Hard rule reminder:** none of the above may read or branch on the `country` column as a model feature
— only as a blocking key, per `CLAUDE.md`. Vocabulary-coverage and token-overlap signals are legitimate
because they're computed the same way for every record regardless of country label.

## 15. Segment-specific thresholds (S2 vs S3, missing-address, native-script)

**Idea:** `docs/IDEAS.md` B.6 already flags this: sharpen precision on known-hard segments — S2 vs S3
(different noise profiles), missing pool address, native-script pairs — with segment-specific decoder
parameters, explicitly **not** per-country segments (France would have no train-side segment to tune
against, so a country-keyed threshold is untunable and would violate the country-feature spirit even if
technically legal since country is only ever a blocking key here).

**Action:** after Exp J1/J2 and the hard-negative loop land, grid `decide.tune`'s threshold/tau/lam
separately per segment (defined by existing country-agnostic columns: `is_s3`, `f_addr_empty_a`/`_b`,
`f_script_a`/`_b`) instead of one global decoder, using the same OOF discipline. Keep only segments where
the local grid beats the global decoder's OOF score on that segment by enough to be worth the added
decode-path complexity.

## 16. Ensembling / seed averaging (low priority — Day 3 if time allows)

Cheap variance reduction (`docs/IDEAS.md` B.5 already flags this as 🧪, "Day 3 if time allows" tier):
average predictions across a few different `SEED` values or fold splits. Genuine differentiation
potential is low relative to items 11-15 above — only worth spending time on once those are exhausted or
if a shift has slack time near the end.

## Ranked order for a friend picking this up cold

**Status update:** items 1, 2, 3, 6 and 10 are DONE. Item 1 (blocking recall) landed at 0.9534/0.9523
(B05/E06). Item 2 (stage-2 merge + real dev eval) landed as E05 (0.96519, kept — beats E04 on every
metric). Item 3 (decoder re-tune) happened for E05 (ef05, tau=0.8, lam=0.0 — first time ef05 beat
threshold on OOF) though it hasn't been re-run since E06's feature changes — see the new item 1 below.
Item 6 (normalization v2: alias split + French etablissements/bis-ter) landed as E06, current best
(0.96817), improving f05_all, singletons and both transfer directions simultaneously. Item 10 (own-EDA
audit) is fully resolved. Items 4, 5, 7, 8, 9 remain open refinements (see below); items 11-16 are the
new post-E06 priority list and are now the main path forward.

1. **Re-tune the decoder against E06's features** (item 3, carried forward) — cheap, do this before
   anything else below: `decide.tune` on E06's OOF scores, confirm whether ef05 tau=0.8/lam=0.0 (E05's
   winner) still wins or whether E06's alias/frequency features shift the optimum. Near-zero-cost check
   that should happen before investing in items 11-16.
2. **Item 11 — Joint S2↔S3 cluster consistency (Exp J1 then J2).** Ranked highest of the new items: most
   novel, extends an already-built differentiator, has concrete evidence it should help
   (`docs/RESEARCH_JOINT_S2S3.md`). Start with J1 (cheap feature-only variant) before attempting J2
   (decoder surgery).
3. ~~"Fix the stage-2-silently-disabled bug"~~ **CORRECTED, not a bug** — see Warning 13. The one full
   run to date deliberately passed `--no-stage2`; E05/E06 already prove stage 2 trains and scores
   correctly when enabled. Nothing to fix — just don't pass `--no-stage2` on the next full run (Prompt 9).
4. **Item 13 — blocking recall full-data re-measurement.** Confirms whether 0.9523 holds at scale before
   trusting it as "close enough" to the 0.97 target.
5. **Item 12 — hard-negative mining loop.** Metric-native precision lever; do after item 11's J1 lands so
   the hard-negative pool reflects the best available stage-2 model, not a stale one.
6. **Item 14 — France-specific robustness hardening.** Do this before spending a second leaderboard
   upload — everyone's weak on France, so this is where disproportionate leaderboard gain lives, and it's
   cheap to check (`scripts/france_check.py` already exists).
7. **Item 15 — segment-specific thresholds.** Sequence last among the new items — sharpens what's already
   there rather than adding new signal, most value once items 11/12 have changed what "the best decoder"
   even is.
8. Item 5 (generic-vocab collision check — cheap, read-only, do any time, hasn't blocked anything).
9. ~~Item 9 near-miss number flag~~ DONE — `f_nz_num_near_miss` (noise.py), confirmed not covered by
   E06's retrieval-evidence features before implementing (grepped for it, genuinely missing). Item 9's
   other half (NaN-vs-0 address feature test) is still open — cheap, still untested.
10. Item 7 (normalization-as-identity-key experiment, `ER_NAME_KEY_MODE=features_only` toggle — built in
    `config.py` but never measured against the current default, per `SOLUTION_DEEPDIVE.md` §4).
11. Item 4 (hard-negative ingestion from teammate's EDA — blocked on teammate pushing files; distinct
    from item 12's self-mined hard negatives, both worth doing, this one waits on someone else).
12. Item 16 (ensembling/seed-averaging — lowest priority, Day 3 filler only).

Every item: one experiment, one `experiments/log.csv` row, commit before the next change. Only spend a
leaderboard upload on a meaningful local OOF gain that also doesn't hurt US↔IN transfer.

---

## Warnings (read before touching anything)

1. **0.98 is aggressive** — treat as "push as far as possible," not a guaranteed number. Blocking
   recall is the real ceiling: you cannot score a match that was never shortlisted.
2. **`models2 is None` is NOT a bug** — `NO_STAGE2 = True` in `train.ipynb` cell 2 was set deliberately
   for the MVP path. Don't "fix" it as an error; just flip the flag (item 2).
3. **0.94642 is stage-1 only.** The real stage-2 pass (group/competition/triangle-support) has never
   run. Don't treat the current number as this architecture's ceiling.
4. Item 2 (stage 2 on): keep the change **only if** f05_all improves **and** transfer doesn't drop
   more than 0.002.
5. Item 4B / B07 (char n-gram blocking channel): keep **only if** recall gain ≥ 0.5pt at cands/entity
   ≤ 70 **and** runtime stays under ~2x (the full test set is ~8x the dev sample).
6. Item 4 (teammate's S3 push): their IAM role may hit the **same "Unable to locate credentials"**
   issue we hit on `aws-hack-sagemaker` — confirm their role can actually write before assuming the
   push worked.
7. Item 7 (normalization-as-identity-key): the teammate's collision-group numbers (226→429) came from
   a raw-sample collision-count study, **not** a retrieval-recall test through our IDF-weighted union —
   don't assume their conclusion transfers 1:1, still run the one-experiment check (remove abbrev
   families from blocking keys, compare recall) rather than trusting the analogy blind.
8. Decoder re-tune (item 3): old t_match=0.7 / t_empty=0.3 was tuned against stage-1-only scores —
   don't assume it still holds once stage-2 scores exist. Also fold in item 8's singleton finding:
   37.8% of singletons have an exact-name false lead, so the empty gate must key off address/number
   support, not name similarity.
9. **France risk:** the US↔IN transfer proxy is optimistic for France — 45% of France name tokens are
   unseen vs 28–36% in the proxy.
10. **Don't cite PILOT or UNVERIFIED numbers** from `docs/EDA_TEAMMATE.md` in the methodology doc
    without re-measuring: the 99.7% recall @ K=25 retrieval pilot used only ~4,000 random background
    distractors (real test competes against millions), and the exclusivity-gain (+0.005) used the same
    easy pilot setup — both are optimistic. Indic-script-share and URL-substitution numbers are
    hard-coded strings with no computing code behind them (UNVERIFIED tag, §13 there).
11. **Postcode tension** (item 6): our own EDA says drop the postcode *feature* (dead, ≤1.9% coverage);
    teammate's EDA measures postal/PIN as a *blocking channel* (4.8% recall, small but nonzero) — these
    are different questions, don't let one decision imply the other.
12. General: one change per experiment, log every run, only spend a leaderboard upload on a meaningful
    local gain that doesn't also hurt transfer.
13. **CORRECTED — `models2 is None` was never an unexplained bug.** An earlier pass at this doc
    (and `SOLUTION_DEEPDIVE.md`) treated it as an open blocker needing investigation. It isn't: the one
    full (non-eval-only) run to date explicitly passed `--no-stage2` (`train.ipynb` cell `5bc14283`),
    deliberately, for the MVP path — same as Warning 2. E05/E06's dev OOF numbers already prove stage 2
    trains and scores correctly when enabled. What's actually true is narrower: no full run has *ever*
    had stage 2 enabled — that's an unexercised path, not a broken one. Nothing to fix; just don't pass
    `--no-stage2` on the next full run.
14. **Item 11 (Exp J2) leakage discipline:** the correlated-decoder mechanism in
    `docs/RESEARCH_JOINT_S2S3.md` must build its similarity graph from OOF `p1`/`p2` only, exactly like
    `triangle_features` already does — easy to accidentally wire up with in-fold scores during a refactor
    of `decide.py`. State explicitly in the `experiments/log.csv` `change` note that OOF-only scores were
    used.
15. **Item 12 (hard-negative mining) transfer risk:** hard negatives mined from US/India OOF errors may
    not represent France's failure modes at all — re-check US↔India transfer after each mining iteration,
    same tolerance rule as everything else (~0.002).
16. **Item 14 (France hardening):** every mitigation must be computable identically regardless of country
    label (vocabulary coverage, token overlap — not a `country == "France"` branch anywhere). Re-read
    `CLAUDE.md`'s hard rule before implementing any of item 14's three options.

## Leads (follow-up pointers)

1. Item 1 — run `docs/PROMPTS.md` Prompt 4 (B04/B05/B06), `ER_CACHE_TAG=v2`.
2. Item 1 fallback — if recall still < 0.95 after B04–B06, run Prompt 4B (char 3–4-gram TF-IDF
   channel, B07).
3. Item 2 — Prompt 5 config: stage 2 on, `--exp-id E04`, compare against E03 baseline numbers.
4. Item 2 — also test the one-to-one assumption (`docs/IDEAS.md` B.8): score with/without
   `decide.one_to_one` on the same OOF scores, keep hard one-to-one only if it wins.
5. Item 4 — ask the teammate to push `hard_negatives.csv`, `easy_negatives.csv`,
   `false_collision_examples.csv`, `positive_pairs.tsv.gz` to `s3://aws-hack-sagemaker/eda-teammate/`.
6. Item 4 — diff those against the unmerged `worktree-agent-a0b1edbe6ec982855` branch's
   `src/sampling.py` stratification to find sampling gaps.
7. Item 5 — check named collision terms ("physical therapy", "family center", "meridian", "eye
   group", "pediatric group") against `cache/v1_generic_vocab.json`.
8. Item 5 — fold findings into the error taxonomy, `docs/PROMPTS.md` Prompt 7 (`docs/ERRORS.md` not
   yet written).
9. Item 7 — run the blocking union with abbreviation-normalized families removed as part of
   `docs/PROMPTS.md` Prompt 8, compare recall/cands-per-entity against the Prompt-4 best; tell the
   teammate we're testing their exact recommendation.
10. Item 8 — when re-tuning the decoder (item 3), specifically check the singleton f05 slice against
    the empty-gate behavior, per `docs/EDA_TEAMMATE.md` §8.
11. Item 9 — implement near-miss number flag + NaN-address features as optional E04b in
    `docs/PROMPTS.md` Prompt 5, report impact before keeping.
12. Full underlying evidence for all of the above: `docs/EDA_TEAMMATE.md` (48-file EDA write-up,
    evidence-level tags in §0/§13).
13. Item 11 — read `docs/RESEARCH_JOINT_S2S3.md` in full before starting; run `docs/PROMPTS.md`
    Prompt 8C (Exp J1) then Prompt 8D (Exp J2).
14. Item 12 — run `docs/PROMPTS.md` Prompt 8E; pull the hard-negative pool from the same
    `p >= t_match & y == 0` selection `notebooks/02_errors.ipynb` already uses.
15. Item 13 — run `docs/PROMPTS.md` Prompt 8F; use `blocking.recall_at_ks` (already built,
    `src/blocking.py`) at full scale rather than re-running `generate_candidates` per K.
16. Item 14 — run `docs/PROMPTS.md` Prompt 8G; `scripts/france_check.py` already exists and does the
    match-rate/mean-matches comparison, just needs running after each relevant change.
17. Item 15 — run `docs/PROMPTS.md` Prompt 8H once items 11/12 have landed, not before.
