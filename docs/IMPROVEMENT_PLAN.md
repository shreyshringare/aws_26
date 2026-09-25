# Improvement Plan — from teammate EDA cross-validation

Source: independent EDA by a teammate (full dataset, deterministic blocking benchmark + normalization +
ground-truth topology). Numbers cross-validated against our own `docs/EDA.md` and found consistent
(singleton rate 5.58% vs our 5.6%, mean matches/S1 3.46 vs our ~3.5, max 11, France ~15% of test/absent
from train — all match). Treated as real evidence, not just a second opinion. The **full** report (48
files, all sections, evidence-level tags) is now written up in `docs/EDA_TEAMMATE.md` — this document
summarizes the actionable deltas; go there for the underlying tables and the EXACT/SAMPLE/PILOT/
UNVERIFIED evidence audit (§0 and §13 there) before quoting any number in the methodology doc. Raw data
artifacts (`hard_negatives.csv`, `positive_pairs.tsv.gz`, `false_collision_examples.csv`,
`singleton_candidate_load_all.csv.gz`, `analysis/` scripts — full index in `docs/EDA_TEAMMATE.md`
Appendix) are still NOT in this repo — see item 5.

Ranked by expected impact. One change per experiment per `CLAUDE.md`; log every run in
`experiments/log.csv`.

---

## 1. Close the blocking recall gap (highest priority, unblocks everything downstream)

**Finding:** Our IDF-weighted sparse union (blocking.py) gets 91.85% recall at 42.5 cands/entity (B03).
Teammate's plain deterministic union caps at 96.31% recall but needs 11,231 cands/entity to get there —
confirms our sparse-retrieval architecture is the right one (~260x more efficient), but we haven't hit
the 0.97 target yet either.

**Action:** This is already queued as B04-B06 in `docs/PROMPTS.md` Prompt 4 and `docs/IDEAS.md` B.1
(EDA-weighted family re-weighting, name×address cross keys, wider top-K). No new work needed here beyond
executing what's already planned — this EDA is confirmation to prioritize it, not a new idea.

**Owner/next step:** Run Prompt 4 (blocking experiments) once the current MVP submission is uploaded and
the worktree branch is merged.

---

## 2. Ingest teammate's negative/collision examples into our hard-negative sampling

**Finding:** Hard negatives average name similarity 0.872 — *higher* than the median positive name
similarity (0.883 S1-S2, 0.922 S1-S3 per teammate's numbers) is close enough that a name-only threshold
would false-merge constantly. `hard_negatives.csv` and `false_collision_examples.csv` are concrete,
already-computed examples of this failure mode.

**Action:**
- Ask the teammate to push `hard_negatives.csv`, `easy_negatives.csv`, `false_collision_examples.csv`,
  and `positive_pairs.tsv.gz` somewhere shared (the S3 bucket, or their own branch) — the summary alone
  isn't enough to act on, we need the rows.
- Cross-check these against the stratified hard-negative sampling already built in the unmerged
  `worktree-agent-a0b1edbe6ec982855` branch (`src/sampling.py`) — see if it already captures the same
  hard cases, or if teammate's examples reveal gaps (e.g. a name-similarity band we're under-sampling).

**Owner/next step:** After merging the worktree branch (Prompt 3), diff teammate's hard negatives against
what `src/sampling.py`'s stratification actually selects for training. If there's a gap, that's a
concrete B-series experiment (e.g. "oversample name-similarity > 0.85, non-match pairs").

---

## 3. Add named collision-group examples to the error taxonomy

**Finding:** Concrete high-volume generic/franchise name collision groups: "physical therapy" (1,042
candidate records), "family center" (459), "meridian" (409), "eye group" (378), "pediatric group" (354).

**Action:**
- Check each term against our learned generic vocab (`get_generic_vocab` output, cached at
  `cache/v1_generic_vocab.json`) — if any of these aren't already flagged as generic add-on tokens, that's
  a normalization/feature gap worth fixing before the next stage-1 dev run.
- Fold these into the error-analysis taxonomy work already queued (`docs/PROMPTS.md` Prompt 7,
  `docs/ERRORS.md` — not yet written) as real test cases for the "name collision (same name, different
  business)" error class.

**Owner/next step:** Quick check against `v1_generic_vocab.json` can happen any time (read-only, no
pipeline run needed). Full taxonomy work waits for Prompt 7.

---

## 4. Normalization-as-identity-key — now resolved, has a concrete experiment (was: needs discussion)

**Finding, now with full methodology (`docs/EDA_TEAMMATE.md` §9, their own normalization-collision
study, SAMPLE, 35,000 records):** their collision groups grow raw 226 → conservative (NFKC + casefold +
punctuation/whitespace) 389 → abbreviation-expanded 429 → token-sorted 436. Their own explicit
recommendation: conservative normalization for blocking *identity keys*, abbreviation expansion and
token-sorting for *scoring features only*. Our `blocking.py` does the opposite — abbreviation-normalized
`name_core`/`legal`-split tokens are blocking key families (`n`, `nb`, `nm`, `np`, `nc`, `n2`) directly.

This is a real architectural difference, not obviously wrong on our side — our blocking weights keys by
IDF rarity and drops keys with document frequency > `BLOCK_MAX_DF=2000` (`config.py`), a different
mitigation for the same collision-inflation problem. But their study was measured on **raw sample
collision-group counts**, not on retrieval recall through an IDF-weighted union — it doesn't directly
prove our approach is wrong, just that the risk they're describing is real and measured.

**Action — no longer blocked on a teammate sync, this is now a one-experiment check** (also written up
as part of `docs/PROMPTS.md` Prompt 8): after the Prompt 4 blocking work lands, run the blocking union
with abbreviation-normalized families removed (keep them as scoring features only, per their
recommendation) and compare recall/cands-per-entity against the Prompt-4 best. Keep whichever wins.

**Owner/next step:** Fold into the Prompt 8 (normalization v2) run — no separate sync needed now that
their full methodology is documented, though still worth telling them we're testing their exact
recommendation.

---

## 5. Get the raw artifacts into the shared bucket

Everything above except item 1 is blocked on actually having the teammate's generated files, not just
the summary numbers. `aws-hack-sagemaker` (the shared S3 bucket) already exists — ask them to
`aws s3 sync analysis/ s3://aws-hack-sagemaker/eda-teammate/` (or similar) once their bucket-policy access
is confirmed working. Full expected file list is in `docs/EDA_TEAMMATE.md` Appendix (48 files): the CSVs/
tables referenced throughout that doc, plus `positive_pairs.tsv.gz` (336 MB, not yet opened by anyone),
`cache/token_stats.json`, and the `analysis/*.py` scripts themselves (`run_eda.py` is their main exact
pipeline; the Brahmi transliterator lives in `run_retrieval_benchmark.py`).

## 6. Singleton false-positive risk — feeds the empty-gate design directly

**Finding (`docs/EDA_TEAMMATE.md` §8, EXACT, full 123,247 train singletons):** 37.8% of singletons have
an *exact* same-name candidate in the same country, and only 3.0% have an exact same-address candidate.
`hard_singletons.csv` shows deceptive best-candidates with name similarity 0.83–0.99. An exact name match
is explicitly **not** evidence of identity for a no-match entity — address and numeric agreement have to
confirm it.

**Action:** when re-tuning the decision layer (empty gate / t_empty) after stage 2 lands, check the
singleton slice specifically against this — the gate should fire on absence of address/number support,
not on name-similarity alone. This is a design constraint for `decide.py`, not just a feature idea.

**Owner/next step:** fold into the decoder re-tune step once stage-2 (`docs/NEXT_STEPS.md` item 2/3) is
running for real.

## 7. Two concrete feature gaps (checked against our own `features.py`, not just their numbers)

Cross-checked their §5/§7 recommendations against what's actually implemented. Most already exist
(`is_s3`, legal-form agreement, name IDF, numeric Jaccard, address-empty flags) — two real gaps found:

1. **Near-miss number flag** — not implemented. Their dominant distractor pattern (`hard_negatives.csv`)
   is small numeric edit distance ("511 Yacht Drive" vs "513 Yacht Drive"), which current features don't
   distinguish from "numbers agree" (Jaccard catches exact overlap) or "no numbers" (missing flag).
2. **Address-missing → 0, not NaN**, in similarity features when one side has no address. The
   `f_addr_empty_{a,b}` flags exist so the model can partly compensate, but worth testing whether NaN
   (LightGBM handles it natively) changes anything, since right now a missing address scores identically
   to a maximally different one on the raw similarity columns.

**Action:** written up as an optional E04b step in `docs/PROMPTS.md` Prompt 5 — add both, report whether
either moves f05_all before committing to them.

---

## Explicitly not adopting

- Any of their deterministic blocking strategies as a replacement for our sparse IDF union — already
  shown to be far less efficient at comparable recall (item 1).
- Citing PILOT-tagged numbers (99.7% recall @ K=25, exclusivity +0.005) or UNVERIFIED hard-coded claims
  (Indic script share, URL substitution rate, channel-agreement 3.1 vs 1.1) in the methodology doc —
  see `docs/EDA_TEAMMATE.md` §0/§13 evidence audit. Re-measure on full data first if we want to use them.
