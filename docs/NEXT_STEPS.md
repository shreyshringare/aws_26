# Next Steps — combined plan (current: 0.942 local OOF → target 0.98)

Merges `HANDOFF.md` current state + `docs/IMPROVEMENT_PLAN.md` teammate-EDA findings into one ranked
action list. One change per experiment, log every run in `experiments/log.csv` per `CLAUDE.md`.

**Reality check on the target:** 0.98 macro F0.5 is very high for noisy fuzzy-matched business records —
treat it as "push as far as possible," not a guaranteed number. The honest ceiling depends on blocking
recall (you cannot score what you never shortlist) and on how separable the hard-negative collision
groups actually are. Items 1–3 below are the real path from 0.942 upward; items 4+ are refinements.

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
   distinguish from "numbers agree" or "no numbers present."
2. **Address-missing scores 0, not NaN**, in similarity features when one side has no address. The
   `f_addr_empty_{a,b}` flags exist so the model can partly compensate, but worth testing whether NaN
   (LightGBM handles it natively) changes anything.

**Action:** optional E04b step alongside item 2/3 (`docs/PROMPTS.md` Prompt 5) — add both, report
whether either moves f05_all before committing.

---

## Ranked order for a friend picking this up cold

1. Item 1 (blocking recall → 0.97) — nothing else matters if true matches never reach scoring.
2. Item 2 (turn stage 2 on, real E04 run) — biggest single score lever currently unused.
3. Item 3 (re-tune decoder against real stage-2 scores) — fold in item 8's singleton constraint here.
4. Item 5 (quick generic-vocab check — cheap, read-only, do any time).
5. Item 9 (near-miss number + NaN-address feature gaps — cheap, testable alongside item 2/3).
6. Item 6 (normalization v2 — France-specific, meaningful for the test set's hardest 15%).
7. Item 7 (normalization-as-identity-key experiment — now resolved with methodology, run it during
   item 6's Prompt 8 pass).
8. Item 4 (hard-negative ingestion — blocked on teammate pushing files).

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
