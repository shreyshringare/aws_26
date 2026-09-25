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

**Teammate's independent EDA confirms the architecture, not a new blocking method:** their plain
deterministic union needs 11,231 cands/entity for 96.31% recall vs our 42.5 — don't adopt their blocking
strategy, just their evidence that the ceiling is reachable.

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

Named generic/franchise collisions: "physical therapy" (1,042 candidate records), "family center" (459),
"meridian" (409), "eye group" (378), "pediatric group" (354).

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

Why this matters for the 0.98 push specifically: France is 15% of test S1, absent from train, and 45% of
its name tokens are unseen (vs 28–36% in the US↔IN transfer proxy) — the proxy is optimistic for France.
Char-level / alias-aware similarity is the mitigation.

---

## 7. Open disagreement — do not silently resolve

Teammate recommends NFKC + casefold + punctuation-strip **only** for blocking identity keys (treating
abbreviation expansion as a scoring feature, not a blocking key), because abbreviation normalization grew
their collision groups 226→429 on a naive exact-match blocking test. Our `blocking.py` uses
abbreviation-normalized tokens directly as blocking key families, but mitigates collision-inflation with
IDF weighting + `BLOCK_MAX_DF=2000` drop — a different fix for the same problem, not obviously wrong, but
unverified at our recall level.

**Action:** ask the teammate exactly which blocking method they measured the 226→429 growth on before
touching `normalize.py` or `config.BLOCK_W`. If time allows after item 1: one experiment removing
abbreviation-normalized families from the union, compare recall/cands-per-entity.

---

## Ranked order for a friend picking this up cold

1. Item 1 (blocking recall → 0.97) — nothing else matters if true matches never reach scoring.
2. Item 2 (turn stage 2 on, real E04 run) — biggest single score lever currently unused.
3. Item 3 (re-tune decoder against real stage-2 scores).
4. Item 5 (quick generic-vocab check — cheap, read-only, do any time).
5. Item 6 (normalization v2 — France-specific, meaningful for the test set's hardest 15%).
6. Item 4 (hard-negative ingestion — blocked on teammate pushing files).
7. Item 7 (blocking-key architecture discussion — sync first, don't change unilaterally).

Every item: one experiment, one `experiments/log.csv` row, commit before the next change. Only spend a
leaderboard upload on a meaningful local OOF gain that also doesn't hurt US↔IN transfer.
