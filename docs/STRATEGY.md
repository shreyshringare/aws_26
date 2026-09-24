# Solution Strategy: Business Entity Resolution (Amazon ML Challenge 2026)

Status as of 2026-09-25 (Day 1). Companion docs: design spec
`docs/superpowers/specs/2026-09-25-entity-resolution-design.md`, task plan
`docs/superpowers/plans/2026-09-25-entity-resolution.md`, rules `CLAUDE.md`, shift state `HANDOFF.md`.

---

## 1. The problem in one page

- **Input:** three TSV sources of business records (`entity_id, business_name, business_address, country`).
  - S1 is a deduplicated reference list.
  - S2 and S3 are noisy copies of the same businesses, plus distractors.
- **Output:** for every S1 entity, the list of S2/S3 ids that are the same business (zero, one or many).
  - `matching_results.tsv` is scored.
  - `candidate_pairs.tsv` is audited for blocking quality.
- **Metric:** macro F0.5 per S1 entity. Per entity, F0.5 = 1.25·TP / (k + 0.25·G), where k = predicted count and G = true count.
  - Precision counts 2× recall.
  - A singleton (no true match) scores 1 only if we predict empty.
- **Hard constraints:**
  - No external data or lookups.
  - Model MIT/Apache and ≤ 8B parameters.
  - France appears only in test.
  - 5 uploads per day.
  - Zip package: code + README + requirements + methodology doc.

## 2. What the data told us (EDA on train)

| Finding | Number | Why it matters |
|---|---|---|
| Scale | train S1 2.2M, S2 5.0M, S3 5.3M; test S1 1.73M, S2 4.9M, S3 5.1M | all-pairs comparison is impossible → blocking; memory-aware code; SageMaker for full runs |
| Singletons | 5.6% of S1 (same in US and India) | most entities *do* have matches → recall matters, not only precision |
| Matches per entity | mean ≈ 3.5, up to 11; typically 1–3 in S2 and 1–3 in S3 | we must output *sets*, not a single best match |
| One-to-one | 0 of 7.6M S2/S3 ids belong to two S1 entities | each S2/S3 record can be assigned to at most one S1 (exact constraint) |
| Cross-country matches | 0 | safe to compare only records sharing a country label |
| Distractors | ~26% of S2/S3 records match nothing | the model must be able to say "no" |
| Native script | 11–15% of S2/S3 names in Devanagari, Malayalam, Kannada… | naive transliteration fails ("मॉडर्न" → "monddrn"), so we learn a token table from train pairs |
| France | 259k test S1 (15%), unseen in train | features must be country-agnostic; French street/legal words are in our dictionaries |
| Noise operators seen | house number digit drop (3424→424), legal suffix moved to front, generic words added/dropped ("Services", "Care"), accents, TX↔Texas, typos ("Vlley"), empty address, completely different trade name at the same address | these become explicit "noise-channel" features |

## 3. Solution architecture

```
raw TSV
  → 1. normalize        (dictionaries + learned transliteration)
  → 2. block            (per-country IDF key matching, top-40 candidates per S1)
  → 3. stage-1 features (string similarity + noise-channel features)
  → 4. LightGBM stage 1 (5-fold GroupKFold, out-of-fold scores)
  → 5. stage-2 features (group context, competition, triangle support)
  → 6. LightGBM stage 2
  → 7. decode           (calibrate → global one-to-one → expected-F0.5 subset per entity)
  → output TSVs + official validator
```
Every stage caches parquet, so a later shift resumes without recomputing.

### 3.1 Normalization (`normalize.py`, `translit.py`) — implemented ✅
- **Cleaning:** Unicode NFKC, lowercase, dots of single-letter abbreviations removed (`L.L.C.`→`llc`, `R.`→`r`), `&`→`and`, punctuation removed, junk tokens (`null`) dropped.
- **Names:** split into `name_core` (meaningful words) and `legal` (canonical suffixes: inc, llc, corp, ltd, pvt, sarl, sas, eurl…), plus a flag for whether the suffix was at the front.
- **Addresses:**
  - Short canonical forms on both sides (street/st, road/rd, avenue/av, rue/r, allée/all…).
  - US state names mapped to codes.
  - "Near SBI ATM"-style landmarks split into their own field.
  - Numbers, postcode (5–6 digits) and a street key (house number + street word) extracted.
- **Learned transliteration:**
  - From ground-truth pairs whose pool name is in native script, align tokens by position and keep the majority Latin spelling ("प्राइवेट" → "private").
  - Learned only from the provided training data.
- **Status:** 15 unit tests pass. Throughput is ~49k rows/s per core. Real-data smoke check: 16% of rows contain native script, 0% end up with an empty core name.

### 3.2 Blocking (`blocking.py`) — implemented, first real run in progress ⏳
- **Keys per record:**
  - name words, adjacent name-word pairs, sorted name initials
  - address words, house-number+street key, postcode
- **Scoring:**
  - Each key is weighted by rarity (IDF). Very common keys (> 2000 records) are ignored.
  - Candidate score = sum of shared key weights, computed as a sparse matrix product per country label, in parallel chunks. The top 40 per S1 are kept.
- **Why:**
  - Rare shared tokens are the strongest cheap evidence.
  - Address keys still catch trade-name swaps, where the names share nothing.
  - Scales to 10M records.
- **Target:** ≥ 97% of true pairs retained, with about 40 candidates per entity. The first dev measurement (E01) is running now.
- `candidate_pairs.tsv` is exactly this candidate set, the same one the model scores.

### 3.3 Features (`features.py`, `noise.py`) — planned (Phase 3)
- **Similarity:**
  - rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler on the full name, core name and address.
  - Jaccard overlaps of name, address and number tokens.
  - First house number equal, postcode equal/missing, legal-suffix agreement.
- **Record statistics:** blocking score and rank, source (S2/S3), how common the name is, native-script flags, empty address/landmark flags.
- **Noise-channel features (differentiator #2).** These say *why* two records differ:
  - `digit_deletion`: 3424 vs 424.
  - `generic_only`: the names differ only by learned generic words like "services".
  - `legal_moved`, `edit1_tokens` (typos).
  - `addr_only` (trade-name swap).
  - `accent_only`, `addr_norm_only` (abbreviation or state expansion only).

  A generic similarity score can't separate "different business" from "same business with a typical perturbation"; these features can.
- **Country is never a feature.** That's how the model transfers to France.

### 3.4 Models (`model.py`, `context.py`) — planned (Phases 3–4)
- **Stage 1:** LightGBM binary classifier. 5-fold GroupKFold grouped by S1 entity, so an entity's candidates never sit in both train and validation. Produces out-of-fold (OOF) probabilities.
- **Stage 2 (differentiator #1a)**, built only from OOF stage-1 scores:
  - **Group context:** rank in its S1 group, gap to the group's best, number of confident candidates.
  - **Competition:** how strongly *another* S1 claims the same record. This is the one-to-one signal.
  - **Triangle support:** true S2 and S3 matches of one business also resemble *each other*. A candidate that looks like the group's other high-scoring candidates gets support; a lone look-alike does not.
- Test inference uses models refit on all training rows.

### 3.5 Decoding (`decide.py`) — planned (Phase 3) (differentiator #1b)
1. **Isotonic calibration** of the OOF probabilities.
2. **Global one-to-one:** each S2/S3 record stays only with the S1 entity that scores it highest. This is exact per the EDA.
3. **Expected-F0.5 subset selection:**
   - For each entity, choose the number of top candidates k (including k = 0) that maximizes the *expected* F0.5 under the probabilities.
   - Estimated by Monte Carlo, and it accounts for matches blocking missed.
   - Singletons are handled by the math, not by a hand-set cut-off.
4. **Baseline for comparison:** a tuned global threshold + empty gate. Both are grid-searched on OOF with the exact metric, and the better one is used.

### 3.6 Validation
- **Dev loop:**
  - Stratified 10% of train S1, searched against the *full* train pool (realistic distractors).
  - 5-fold OOF.
  - Prints overall / per country / singletons / matched F0.5, blocking recall and candidates per entity.
- **France proxy:** train on US → score India, and India → US. A change that raises overall F0.5 but lowers transfer is rejected.
- **Experiment log:** every experiment is one row in `experiments/log.csv`. Every upload is git-tagged.

## 4. Compliance audit: is this in line with the guidelines, problem statement and dataset?

| Requirement (source) | Our handling | Status |
|---|---|---|
| Tab-separated I/O, exact headers, one row per test S1, test order (PS) | `io_utils.write_lists`; round-trip test passes the official validator | ✅ tested |
| Matches ⊆ candidates; candidates = last stage before the model (PS) | the model scores exactly the blocking output, and that same set is written out | ✅ by construction |
| Every test S1 present, incl. France (PS) | writer iterates over all test S1 ids; E00 file (1,732,544 rows) passes the validator | ✅ verified |
| Country is an open set; no hard-coding/one-hot of {US, India} (PS) | blocking loops over whatever labels exist; country is never a feature | ✅ |
| No external data / APIs / geocoding (PS) | only `dataset/` files. Dictionaries are hand-written. Transliteration table and generic-word list are learned from train. `unidecode` is a generic character-transliteration library, not a business or address database | ✅ (dictionaries and the library get declared in the methodology doc) |
| Model MIT/Apache, ≤ 8B params (PS) | LightGBM (MIT), a few MB | ✅ |
| Validate locally on a held-out split with F0.5 (PS) | exact metric (PS example 0.714 reproduced in tests), GroupKFold OOF | ✅ |
| Zip structure: `output/`, `code/business_entity_resolution/{src,README,requirements}`, doc (PS) | `scripts/make_zip.py` (Phase 5); README + pinned requirements exist | ⏳ planned |
| Methodology doc: methodology, blocking, model + features, other (PS + guidelines) | `Documentation_template.md` filled in Phase 5: core ≤ 2 pages (guidelines), details in the Appendix (PS allows unlimited) | ⏳ planned |
| Source code with comments describing functions (guidelines) | docstring on every function (enforced in the plan) | ✅ so far |
| ≤ 5 uploads/day, keep version history (guidelines) | only the shift driver uploads; git tag per upload; log.csv | ✅ process |
| One login per participant, desktop only (guidelines) | human process: one portal login per person | ℹ️ team to follow |
| Dataset fit | every design choice traces to an EDA number in §2 | ✅ |

**Honest caveats (not rule violations):**
- **Validation optimism:** the transliteration table and generic-word list are fitted on all train labels before the OOF split, so OOF scores are slightly optimistic. It's small because they're coarse dictionaries, but it's declared in the doc.
- **Blocking uses test-pool statistics:** key rarity (IDF) on test is computed from the *unlabeled* test pool. That's standard and uses no labels or external data.
- **France is never directly measured:** the US↔India transfer proxy and one diagnostic upload are the only checks.
- **Postcodes:** US addresses in this data carry no ZIP, so the postcode key mostly helps India/France.

## 5. Differentiators (why this should beat a standard entry)
1. **Collective, metric-aware decoding:** triangle support + global one-to-one + expected-F0.5 subset selection. Most teams score each pair independently and apply one global threshold.
2. **Noise-channel features:** they model the generator's perturbation operators explicitly.
3. **Learned transliteration** for native-script names, from the training labels only.

## 6. Progress

| Phase | Content | Status |
|---|---|---|
| 0 Foundation | package, config, metric + writer tests, E00 empty submission | ✅ done (E00 ready to upload) |
| 1 Normalization | dictionaries, learned transliteration, real-data smoke | ✅ done (15 tests) |
| 2 Blocking | IDF sparse blocking, pipeline skeleton, dev recall run | ⏳ code done, first dev run (E01) running |
| 3 Stage 1 | noise + pair features, LightGBM OOF, decoders, first real upload | next |
| 4 Differentiators | stage-2 context/triangle, decoder comparison, iteration | Day 2 |
| 5 Packaging | zip, methodology doc, final selection | Day 3 |

## 7. Next steps (in order)
1. **Upload E00** (all-empty, expected LB ≈ 0.056) to confirm the portal path. Record it in `HANDOFF.md`.
2. **Blocking recall to ≥ 0.97** (Task 9): read the E01 misses, then adjust one lever per run (top-K, key frequency cap, address weights).
3. **Stage 1** (Tasks 10–14): noise features → pair features → LightGBM OOF → decoders → dev F0.5 + transfer (E02).
4. **First real upload** (Task 15): full run on SageMaker `ml.r5.4xlarge`, validator PASS, upload, tag `sub-d1-02`.
5. **Stage 2 + expected-F0.5** (Tasks 16–17): keep only if dev F0.5 rises and transfer does not drop (E04).
6. **Day-2 iteration** (Task 18): error analysis of false merges and misses, dictionary fixes, top-K/model tuning. Upload only on a gain ≥ 0.002, or to test France behaviour.
7. **Day 3:** feature freeze 14:00 IST. Final full run, final pick by OOF + transfer (not public LB), zip + methodology doc, final validator, submit before 23:59 IST.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Laptop RAM (16 GB) insufficient for full runs | dev loop on 10% sample locally; full runs on SageMaker 128 GB |
| Blocking recall too low (it caps the score) | measured every run; tune one lever at a time with miss diagnostics |
| France behaves differently | country-agnostic features, FR dictionary entries, transfer proxy, one diagnostic upload |
| Overfitting the public LB | final choice by local OOF + transfer, not public rank |
| Running out of time/credits | shift relay with `HANDOFF.md`; each member spends their own credits; long jobs start at shift ends |
