# EDA Learnings — Amazon ML Challenge 2026 (Business Entity Resolution)

Source: team uploads `eda/` (48 files: tables, benchmarks, plots, samples) and `analysis/` (the Python
scripts that produced them), dated 25 Sep 2026. This document consolidates every finding, rates how much
each can be trusted, and turns them into pipeline decisions.

---

## 0. How to read this document: evidence levels

The EDA mixes three kinds of numbers. They are not equally reliable, and several headline figures are
not reproducible from the uploaded code.

| Tag | Meaning | How to treat it |
|---|---|---|
| EXACT | Computed over the full training (or test) files by the scripts | Trust |
| SAMPLE | Computed on a reproducible sample (e.g. 246,867 positive pairs, 35,000 records, 900 hard negatives) | Trust direction and rough size |
| PILOT | Computed on 500 S1 queries scored against only ~4,000 random background distractors | Optimistic; real numbers will be worse. Re-measure on full data |
| UNVERIFIED | Written as hard-coded strings in `run_source_and_negative_analysis.py`, `run_retrieval_benchmark.py` or `generate_meta_matrices.py`; no computing code is in the package | Plausible, but do not quote in the methodology doc until recomputed |

Section 13 lists exactly which claims fall into each bucket.

---

## 1. Top 12 learnings (TL;DR)

1. **Scale is large.** Train has 2.21M S1 records and 10.32M S2+S3 candidate records. Test has 1.73M S1
   and 9.97M S2+S3. The full train cross product is ~22.8 trillion pairs. Pandas plus brute-force kNN
   will not work naively; the pipeline must be chunked, per-country and disk-backed. (EXACT)
2. **Only 5.58% of S1 entities are singletons.** 89% have two or more matches (mean 3.46, max 11). An
   all-empty submission scores about 0.056. The competition is mostly about list completeness plus
   precision within multi-match lists, not singleton detection. This revises the earlier assumption that
   the empty/non-empty decision dominates. (EXACT)
3. **Candidate exclusivity holds perfectly in train.** Zero S2/S3 records link to more than one S1 across
   7,638,365 positive pairs. One-to-one (argmax) assignment is safe to use. (EXACT)
4. **Matches never cross countries.** Zero cross-country positives, so country is a safe blocking
   partition. It must stay an open set because of France. (EXACT)
5. **About 26% of S2 and S3 records are unmatched distractors.** That is 1.341M in each, within 140
   records of each other, which strongly suggests synthetic injection. (EXACT counts; the "synthetic"
   cause is a hypothesis)
6. **Name similarity alone fails.** Hard negatives have name similarity (mean 0.87) as high as or higher
   than true positives. Address tokens and numbers are what separate them. (SAMPLE)
7. **Distractors are near-duplicates with perturbed numbers** (e.g. "511 Yacht Drive" vs "513 Yacht
   Drive", "Sn 2101" vs "Sn 2108") and swapped legal forms or inserted words. Numeric agreement is the
   strongest precision signal. (SAMPLE; examples read directly from `hard_negatives.csv`)
8. **S2 and S3 are corrupted differently.** S2 has cleaner addresses (12.35% exact address match vs
   4.41%). S3 has cleaner names (median name Jaro-Winkler 0.922 vs 0.883). Use a source feature and
   per-source thresholds. (SAMPLE)
9. **Test differs from train in country mix.** Test S1 is France 15.0%, India 46.8%, US 38.3%; train is
   US 60%, India 40%. France is 259,452 test S1 entities, all unseen in training. (EXACT)
10. **Deterministic blocking cannot reach high recall cheaply.** The best exhaustive union reaches 96.3%
    recall only at ~11,000 candidates per S1. Bounded top-K character n-gram retrieval is required.
    (EXACT)
11. **Top-25 per source looked like ~99.7% recall in the pilot**, but that was measured against only
    4,000 random distractors. Real recall at K=25 against millions of same-country records is unknown
    and will be lower. (PILOT)
12. **Native Indic scripts** (Devanagari and others) appear in S2/S3 while S1 is Latin. Transliteration
    to Latin is needed for India matching. The effect-size numbers are UNVERIFIED, but script presence is
    confirmed in the token statistics.

---

## 2. Dataset scale (EXACT, `01_dataset_overview.csv`)

| Split / source | Rows | Unique names | Unique addresses | Countries |
|---|---|---|---|---|
| train S1 | 2,206,821 | 1,539,229 | 2,130,606 | 2 |
| train S2 | 5,034,616 | 4,402,009 | 4,337,262 | 2 |
| train S3 | 5,285,603 | 4,651,609 | 4,632,765 | 2 |
| test S1 | 1,732,544 | 1,238,867 | 1,677,483 | 3 |
| test S2 | 4,887,273 | 4,311,041 | 4,224,784 | 3 |
| test S3 | 5,082,316 | 4,521,929 | 4,456,436 | 3 |

The files are about 1.2 GB for S2/S3 and about 200 MB for S1. There are no duplicate IDs, no malformed
rows and no exact duplicate rows.

**Implications**
- 30% of S1 names are not unique (2.2M rows, only 1.54M distinct names). Many different businesses share
  a name, so the name can never be a key.
- The EDA environment had no pandas or scikit-learn, and the team used SQLite streaming. For our
  pipeline, plan memory explicitly. Use polars or pandas with `dtype=str`, process one country at a time,
  and use chunked sparse matrix products for top-K retrieval. Cache intermediate results as parquet.
- At K=25 per source, test inference is about 86.6M candidate pairs (1.73M × 50). Feature extraction must
  be vectorized (rapidfuzz `cdist`/`cpdist`, numpy), not per-row Python.

---

## 3. Ground-truth structure (EXACT, `04_ground_truth_stats.csv`)

| Matches per S1 | Entities | % | Cumulative % |
|---|---|---|---|
| 0 (singleton) | 123,247 | 5.58 | 5.58 |
| 1 | 119,157 | 5.40 | 10.98 |
| 2 | 375,212 | 17.00 | 27.99 |
| 3 | 530,841 | 24.05 | 52.04 |
| 4 | 484,115 | 21.94 | 73.98 |
| 5 | 321,957 | 14.59 | 88.57 |
| 6 | 164,868 | 7.47 | 96.04 |
| 7–11 | 87,424 | 3.96 | 100 |

There are 7,638,365 positive pairs: 48.4% S1–S2 and 51.6% S1–S3. 80.5% of S1 entities match records in
both S2 and S3.

The singleton rate is the same across countries (US 5.58%, India 5.59%).

Metric implications under macro F0.5, for a typical entity with 4 true matches:

| Prediction | P | R | F0.5 |
|---|---|---|---|
| 4 correct | 1.00 | 1.00 | 1.000 |
| 3 correct, 0 wrong | 1.00 | 0.75 | 0.938 |
| 2 correct, 0 wrong | 1.00 | 0.50 | 0.833 |
| 4 correct + 1 wrong | 0.80 | 1.00 | 0.833 |
| 3 correct + 1 wrong | 0.75 | 0.75 | 0.750 |
| empty | – | 0 | 0.000 |

Missing one match (0.938) is much cheaper than adding one wrong one (0.833). Keep per-candidate
thresholds conservative.

Predicting empty for a matched entity costs a full point, and 94% of entities have matches. For a matched
entity, the top-1 candidate is usually worth including. The singleton gate must therefore be narrow and
precise, not a broad "abstain when unsure".

The empty baseline is about 0.056 on the leaderboard. Any real model should be far above that.

---

## 4. Structural properties you can exploit

### 4.1 Candidate exclusivity (EXACT, EXP-01)

0 of 7,638,365 links put an S2/S3 record in two S1 clusters. There are 3,693,619 unique matched S2
records and 3,944,746 unique matched S3 records.

The problem statement does not guarantee this for test. The team treats it, correctly, as an empirical
regularizer rather than an axiom.

**Decision:** after scoring, assign each S2/S3 candidate only to the S1 with its highest score
(competitive argmax). Also add "gap to this candidate's best rival S1" as a model feature.

### 4.2 Country partition (EXACT, EXP-02)

0 cross-country positives. Blocking within `country` removes all cross-country comparisons with no
recall loss.

**Decision:** group by whatever country labels exist (open set). Do not use country as a model feature,
since France is unseen. Build one generalized model, not per-country models; the team explicitly rejected
per-country models (evidence matrix).

### 4.3 Unmatched distractor pool (EXACT counts)

Train S2 has 1,340,997 unmatched records (26.64%) and S3 has 1,340,857 (25.37%). The two counts differ by
only 140.

These distractors share regional vocabulary with the true entities (see §7), so they are hard negatives
by design.

The "synthetic distractor injection" explanation is a hypothesis. Don't build rules that reverse-engineer
a generator; use measured address and numeric contrast instead.

---

## 5. Data quality (EXACT, `02_data_quality.csv`, `table_data_quality_census.csv`)

| Issue | Train S2 | Train S3 | Test S2 | Test S3 |
|---|---|---|---|---|
| Empty address | 3.36% | 3.33% | 2.65% | 2.68% |
| Very short name (≤ 2 chars) | 0.014% | 0.175% | 0.208% | 0.392% |

S1 is complete: no missing names or addresses in train or test.

Missing addresses occur only in S2/S3. The team estimates they affect about 4.2% of ground-truth matches
(UNVERIFIED). Address-only blocking channels will miss those pairs.

Very short names are a distribution shift. In S2 they are about 15× more frequent in test than in train
(0.014% → 0.208%), and in S3 about 2× (0.175% → 0.392%). These are likely initials or abbreviations.
France may be contributing. Check them on test before trusting name features for those rows.

**Decisions**
- Add `addr_missing` and `name_len` / `name_very_short` features.
- For a missing address, don't treat it as disagreement: set address features to NaN, not 0, so LightGBM
  can learn a separate path.
- Keep a name-driven retrieval channel so records with no address can still be found.

---

## 6. Source asymmetry: S2 vs S3 (SAMPLE, `positive_similarity_summary.csv`, 246,867 positive pairs)

| Metric on true matches | S1–S2 | S1–S3 | Reading |
|---|---|---|---|
| Exact normalized name | 21.4% | 22.2% | similar |
| Name Jaro-Winkler (median) | 0.883 | 0.922 | S3 names cleaner |
| Name Levenshtein ratio (mean) | 0.568 | 0.676 | S3 names cleaner |
| Exact address | 12.35% | 4.41% | S2 addresses much cleaner |
| Address token Jaccard (median) | 0.696 | 0.500 | S2 cleaner |
| Address char-3gram cosine (median) | 0.830 | 0.737 | S2 cleaner |
| Numeric-token overlap (mean / median) | 0.775 / 1.0 | 0.787 / 1.0 | same |
| Address length difference (mean chars) | 8.5 | 12.5 | S3 reordered and padded |

**Decisions**
- Add an `is_s3` feature so the model can weight name vs address differently by source.
- Tune the match threshold per source (S2 and S3 separately) on out-of-fold predictions.
- Prefer order-invariant address similarities (token-set ratio, char n-gram cosine) over plain
  Levenshtein. S3 addresses are heavily reordered, with the state or city moved to the front.

---

## 7. Hard negatives: what makes this problem hard (SAMPLE)

Files: `hard_negatives.csv` (900 pairs), `easy_negatives.csv` (600), `negative_similarity_summary.csv`.

| Signal | True positives | Hard negatives | Easy negatives |
|---|---|---|---|
| Name similarity (mean) | 0.76 (S2) / 0.84 (S3) JW | 0.87 | 0.24 |
| Address similarity (mean) | ~0.6–0.8 | 0.34 | 0.08 |
| Numeric-token overlap* | mean ~0.78, median 1.0 | mean 0.06, median 0.0 | ~0 |
| Exact same name | 21–22% | 4.2% | – |

*The hard-negative figure was recomputed from `hard_negatives.csv` for this document (overlap / min set
size, over pairs where both sides have numbers). The positive figure comes from the positive sample
summary. The team's hard-coded table says 0.892 vs 0.215; the direction is the same, but use the
recomputed numbers.

**What the distractors look like** (verbatim patterns from the file):
- Perturbed house numbers: "511 Yacht Drive, Grand Rivers, KY" vs "513 Yacht Drive, Grand Rivers,
  Kentucky", and "Sn 2101" vs "Sn 2108". These have the same street and city but a different number.
- Legal-form or word swaps: "Ramsey, Zhou & England LLC" vs "Ramsey, Zhou & England Partners", and "Indo
  Properties Private Limited" vs "Indo Properties Hardware Private".
- Same name, different place: generic names ("Physical Therapy", "Family Center", "Eye Center", "Vision
  Group") in different cities.
- Near-name, different business: "Candice's Restaurant" vs "Patrice's Restaurant", and "Harvin Seva
  Samiti" vs "Marvel Seva (Samiti)".

**Decisions and feature ideas**
- Numbers are the strongest precision signal. Add these features:
  - numeric-token Jaccard;
  - an exact house-number match flag;
  - a numeric conflict flag (both sides have numbers and none agree);
  - a near-miss number flag (numbers differ by a small amount, the typical distractor pattern);
  - postcode/PIN equal, conflicting, or missing.
- A missing number is not a conflict. Keep the missing and conflict flags separate.
- Add a name-frequency (IDF) feature, so that common names need address evidence while rare names can
  carry more weight.
- Add a legal-form agreement feature (LLC vs Partners vs Pvt Ltd), but not as a hard rule, since true
  matches also vary legal suffixes.
- Mine training negatives the way these hard negatives were mined (from the same blocks and retrieval),
  not at random. Random negatives are far too easy (easy-negative name similarity is 0.24).

---

## 8. Singletons (EXACT, `singleton_candidate_load_all.csv.gz`, all 123,247 train singletons)

| Statistic (upper bound, before dedup, deterministic blocks) | Value |
|---|---|
| Mean candidates per singleton | 198.6 (median 123, max 8,667) |
| Singletons with an exact same-name candidate in the same country | 37.8% |
| Singletons with an exact same-address candidate | 3.0% |
| Singletons with zero block candidates | 0.9% |

`hard_singletons.csv` (a sample of 16) shows singletons whose best deceptive candidate has name
similarity 0.83–0.99 and combined similarity 0.64–0.87.

**Implications**
- An exact name match is not evidence of identity: more than a third of singletons have one. Address and
  numeric agreement must confirm it.
- With only 5.6% singletons, the empty gate should fire only when the best candidate lacks
  address/number support. Tune it on out-of-fold predictions, and check that the singleton slice and the
  matched slice both improve.

---

## 9. Normalization (SAMPLE, 35,000 records; `03_normalization_comparison.csv`,
`table_collision_risk_study.csv`)

| Variant | Name collision groups | Collision rate | Recommended role |
|---|---|---|---|
| raw | 226 | 0.65% | baseline |
| conservative (NFKC + casefold + punctuation + whitespace) | 389 | 1.13% | blocking keys |
| Indic transliterated + conservative | 396 | 1.15% | blocking (India) |
| URL/domain cleaned + conservative | 397 | 1.15% | blocking |
| aggressive alphanumeric | 411 | 1.19% | char n-grams only |
| abbreviation expanded | 429 | 1.24% | features only |
| token sorted | 436 | 1.26% | features only |

Name + address + country together are 99.99% unique in the sample, so the full record is highly
identifying.

Collisions grow with each more aggressive normalization. Expansion and token sorting erase real
differences (for example Corp vs LLC). Use conservative normalization for keys and aggressive views only
as features.

`normalization_collision_examples.csv` and `false_collision_examples.csv` show the worst common-name
blocks: "physical therapy" has 1,042 records in one block, "family center" 459, "meridian" 409, "eye
group" 378.

**Normalization steps the data calls for:**
- Unicode: NFKC plus casefold, and NFKD accent folding for France (é, è, ç, ê and others). The test data
  has 83k é and 29k è.
- Indic scripts: transliterate Brahmi scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
  Telugu, Kannada, Malayalam) to Latin. The team's offset-table transliterator is in
  `run_exclusivity_ablation.py` and `run_retrieval_benchmark.py`. It is hand-written, not an external
  lookup, so it is fine to use.
- URLs in names: strip protocol, `www.` and TLDs. "com" is among the most frequent name tokens (~4,250
  document frequency in the token statistics), which corroborates URL substitution.
- Legal suffixes: split them into their own field. Top suffixes by document frequency are limited, llc,
  private limited, ltd, inc, pvt ltd, corp, co, and for France sarl, sas, eurl, sa, sasu, sci and ei.
- Addresses:
  - strip `C/O <name>` prefixes;
  - normalize zero-padded numbers ("00501 12th Street" → "501", "0013535" → "13535");
  - canonicalize state names ("MH" ↔ "Maharashtra", "Keralam" ↔ "Kerala", "KY" ↔ "Kentucky");
  - separate "Unit / STE / # / Apt" tokens;
  - handle glued tokens ("Bismarckcity", "Koramangalaindustrial").

---

## 10. Blocking benchmarks (EXACT full train; candidate volumes estimated from a 300-S1 sample)

### 10.1 Individual channels (`05_blocking_benchmark.csv`)

| Channel (within country) | Recall | Avg candidates / S1 | Verdict |
|---|---|---|---|
| Exact normalized name | 21.8% | 9.8 | cheap anchor |
| Exact normalized address | 8.3% | 0.35 | ultra-precise anchor (S2 12.5%, S3 4.3%) |
| Postal/PIN code | 4.8% | 3.2 | cheap, small gain |
| Rarest address token | 72.0% | 189 | best recall per candidate |
| Rarest name token | 64.2% | 2,196 | too heavy uncapped |
| Rare name trigram | 57.0% | 1,114 | heavy |
| Alphanumeric prefix-8 | 65.2% | 2,347 | heavy |
| First-name-token prefix-4 | 77.1% | 9,444 | reject: 20.8 billion pairs |

### 10.2 Cumulative unions

- Exact name + exact address + postal: 31.9% recall at about 11 candidates per S1 (median 2). This is a
  high-precision seed set.
- Adding the rare address token: 78.7% at 231 per S1.
- Adding the rare name token: 90.5% at 2,075 per S1.
- Everything including prefixes: 96.3% at 11,231 per S1 (p95 33k, max 71k). Infeasible for 1.7M test S1
  entities.

### 10.3 Takeaways

- Pure key-based blocking hits a wall around 96% recall with an unmanageable number of candidates.
- Use bounded, ranked retrieval instead: character n-gram TF-IDF top-K on name and on address, per
  country and per source, unioned with the cheap exact channels. Rank inside large blocks rather than
  taking whole blocks.
- The number of channels that retrieved a pair is itself a strong feature. The team claims true pairs are
  hit by about 3.1 channels vs about 1.1 for hard negatives (UNVERIFIED), which is cheap to recompute:
  keep channel flags for every candidate.

---

## 11. Top-K sparse retrieval pilot (PILOT, `table_retrieval_k_sweep.csv`)

Setup: 500 sampled S1 queries, their true matches, plus only ~4,000 random background distractors.
Scoring was 0.55 × name char-3gram cosine (transliterated) + 0.35 × address char-3gram cosine + 0.10 ×
numeric overlap.

| K per source | Candidates / S1 | Micro pair recall | Macro all-match recall | Exact-set recovery |
|---|---|---|---|---|
| 1 | 2 | 49.3% | 62.9% | 26.0% |
| 5 | 10 | 98.6% | 98.9% | 96.2% |
| 10 | 20 | 99.3% | 99.6% | 98.2% |
| 25 | 50 | 99.7% | 99.8% | 99.0% |
| 50 | 100 | 99.9% | 99.95% | 99.8% |

In the same pilot, Indic matches were recovered 97.5% of the time with transliteration vs 4.2% without it
(at K=25).

**Why this is optimistic:** in the real test, each S1 competes with hundreds of thousands to millions of
same-country records, including the near-duplicate distractors from §7. With only 4,000 random
distractors, true matches easily rank in the top 5. The shape of the curve (fast saturation) is
informative, but absolute recall at K=25 must be re-measured on the full training pool before it goes
into the methodology doc.

**Decisions**
- Start with K = 25–50 per source per retrieval view (name view and address view), union them with the
  exact channels, then measure full-train recall and candidates per S1.
- The candidate file is audited for reduction ratio. At 50–100 candidates per S1 the reduction ratio is
  still above 99.999%, so recall is the priority here.
- Keep the transliterated view in retrieval, not just in features.

---

## 12. Exclusivity ablation (PILOT, `table_exclusivity_ablation.csv`)

Setup: the same 500-query pilot, the same hand-weighted score (no trained model), and a threshold sweep.

| τ | No exclusivity | Hard argmax | Soft margin |
|---|---|---|---|
| 0.50 | 0.9534 | 0.9586 | 0.9591 |
| 0.60 | 0.9238 | 0.9238 | 0.9236 |
| ≥ 0.65 | identical | identical | identical |

Exclusivity helps only at loose thresholds (+0.005 at τ=0.5), which is when two S1 entities both pass the
threshold for the same candidate.

The absolute F0.5 of about 0.95 is not a realistic estimate, because the distractors are too easy.

Scores fall monotonically with τ from 0.5 upward. For this heuristic, looser was better, which is
consistent with §3: recall matters because 94% of entities have matches.

**Decision:** keep argmax exclusivity. It is supported by the exact training property and costs almost
nothing. Re-measure its gain with the real model and real candidates.

---

## 13. Evidence audit: what is computed vs hard-coded

| Claim | Status | Note |
|---|---|---|
| Row counts, country mix, missingness, short names | EXACT | full scan |
| Singleton rate 5.58%, cardinality distribution, 7.64M positives | EXACT | full ground truth |
| 0 multi-S1 candidates; 0 cross-country matches | EXACT | full ground truth |
| 1.34M unmatched S2 and S3 records each | EXACT | set difference |
| Blocking channel recall and candidate counts | EXACT recall; volumes estimated from 300 S1 | `05_blocking_benchmark.csv` |
| Singleton load (198.6 mean, 37.8% exact-name hit) | EXACT | recomputed here from the .csv.gz |
| S2 vs S3 similarity profiles | SAMPLE (246,867 pairs) | hash-sampled |
| Hard vs easy negative profiles | SAMPLE (900 / 600) | EXP-07 text says 5,000 hard negatives; the file has 900 |
| Normalization collisions | SAMPLE (35,000) | |
| France lexicon, suffixes, accents | EXACT | test scan |
| K-sweep recall 99.66% at K=25; Indic 4.2% → 97.5% | PILOT | 4,000 random distractors only |
| Exclusivity +0.0052 | PILOT | heuristic score, not a model |
| Indic script share 7.40% of names (18.4% India), JW 0.439 → 0.842 | UNVERIFIED | hard-coded strings; script presence itself is confirmed in the token statistics |
| URL substitution in 5.11% of pairs (100,602-pair sample), JW 0.901 → 0.924 | UNVERIFIED | hard-coded; the frequent "com" name token supports the pattern |
| Numeric overlap 0.892 (positive) vs 0.215 (hard negative) | UNVERIFIED | recomputed here: ~0.78 vs ~0.06, same conclusion |
| Channel agreement 3.1 vs 1.1 | UNVERIFIED | hard-coded |
| "~41.5% of positives have address permutation / PO box" | UNVERIFIED | hard-coded |
| Indic share by source 4,648 (S2) / 2,798 (S3) | UNVERIFIED | hard-coded |
| Retrieval channel table CH-6/CH-7 (89.45%, 92.15%) | UNVERIFIED | hard-coded descriptive table |
| "S2 = postal registry, S3 = web crawl" | HYPOTHESIS | the team correctly rejects relying on it |

Minor inconsistencies to fix before reusing text: EXP-09 says 1,731 true positives while the script
docstring says 1,772; "5,000 hard negatives" vs 900 in the file.

---

## 14. France: everything known about the unseen country (EXACT, test scan, `table_france_lexicon.csv`)

**Volume:** 259,452 test S1 entities (15.0%), 703,378 in test S2 and 731,615 in test S3. The
candidate-to-S1 ratio is about 5.5, similar to the other countries.

**Legal forms:** SARL 73k, SAS 52k, EURL 17k, SA 12.8k, SASU 10.7k, SCI 8.4k, EI 4.2k.

**Name pattern:** many names are associations or schools with a legal form appended ("club sarl", "ecole
sas", "amicale sarl", "comite sas", "sportive sarl"). This suggests suffixes are added as noise
(hypothesis), so strip them before comparing names.

**Geography** is concentrated in three regions:
- Hauts-de-France: Lille, Roubaix, Tourcoing, Dunkerque, Calais;
- Nouvelle-Aquitaine: Bordeaux, Pessac, La Teste-de-Buch;
- Pays de la Loire: Nantes, Saint-Nazaire.

So there will be many same-city distractors: expect hard negatives similar to Indian metros.

**Address vocabulary:** rue, avenue, de, la, du, des, saint, and the house-number suffix "bis".

**Characters:** é 83k, è 29k, É 13k, plus ’, ç, ê, â, ë, ô.

**Postal codes:** 5 digits (the team's regex is `(0[1-9]|[1-8]\d|9[0-8])\d{3}`). They collide in format
with US ZIP codes, which is fine because matching is within country.

**Decisions**
- Use NFKD accent folding everywhere.
- Add French legal forms to the suffix list; this is domain knowledge, so document it.
- Handle "bis" / "ter" as house-number modifiers.
- Character n-gram features transfer across languages; word-level abbreviation dictionaries do not.
- Validate transfer by training on US and scoring India (and the reverse) as a France proxy.
- Consider stricter thresholds for country labels unseen in training, and test that with a paired
  leaderboard submission (France predicted vs France empty).

---

## 15. Consolidated pipeline decisions

**Normalize into multiple views per record:**
- `conservative` (keys);
- `translit` (Indic → Latin);
- `url_clean`;
- `accent_folded`;
- `legal_suffix` field;
- `numbers` set;
- `postcode`;
- `addr_tokens_sorted` (features only).

**Candidate generation (per country, per source):**
- exact name, exact address and postcode channels;
- top-K char-3gram TF-IDF on name (translit view) and on address;
- union of all channels, keeping channel flags.
- Target at least 99% measured full-train recall at a bounded candidate count.

**Features:**
- name and address similarity suite (Jaro-Winkler, token-set, char-cosine, Jaccard);
- numeric Jaccard, conflict and near-miss flags;
- postcode equal, conflict or missing;
- address missing;
- name IDF / frequency;
- legal-form agreement;
- `is_s3`;
- channel count and flags;
- rank / gap-to-best within the S1 group;
- gap to the candidate's best rival S1.

**Model:** a single LightGBM for all countries, with no country feature. GroupKFold by S1. Negatives
mined from the same retrieval.

**Decision:**
- argmax exclusivity;
- per-source thresholds;
- a narrow empty gate (the best candidate lacks address/number support);
- all tuned on out-of-fold predictions with exact macro F0.5, reported by country, source, singleton and
  cardinality.

**Engineering:** stream and chunk everything, process one country at a time, cache parquet, and vectorize
feature extraction (86M+ test pairs).

---

## 16. Open questions and next experiments

1. **Full-data retrieval recall:** measure recall and candidates per S1 for top-K (K = 10/25/50) over the
   full train pool, by country and source. This replaces the pilot numbers.
2. **Recompute the UNVERIFIED metrics** before citing them: Indic share, URL share, channel agreement,
   address permutation rate.
3. **Short-name shift:** inspect the test S2/S3 names of 2 characters or fewer. What are they, and are
   they French?
4. **Near-miss numbers:** how often do true matches have slightly different numbers (typos) vs
   distractors? This decides how to encode the near-miss flag.
5. **Threshold behaviour with the real model:** the pilot showed loose thresholds win. Confirm with the
   trained model and per-source thresholds.
6. **Exclusivity gain with real candidates:** re-run the argmax ablation.
7. **Empty-baseline submission:** its public score (expected about 0.056) confirms whether the test
   singleton rate matches train.

---

## Appendix: file index

| File | Content |
|---|---|
| `01_dataset_overview.csv` | Row, unique and duplicate counts per file |
| `02_data_quality.csv`, `tables/table_data_quality_census.csv` | Missing, empty and short/long fields |
| `country_distribution.csv` | Country share per split/source |
| `04_ground_truth_stats.csv`, `tables/table_match_cardinality.csv` | Singleton and cardinality distribution |
| `positive_similarity_summary.csv` | Similarity metrics on 246,867 true pairs, by source |
| `hard_negatives.csv`, `easy_negatives.csv`, `negative_similarity_summary.csv` | Negative pair samples and their similarity |
| `hard_negative_analysis.csv` | Positive vs negative separability (partly hard-coded) |
| `hard_singletons.csv`, `singleton_candidate_load_all.csv.gz` | Singleton deceptive candidates and candidate load |
| `03_normalization_comparison.csv`, `normalization_collision_examples.csv`, `tables/table_collision_risk_study.csv` | Normalization variants and collisions |
| `false_collision_examples.csv` | Largest common-name blocks |
| `05_blocking_benchmark*.csv`, `blocking_combo_benchmark.csv`, `blocking_benchmark_by_source.csv` | Blocking channel and union recall/volume |
| `retrieval_benchmark.csv`, `tables/table_retrieval_k_sweep.csv`, `tables/table_all_match_recall_by_cardinality.csv` | Top-K pilot |
| `tables/table_exclusivity_ablation.csv` | Argmax pilot |
| `tables/table_france_lexicon.csv` | France suffixes, address tokens, accents |
| `source_comparison.csv`, `evidence_matrix.csv`, `failure_mode_matrix.csv`, `experiment_log.csv` | Narrative summaries (mix of exact and hard-coded values) |
| `cache/token_stats.json` | Token, suffix and length statistics |
| `positive_pairs.tsv.gz` (336 MB) | All positive pairs with features (not opened for this document) |
| `analysis/*.py` | Scripts; `run_eda.py` is the main exact pipeline; Brahmi transliterator in `run_retrieval_benchmark.py` |
