# Solution Deep-Dive — Amazon ML Challenge 2026: Business Entity Resolution

Self-contained technical writeup of the actual implementation, written to be handed to another AI
session (or a teammate) with zero prior context. It describes what the code *does*, file by file,
at the level of design decisions and algorithms — not just the pipeline diagram. For rules/constraints
see `CLAUDE.md`; for chronological state see `HANDOFF.md`; for every experiment's numbers see
`experiments/log.csv`. All source lives under `code/business_entity_resolution/src/`.

## 1. Problem recap

Three noisy sources of business records (`business_name`, `business_address`, `country`), no shared
key. Source 1 (S1) is deduplicated — the canonical entity list. For every S1 entity, predict which
Source 2 / Source 3 ("pool") records refer to the same real business (zero, one, or many). Scored by
**macro F0.5 per S1 entity**: for a given entity, P = tp/|pred|, R = tp/|true|,
F0.5 = 1.25PR/(0.25P+R); empty-true & empty-pred = 1.0; empty-true & non-empty-pred = 0.0;
non-empty-true & (empty-pred or tp=0) = 0.0. Implemented exactly in `src/evaluate.py`.

Constraints that shaped every design choice: no external data/APIs; models ≤8B params, permissive
license; **country is an open set** — test includes France, absent from train, so country can only be
a *blocking* key, never a model feature; every test S1 row must appear in the output, empty list
allowed.

## 2. Pipeline shape

```
normalize (src/normalize.py)
   -> block / candidate generation (src/blocking.py)
   -> stage-1 pair features (src/features.py, src/noise.py)
   -> stage-1 LightGBM, GroupKFold OOF (src/model.py)
   -> stage-2 context features from stage-1 OOF scores (src/context.py)
   -> stage-2 LightGBM, GroupKFold OOF (src/model.py)
   -> calibration + decision layer (src/decide.py)
   -> write matching_results.tsv / candidate_pairs.tsv
```
Orchestrated end-to-end by `src/run_pipeline.py`. Everything is cached as parquet/JSON/TSV under
`cache/`, namespaced by `config.CACHE_TAG` so a normalization/blocking/feature change invalidates
stale caches by bumping the tag (currently should be v3, see §9).

## 3. Normalization (`src/normalize.py`)

Goal: turn raw `business_name`/`business_address` strings into a canonical form so that superficial
differences (case, punctuation, abbreviation, script) collapse, while keeping enough structure for
downstream similarity features.

Key pieces:
- **`split_raw`**: NFKC normalize, lowercase, `&` → `and`, strip single-letter abbreviation dots
  (`L.L.C.` → `LLC` via `_ABBR_DOT`), then split on a precomputed translation table (`_sep_table`)
  that maps every Unicode punctuation/symbol/separator/control character to a space. Native-script
  tokens are kept intact at this stage (transliteration happens later, per-token).
- **`tokens`**: for each raw token, if it's not ASCII, look it up in a *learned* transliteration
  table (see §7) first, falling back to `unidecode` (`to_ascii`); junk tokens (`null`/`none`/`nan`)
  are dropped.
- **`normalize_name`**: splits an alias marker (`dba|d/b/a|aka|a/k/a|fka|f/k/a|formerly (known as)|nee|
  doing business as`, via regex `_ALIAS`) out of the raw name into `main_text`/`alias_text` *before*
  tokenizing. Legal-suffix tokens are pulled out via a hand-written dictionary `LEGAL` (inc, corp,
  llc, ltd, pvt, llp, plc, sarl, sas, sa, eurl, sci, snc, cie, etablissements/ets, …) into their own
  `legal` column; the remaining tokens minus stopwords (`NAME_STOP`: and/the/of/de/du/des/la/le/les/
  et/d/l/www/com/net/org) become `name_core`. The alias text goes through the same
  legal/stopword split to produce `name_alias`. Returns
  `(name_norm, name_core, name_alias, legal, legal_first)` — `legal_first` flags whether the first
  token of the *main* name was a legal suffix (e.g. "Inc Foo Bar" vs "Foo Bar Inc" — same tokens,
  different position, used later as a noise signal for "legal suffix moved").
- **`normalize_address`**: splits off landmark clauses (`near|opp|behind|beside|next to|…` via regex
  `_LANDMARK`) into their own `landmark` field so "opposite City Hall" doesn't pollute address
  matching. Drops filler tokens (`ADDR_DROP`: no/hno/door/unit/number/po/box/null/…). Maps US state
  names to 2-letter codes (`_map_states`, handles 2-word states like "new york"). Canonicalizes street
  types via `ADDR_CANON` (street/road/avenue/boulevard/drive/lane/court/place/parkway/highway/circle/
  trail/terrace/square/apartment/suite/floor/building/north/south/east/west, **plus French**: rue→r,
  allee→all, impasse→imp, chemin→ch, route→rte, faubourg→fbg — and `saint`/`sainte` → `st`/`ste`, so
  the abbreviation collision with English "Street" is intentional and pair features are expected to
  disambiguate from context). Extracts digit tokens into `nums`, guesses a `postcode` (6-digit, or
  5-digit not in first position — India vs US postcode length heuristic), and builds a `street_key`
  = `"{house_number}_{next_alpha_token}"`, explicitly skipping French house-number modifiers
  `bis/ter/quater` (`_STREET_MOD`) so "12 bis rue de la Paix" and "12 rue de la Paix" key the same.
- **`normalize_frame`**: parallelized (joblib) driver that normalizes an entire source file and adds
  `rid` (0-based positional row id, the id space used everywhere internally instead of the string
  `entity_id`), `is_s3` (entity_id starts with "S3-"), and `script_flag` (name+address contains a
  non-Latin codepoint, via `_NON_LATIN = r"[^\x00-ɏ]"`-equivalent range check).

All dictionaries above are hand-written, not learned from any external data — satisfies the
"no external lookups, hand-written normalization dictionaries allowed" rule.

## 4. Blocking / candidate generation (`src/blocking.py`)

Goal: for each S1 entity, cheaply retrieve a small candidate set (tens, not hundreds) from the pool
with recall ≥0.97, so the expensive pairwise feature+model stage only scores plausible pairs, not the
full S1×pool cross product.

Mechanism: **IDF-weighted sparse matrix multiplication** over hashed "blocking keys," done separately
per country label (train shows literally 0 cross-country matches, so this loses no recall and keeps
each matmul small). No external index library (e.g. FAISS) — hand-rolled with scipy sparse matrices.

Key families (each record produces a *set* of `(family, key)` pairs, `record_keys`):
- `n` — plain `name_core` tokens.
- `nb` — adjacent token bigrams (`"a_b"`), catches word-order-sensitive similarity.
- `nm` — sorted-initials monogram (all first letters of tokens, sorted, joined) — catches heavy
  reordering.
- `np` — phonetic skeleton per token (via `normalize.phonetic`, see below) — bridges transliteration
  variants.
- `n2` — unordered pairs of the first `PAIR_TOKS=6` distinct tokens — catches reordering + inserted
  words, capped so it doesn't grow quadratically on long names.
- `nc` — all tokens concatenated with no separator — catches domain-style/no-space name variants.
- `a` — `addr_alpha` tokens (address tokens with no digit).
- `ab` — the `street_key` (house number + street name token).
- `na` — **cross key**: first `NA_NAME_TOKS=3` name tokens × first `NA_ADDR_TOKS=6` distinct address
  alpha tokens, joined `"{name}@{place}"`. Rationale in the code comment: a common name token alone,
  or a common place token alone, exceeds the max document-frequency cutoff and gets dropped as
  uninformative, but the *conjunction* ("smith" at "main st") is selective enough to survive.

`normalize.phonetic` (used for the `np` family): a hand-rolled, language-agnostic consonant skeleton
— `ph→f`, then a translation table `c/q→k, x→ks, w→v, z→s`, drop `h` after a consonant (kh/th/sh/bh…),
keep only the first character among vowels (a/e/i/o/u/y removed elsewhere), collapse repeated letters.
Example from the docstring: unidecode of a Devanagari "परफेक्ट" and the Latin "perfect" both reduce to
`"prfkt"`. This is what lets an Indian pool record's transliterated name still hash-collide with the
S1 Latin name even when `unidecode` produces a different-looking ASCII string.

Hashing: `key_hash(fam, key)` = first 8 bytes of BLAKE2b of `"{fam}:{key}"`, interpreted as a signed
int64 — stable across processes (unlike Python's salted `hash()`), used to bucket keys into a sparse
vocabulary without ever building a string vocabulary table.

Weighting (`config.BLOCK_W`, from EDA §G.8): `{"n":1.0,"nb":1.0,"nm":0.3,"np":0.5,"nc":1.5,"n2":0.7,
"a":1.0,"ab":2.5,"na":1.0}` — `ab` (street_key) and `nc` (concatenated name) carry the most weight per
occurrence; no postcode-based key family exists at all, because postcodes are present on ≤1.9% of
records (dead signal, confirmed by EDA).

Multi-channel union (`config.BLOCK_CHANNELS`): rather than one global top-K, three channels are
scored independently and unioned:
- `"name"`: `NAME_FAMS + CROSS_FAMS`, top-20 per S1 row.
- `"addr"`: `ADDR_FAMS` (`a`,`ab`), top-20.
- `"all"`: every family, top-30 — this channel's score becomes the candidate frame's `block_score`
  (the historical/default score column) and its rank becomes `block_rank`.
This ensures a record that's strong on name-only or address-only still gets pulled in even if its
combined weighted score wouldn't make an "all-channels" top-30 cut. Each candidate row also records
`rank_<channel>` per channel (sentinel = that channel's own K, meaning "not retrieved by it") and
`n_channels` (how many non-"all" channels independently found the pair) — these become model features
signalling multi-channel agreement.

Core computation (`build_index`/`_score_chunk`): per country, build one weighted sparse S1×vocab
matrix per channel (`build_index`) — a key's document frequency is computed over the *pool* side only,
keys with pool-DF > `BLOCK_MAX_DF=2000` are dropped as too common to be useful, and `idf = log(n2/df)+1`.
S1 rows are chunked (`BLOCK_CHUNK=2000`) and each chunk is matrix-multiplied against the pool's
transposed binary matrix, giving a dense-ish score per (S1 row, pool row) pair per channel; `_topk_rows`
keeps only the top-K per S1 row per channel (via `lexsort` + rank cutoff, no full sort); channels are
unioned via a `(row, col)` composite-key merge so each surviving pair keeps its best per-channel scores
and ranks. Runs in parallel across chunks via joblib (`config.N_JOBS`).

`recall_at_ks` is a diagnostic-only function (not on the default pipeline path) that scores every pair
once at `max(K)` and re-applies smaller K's as cheap rank cutoffs, to sweep recall-vs-K without
rebuilding the sparse index per K — used for the recall@K sweep experiments in `experiments/log.csv`
(B01–B06).

There is also a **measurement toggle** `config.NAME_KEY_MODE` (env var `ER_NAME_KEY_MODE`, default
`"keys"`): a teammate's EDA (`docs/EDA_TEAMMATE.md §9`) found that using abbreviation-normalized
name-derived key families (`nb/nm/np/nc/n2`) directly as blocking *identity keys* inflates false
collision groups (raw 226 → abbreviation-expanded 429 in their 35k-sample study), and recommends
conservative (case-fold/punctuation-only) keys for blocking with abbreviation-expansion reserved for
*scoring features*. Setting `ER_NAME_KEY_MODE=features_only` excludes those families from
`blocking.py`'s key generation while `features.py`/`noise.py` keep computing similarity on the
abbreviation-normalized forms regardless. **Not yet measured/flipped as of the last commit** — it's a
documented future experiment, not the current default.

## 5. Stage-1 pair features (`src/features.py`, `src/noise.py`)

Given the candidate frame (s1_rid, pool_rid pairs) and the two normalized frames, `build_features`
computes one feature row per candidate pair, entirely rapidfuzz/numpy vectorized where possible,
falling back to a joblib-parallel pure-Python loop for set/noise features. Every feature is
country-agnostic by construction (country is never read here).

**Frequency/ambiguity features** (`add_frequency_columns`, computed once on the normalized frames
before feature extraction, not per-pair): `name_freq` = log1p(count of this exact `name_core` in the
pool), `first_tok_freq` = log1p(count of this name's first token in the pool), `street_freq` =
log1p(count of this `street_key` in the pool, but 0 for blank street_key so "no address" isn't treated
as "extremely common address"), `s1_name_dup` = log1p(count of other S1 entities sharing this exact
name_core, i.e. how much competition this S1 row has for a shared name). Motivation from
`docs/EDA_TEAMMATE.md`: 37.8% of singleton (non-matched) S1 entities have an exact same-name pool
record that is *not* their match — a common name needs corroborating address/number evidence, a rare
name can be trusted more on its own.

**Fuzzy string features** (`FUZZ` list, via `rapidfuzz.process.cpdist`, vectorized C computation):
ratio / token_set_ratio / token_sort_ratio / partial_ratio / Jaro-Winkler on `name_norm`; ratio /
token_set_ratio / Jaro-Winkler on `name_core`; ratio / token_set_ratio / partial_ratio on `addr_norm`;
token_set_ratio on `addr_alpha`.

**Alias-aware name feature** (`f_name_alias_best`): computes `name_core` (S1) vs `name_alias` (pool)
and `name_alias` (S1) vs `name_core` (pool) token_set_ratio, zeroed out when the alias side is empty,
and takes the max against the plain `name_core` vs `name_core` score — i.e. "best of name-vs-name,
name-vs-alias, alias-vs-name." This is the E06 addition that lets a DBA/AKA/formerly-known-as name on
either side of a pair still score well against the other side's canonical name.

**Set/number/legal features** (`_py_chunk`, pure Python per row, joblib-chunked):
- `f_name_jac` / `f_addr_jac` / `f_nums_jac`: Jaccard similarity of `name_core` tokens / `addr_alpha`
  tokens / `nums` digit-strings.
- `f_num1_eq`: whether the first number token matches exactly (-1.0 sentinel if either side has no
  numbers, distinguishing "no evidence" from "evidence of mismatch").
- `f_legal_code`: 0 both empty / 1 equal / 2 exactly-one-empty / 3 both-present-different.
- `f_core_len_diff`: absolute difference in name_core token count.
- 11 noise-channel features from `noise.py` (next section).

**Blocking-derived features**: `f_block_score`, `f_block_rank`, per-channel `f_score_<channel>` and
`f_rank_<channel>`, `f_n_channels` — lets the model learn "retrieved by which channels, at what rank"
as signal, not just raw string similarity.

**Record-level features per side** (`a`=S1, `b`=pool): `f_freq_{a,b}` (name_freq), `f_script_{a,b}`
(non-Latin script flag), `f_addr_empty_{a,b}`, `f_land_{a,b}` (has a landmark clause), `f_ntok_{a,b}`
(name_core token count), `f_first_tok_freq_{a,b}`, `f_street_freq_{a,b}`; plus S1-only `f_s1_dup_a`
(the `s1_name_dup` ambiguity feature — no pool-side equivalent since it's about S1-side competition).

**Noise-channel features** (`src/noise.py`) — this is one of the two named "differentiators" in the
codebase (`# differentiator #2` in the docstring). Instead of only measuring *how similar* two records
are, each feature tries to explain *why* they differ, by detecting a specific perturbation the noise
generator likely applied:
- `f_nz_num_digit_del` (`digit_deletion`): shorter house number equals the longer with exactly one
  digit deleted.
- `f_nz_num_missing_one`: exactly one side has a number token at all.
- `f_nz_name_generic_only`: the symmetric-difference of name tokens is a subset of a **learned generic
  vocabulary** (`fit_generic_vocab`) — tokens that appear disproportionately in the symmetric
  difference of *known matched* name pairs (≥300 occurrences, ≥50% of the token's total occurrences
  are in a diff, min length 3 chars so initials/transliteration fragments never qualify). This
  captures things like "Group"/"Enterprises"/"Solutions" being added/dropped without changing the core
  business identity. Learned once from train ground truth, cached as JSON, no external word list.
- `f_nz_name_core_equal`: exact name_core token-set equality.
- `f_nz_legal_moved`: same legal suffix present on both sides but at a different position
  (`legal_first` differs) — "Foo Inc" vs "Inc Foo".
- `f_nz_edit1_tokens` (`_edit1`): count of tokens unique to side A (len≥3) within Damerau-Levenshtein
  distance 1 of some token unique to side B — catches single-character typos localized to one token.
- `f_nz_addr_only`: name tokens share nothing at all, but address tokens are ≥0.6 Jaccard — i.e. "same
  place, totally different name string" (could be a real non-match, but the feature exists so the
  model can weigh it against everything else rather than the raw name-similarity feature drowning it
  out).
- `f_nz_accent_only`: normalized names are identical but raw names differ case-insensitively — an
  accent/diacritic-only difference invisible after normalization.
- `f_nz_addr_norm_only`: same idea for addresses.
- `f_nz_num_conflict`: both sides have a number, they differ, and it's *not* explained by single-digit
  deletion — a genuine contradiction signal (pushes toward non-match).
- `f_acronym`: one side's initials (≥2 tokens, letters joined) appear as a token of the other side, or
  equal the other side's own single-letter tokens joined — catches "IBM" ~ "International Business
  Machines" and "M E" ~ "Mahalakshmi Engineering".

## 6. Model (`src/model.py`) — two-stage LightGBM

**Validation discipline**: `GroupKFold(n_splits=5)` grouped by `s1_rid` — an S1 entity's pairs never
split across train/val within a fold, preventing leakage of "this S1 entity's other candidates" info
across the split. `train_oof` fits one `LGBMClassifier` per fold on that fold's (masked, weighted)
training rows, early-stopping on the fold's held-out validation rows (`config.EARLY_STOP=50`), and
writes out-of-fold predictions covering every row exactly once — this OOF vector is what everything
downstream (stage-2 features, calibration, decoder tuning) is built on, never in-fold predictions.

**Hyperparameters** (`config.LGB_PARAMS`): binary objective, `learning_rate=0.08`, `num_leaves=127`,
`min_child_samples=50`, `feature_fraction=0.8`, `bagging_fraction=0.8`/`bagging_freq=1`,
`lambda_l2=1.0`, up to 2000 trees with early stopping, fixed `random_state=SEED=42`.

**Sampling** (`sampling.training_rows`, referenced in `run_pipeline.py`, not shown above but wired
in): not every candidate pair is used to *fit* the model (though OOF scores every row) — negatives
with `block_rank < TRAIN_KEEP_RANK=10` are always kept, the remaining "easy" negatives are subsampled
at `TRAIN_EASY_RATE=25%` and weighted 4× to compensate, keeping training fast without biasing away
from easy negatives entirely.

**Stage 1** trains on the pure pair features described in §5 (`cols1 = model.feature_cols(feats)`,
i.e. every column prefixed `f_`).

**Stage 2** (`src/context.py`, the other named differentiator, `# differentiator #1a`) adds features
derived from stage-1's OOF probabilities (`p1`), never from in-fold scores — critical to avoid
leakage:
- `group_context`: within each S1 group, `f2_rank` (rank of this candidate by p1, ties broken by
  first-seen), `f2_gap_max` (gap to the group's best p1), `f2_n_hi` (count of candidates in-group with
  p1>0.5), `f2_gsize` (group size). Across S1 groups for the *same pool record*: `f2_best_other` (the
  best p1 this pool record achieves in any *other* S1 group — its own best if it's not this group's
  top choice, otherwise the second-best), `f2_gap_other` (this candidate's p1 minus that), `f2_n_claims`
  (how many S1 groups this pool record appears as a candidate in at all). These directly encode the
  one-to-one intuition — "is this pool record more strongly claimed by some other S1 entity?" — as
  continuous features the model can weigh, rather than a hard constraint.
- `triangle_features`: "true S2 and S3 matches of one entity also resemble each other" — for each
  candidate `i` in an S1 group, look at the group's top-`TRIANGLE_TOPM=5` other candidates `j` by p1
  (only candidates with p1 ≥ `TRIANGLE_PMIN=0.01` get triangle features at all, to bound cost), compute
  name+address token_set_ratio similarity between `i` and each `j`, and aggregate: `f2_support`
  (Σ p1_j × sim(i,j)), `f2_n_supporters` (count of `j` with p1_j>0.5 AND sim>0.8), `f2_max_sim_top`
  (max similarity to any top-m other candidate), `f2_sim_top_other_src` (similarity to the best
  *opposite-source* supporter, i.e. if `i` is from S2, look at S3 supporters and vice versa — since a
  true match is corroborated by evidence from the *other* pool source, not just more of the same
  source). This is a soft cluster-coherence signal: a candidate that resembles other high-confidence
  candidates in its S1 group is more likely itself correct.

Stage 2 trains a second GroupKFold LightGBM on `cols1 + f2_* columns`, producing `p2` — the final
probability used for decoding. Feature importances can be dumped via `ER_DUMP_IMPORTANCE=1` env var.

**Known bug** (see `HANDOFF.md`): on the last full SageMaker run, `models2` came back empty (`m2 =
None` in `run_pipeline.main`), silently falling back to stage-1-only scoring at test time
(`score_stream` path). Root cause not yet traced — likely candidate: `add_stage2_features` or the
stage-2 `train_oof` call erroring/short-circuiting on the full-data run in a way that wasn't reproduced
on the smaller dev run. This is the single biggest open bug in the codebase.

## 7. Transliteration (`src/translit.py`)

A **learned** (not external-data) native-script → Latin token table: for every ground-truth train pair
where the pool name is non-ASCII, if the S1 (Latin) name and the pool (native-script) name tokenize to
the *same number of tokens*, each native token is aligned positionally with the corresponding Latin
token (`fit_table`). A native token's canonical Latin mapping is its most common aligned Latin token,
kept only if it occurred ≥`min_support=3` times and captures ≥`min_share=0.6` of that native token's
total occurrences — filters out noisy/ambiguous alignments. Saved as a two-column TSV
(`translit.save_table`/`load_table`), used by `normalize.tokens` as the first-choice transliteration
before falling back to `unidecode`. This directly targets the France-risk finding that a plain
`unidecode` fallback alone underperforms on script variants seen in training.

## 8. Decision layer (`src/decide.py`) — the other named differentiator

Takes stage-2 (or stage-1-only) probabilities `p` and turns them into per-S1 output sets. Three
sequential transforms:

1. **Isotonic calibration** (`fit_calibrator`/`IsotonicRegression`) on OOF scores vs labels — turns
   raw model scores into calibrated probabilities, which matters because the next step reasons about
   probabilities as if they were literally P(match).
2. **Temperature scaling** (`temper`): sharpens (`tau<1`) or softens (`tau>1`) the calibrated
   probability in logit space — a tunable knob independent of the isotonic calibration's shape.
3. **Optional hard global one-to-one** (`one_to_one`): each pool record is kept only in the single S1
   group where it scores highest — a hard version of the group-competition signal that stage-2's
   `f2_gap_other` feature already encodes softly. Whether this hard constraint helps or hurts is
   *tuned*, not assumed (see below) — motivated by EDA confirming the true label distribution is
   exactly one-to-one on the training pool (0 of ~7.6M pool ids shared across S1 entities), but a hard
   constraint can still hurt if the model's raw ranking is imperfect.
4. **Subset selection**, one of two decoders:
   - `select_threshold`: keep `p ≥ t_match`, but predict the whole group empty if the group's best
     `p < t_empty` — a simple two-threshold baseline.
   - `select_expected_f05`: per S1 group, Monte-Carlo simulate `n_draws=128` independent Bernoulli
     draws of each candidate being a true match (using its calibrated `p`, capped at `kmax=12`
     candidates ranked by `p`, `p_min=0.01` floor to exclude scored-out candidates), compute the
     empirical expected F0.5 for choosing the top-`k` for every `k` from 0..kmax (using
     `extra` = expected true matches *outside* the scored candidates entirely, i.e. missed by
     blocking recall, added to the false-negative count so choosing k doesn't get free credit for
     blocking's own recall gap), and pick the `k` (including 0, compared against the analytically
     exact F0.5-of-empty-output probability `e0 = Π(1-p_i) × exp(-extra)`) that maximizes expected
     F0.5. This directly optimizes the actual competition metric under the model's own probability
     estimates, rather than a fixed global threshold — should be more robust when the "right" number
     of matches genuinely varies per entity.
5. **`tune`**: grid search over both decoder families × `one_to_one ∈ {True, False}`, scored by the
   *exact* competition metric on OOF predictions (`config.GRID_T_MATCH`, `GRID_T_EMPTY`, `GRID_TAU`,
   `GRID_LAM` — `lam` scales `extra` for the expected-F0.5 decoder). Picks whichever combination scores
   highest on OOF, printing every grid point's score. In exp E05 the expected-F0.5 decoder first beat
   the threshold decoder (tau=0.8, lam=0.0); the currently-recorded default is still threshold
   t_match=0.7/t_empty=0.3 pending a fresh tune on E06's features.

## 9. Orchestration (`src/run_pipeline.py`)

Ties every module together, with aggressive caching keyed by `config.CACHE_TAG` (bump when
normalize/blocking/feature logic changes so old parquet caches don't silently mismatch new code) —
E06's normalization changes required bumping this to v3.

Flow: fit/load the translit table → normalize train S1+pool → load ground truth → sample a stratified
subset of S1 entities (`sampling.stratified_s1`, by country × match-count bin) at `DEV_FRAC=10%` for
`--eval-only` runs or `TRAIN_FRAC=15%` for full runs → generate candidates (blocking) → report
blocking recall/cands-per-entity, optionally stop here (`--stop-after blocking`) → fit/load the
learned generic-word vocabulary → build stage-1 features → apply the negative-sampling training mask
→ `train_oof` stage 1 → optionally add stage-2 context features and `train_oof` stage 2 (skippable via
`--no-stage2` / `--stop-after stage1`) → calibrate + tune the decoder on OOF (`decide.tune`) → compute
the OOF breakdown via the exact metric (`Scorer` class wraps id-list conversion + `evaluate.score`) →
compute **cross-country transfer** (`transfer_report`: for every ordered pair of country labels present
in the sampled data, fit stage-1 fresh on one label's rows only, score the other label's rows, report
F0.5 — this is the literal France proxy check) unless `--no-transfer` → append one row to
`experiments/log.csv` → if `--eval-only`, stop; otherwise refit stage-1 and stage-2 (if available) on
*all* rows with `mean_best_iter`-derived tree counts (mean OOF best iteration ×1.1) → normalize test →
generate test candidates → score test pairs in memory-bounded chunks (`score_stream`, 2M pairs/chunk,
only keeps `(s1_rid, pool_rid, p)` per chunk) if stage 2 is unavailable, else full stage-1→stage-2
scoring → apply the tuned decoder → write `matching_results.tsv`/`candidate_pairs.tsv` in test-S1 order
(`write_outputs`) → run the official validator subprocess if present (`utils/validate_submission.py`,
never edited directly) → log final counts and elapsed time.

`Scorer.lists`/`io_utils.pairs_to_lists` convert `(s1_rid, pool_rid)` row pairs into the
`{s1_entity_id: [pool_entity_ids]}` dict shape the exact metric and the TSV writer both expect.

## 10. Notebooks (`code/business_entity_resolution/notebooks/`, `notebooks/`)

Two different notebook locations exist, for two different purposes:

### 10.1 `code/business_entity_resolution/notebooks/train.ipynb` — the pipeline, cell-by-cell

This is a **cell-by-cell mirror of `python -m src.run_pipeline`** — it imports the exact same `src/`
modules the CLI uses (not a reimplementation), so results match the CLI exactly and every stage still
hits the same `cache/` parquet files. It's designed to run unmodified both locally (16GB laptop,
Windows/Mac/Linux) and on the SageMaker `ml.r5.4xlarge` instance — cell 1 walks up from the notebook's
cwd looking for a `src/` directory so it doesn't matter whether Jupyter's working directory is
`notebooks/` or the package root.

Structure (12 numbered sections as markdown-delimited cells):
1. **Locate package + set run parameters** — the params cell is the notebook's control panel:
   `DATA_DIR`/`OUT_DIR`, `EVAL_ONLY` (True = dev OOF only, no test inference), `FRAC` (share of train
   S1 entities; None → `config.DEV_FRAC`/`TRAIN_FRAC`), `NO_STAGE2`, `NO_TRANSFER`, `EXP_ID`/`CHANGE`
   (non-empty `EXP_ID` appends a row to `experiments/log.csv`). Also sets `ER_JOBS`/`ER_LGB_JOBS` env
   vars *before* importing `src.config` (config reads them at import time) — Windows guidance baked
   into the markdown: keep `ER_JOBS` ≤4-6 (blocking spawns that many OS processes; more triggers
   `WinError 1450`, insufficient system resources), `ER_LGB_JOBS` (LightGBM threads, not processes)
   can go much higher safely.
2. Normalize + learned transliteration (train) — calls `run_pipeline.get_translit` +
   `load_split_norm`.
3. Sample S1 entities + positive pairs — calls `sample_s1`/`gt_rid_pairs` (note: the notebook calls
   `rp.sample_s1`, an older name; the current `run_pipeline.py` module function is
   `sampling.stratified_s1` wrapped inline in `main()` — **the notebook may be stale relative to the
   current CLI code path here**, worth checking before relying on it for a fresh run).
4. Blocking — `blocking.generate_candidates`, prints `block_recall`/`cands_per_entity`.
5. Stage-1 features — `features.build_features`, builds the label vector `y` from ground truth.
6. Stage-1 LightGBM (5-fold GroupKFold OOF) — ~20-25 min on a laptop at `FRAC=0.1`. Includes an
   inline matplotlib feature-importance bar chart (fold-0 model only, diagnostic, not persisted).
7. Stage 2 (optional) — skipped entirely when `NO_STAGE2=True` (the MVP path this notebook defaults
   to: **`NO_STAGE2 = True` is the checked-in default**, so a naive "Run All" reproduces the
   stage-1-only MVP, not the current best E06 config).
8. Decode — calibrate, tune threshold/expected-F0.5 decoders on OOF, print the breakdown.
9. Transfer report — US↔India proxy for France, skippable via `NO_TRANSFER`.
10. Log the experiment — appends to `experiments/log.csv` only if `EXP_ID` is non-empty.
11. Test inference — **skipped when `EVAL_ONLY=True`** (the default); when False, refits stage-1 (and
    stage-2 if trained) on all sampled rows at `mean_best_iter`-derived tree counts, frees train-only
    memory (`del feats, scored, sel; gc.collect()`), blocks + scores the test set.
12. Write outputs + validate — `write_outputs` + the official validator subprocess.
13. Save models + decoder locally — `m1_stage1.txt`/`m2_stage2.txt` (LightGBM native format via
    `booster_.save_model`) + `<tag>_decoder.json` under `config.CACHE_DIR`. Explicitly logs and skips
    saving `m2` when stage 2 wasn't trained (`models2` empty) rather than erroring — **this is the
    exact symptom described in `HANDOFF.md`'s open stage-2 bug**: the notebook's own code anticipates
    `m2 is None` as a possible (if unintended) outcome.

Re-running any cell is safe/idempotent thanks to `cached_frame`/`get_translit`/etc. — a kernel restart
resumes from cache instead of recomputing.

### 10.2 `notebooks/` (repo root) — read-only observability tools, no pipeline logic

Three notebooks, none of which call any modeling code themselves — they only read files the pipeline
already produced. All three locate `student_resource/` by walking up from cwd until they find
`CLAUDE.md`.

- **`00_dashboard.ipynb`** — experiment dashboard. Reads `experiments/log.csv` and plots F0.5
  (all/US/India/singletons) by `exp_id`; reads block_recall/cands_per_entity and bar-charts them
  against the 0.97 recall target line; reads every `submissions/*/meta.json` to table public-LB scores
  per upload and flag days that hit/exceed the 5-upload limit; tails the newest `*run*.log` in the repo
  root (last 15 lines); optionally lists running Python processes via `psutil` (useful for checking
  whether a background pipeline run is still alive before starting another).
- **`01_run_mvp.ipynb`** — operational checklist/launcher for a full run, structured as PASS/FAIL
  checks so it can be read top-to-bottom like a runbook (this mirrors `docs/PROMPTS.md` Prompt 1):
  **1A pre-flight** — no other `python.exe`/`run_pipeline` process already running (refuses to start a
  second full run, which would fight over the ~100GB of RAM), current git commit + confirms
  `experiments/log.csv` is non-empty, runs `pytest -q tests`, checks expected `v1_*` cache files exist.
  **1B** — starts `python -m src.run_pipeline --frac 0.1 --no-stage2 --no-transfer --exp-id E03
  --change "MVP run (notebook)"` as a detached background `subprocess.Popen`, logging to
  `mvp_run.log` at the repo root (**note the hard-coded `v1` cache-tag expectation and `--exp-id E03`
  — this notebook was written for the original MVP run and would need updating to launch a fresh
  experiment with the current cache tag**). A separate cell tails the log and greps for
  Traceback/Error/WinError/MemoryError. **1C** — runs the official validator via subprocess and prints
  its stdout/stderr, then dumps any `docs/QA_*.md` files found.
- **`02_errors.ipynb`** — error-analysis / manual QA tool. Loads a given cache tag's cached candidates
  + stage-1 features, **retrains stage-1 OOF from scratch in-notebook** (not persisted — ~20-25 min at
  `FRAC=0.1`), then buckets every scoring error into one of: `blocking_miss` (true pair blocking never
  proposed at all), `false_negative_scored_low` (blocking found it, model scored it below `t_match`),
  `false_positive` (model scored an incorrect pair above `t_match`) — each further sub-classified by a
  heuristic `classify()` function into `native_script` / `generic_word` / `legal_moved` / `addr_norm` /
  `number` / `other` using the noise-channel feature flags from §5. Joins back to the *raw*
  (pre-normalization) `business_name`/`business_address` strings so a human can eyeball real examples.
  Filterable by `COUNTRY`/`ERROR_TYPE` params at the top. This is the tool that produced the findings
  written up in `docs/ERRORS.md`, which fed directly into the E06 normalization-v2 changes (alias
  splitting, French `etablissements`/`bis`/`ter` handling — see CLAUDE.md's Prompt 8 reference to this
  exact audit trail).

## 11. Output-writing scripts (`code/business_entity_resolution/scripts/`)

Four standalone scripts, none of which are imported by `src/run_pipeline.py` — they're operator tools
run manually around a pipeline run or an upload.

- **`make_empty_submission.py`** — writes the **E00 baseline**: every test S1 id with empty
  `matched_entity_ids`/`candidate_entity_ids`. Purpose stated in its docstring: verify the upload path
  works at all before any real model exists; expected public LB ≈ the singleton rate (~0.056, from
  EDA). Uses `io_utils.write_lists` directly, no pipeline dependency.
- **`archive_submission.py`** — run immediately after a real leaderboard upload:
  `python scripts/archive_submission.py --tag sub-dN-MM --exp-id EXX [--public-lb 0.61]`. Gzips the
  current `output/matching_results.tsv` and `output/candidate_pairs.tsv` into
  `submissions/<tag>/{matching_results.tsv.gz,candidate_pairs.tsv.gz}`, writes `meta.json` (tag,
  exp_id, public_lb, current git commit hash, whether the working tree was dirty at archive time,
  timestamp), and creates a git tag `<tag>` on the current commit — this is what gives every upload a
  reproducible commit pointer. `submissions/` itself is gitignored (large binary files) and meant to be
  synced to the shared team S3 bucket instead.
- **`make_zip.py`** — builds the final `<team>_submission.zip` in the **exact** layout the official
  problem statement requires: `output/matching_results.tsv`, `output/candidate_pairs.tsv`, every `*.py`
  under `src/` (recursively — naturally excludes `__pycache__`), `README.md`, `requirements.txt`, and
  `Documentation_template.md` at the zip root. Deliberately excludes `cache/`, `tests/`, `notebooks/`,
  `.venv/` (not part of the required layout, and `cache/` can be huge). `check_required_inputs()` fails
  loudly with an actionable message if any required file is missing (e.g. forgot to run the pipeline
  first). `verify_zip()` reopens the written zip and asserts its member set exactly matches the
  expected set — guards against silently shipping an incomplete or polluted zip. Not yet run for a real
  submission per `HANDOFF.md`'s packaging-incomplete status.
- **`france_check.py`** — read-only diagnostic comparing France's match behavior to US/India on the
  *current* `output/matching_results.tsv` (never calls the pipeline or writes anything). Computes, per
  country label found in `test_source1.tsv`: share of entities with ≥1 match, and mean matches-per-
  matched-entity (averaged only over entities that got a match, so singletons don't dilute it). Flags
  France as "meaningfully different" if the match-rate gap vs the US/India average is ≥10 percentage
  points, or the mean-matches-per-matched relative gap is ≥25% — thresholds chosen because the
  competition metric is dominated by the empty-vs-non-empty decision and matched-set precision/recall,
  so a double-digit swing plausibly reflects a systematic blocking/decision problem on the unseen
  country label rather than noise. If different, recommends considering spending one of the 5/day
  uploads on a France-specific diagnostic (same submission, France rows emptied, to isolate France's
  contribution to the public LB). Can optionally take a `--scored` parquet/tsv dump of kept-match
  probabilities (`s1_id`, `p` columns) for a probability-distribution breakdown per country — this
  artifact isn't produced automatically by any current pipeline run; the docstring gives the exact
  one-liner to dump it from `run_pipeline.py`'s in-memory `selt` if/when someone wants that view.

## 12. Output directory and submission history — current on-disk state

**`output/matching_results.tsv` and `output/candidate_pairs.tsv` (as of this writeup) are the
all-empty E00 baseline, not a real model run**: both files have exactly 1,732,545 lines (header +
1,732,544 test S1 entities) and **every single row has an empty second column** — verified by counting
non-empty rows directly (0 non-empty out of 1,732,544). `HANDOFF.md`'s shift log describes a real
stage-1-only MVP run having been completed and written to `output/` on 2026-09-25 (122,399 empty /
1,610,145 non-empty per its notes) and validated — but the files *currently on disk* don't match that
description, meaning either `make_empty_submission.py` was re-run afterward, or a later run failed
before actually populating matches (or fell all the way through the empty-gate `t_empty` threshold for
some reason). **This is worth checking before trusting `output/` for a fresh upload or zip build** —
run the pipeline (or at minimum inspect `cache/` for a completed `feats1_test`/`scored_t` artifact)
rather than assuming the current files reflect the best model.

`submissions/sub-d1-01/` is the only archived upload so far: `meta.json` records `exp_id: E03`,
`public_lb: 0.942`, commit `08ca592716c75d94bbaaa3903a65c74e65b60cbd`, `dirty: true` (working tree had
uncommitted changes at archive time), archived 2026-09-25 19:21:22 +0530. Its gzipped
`matching_results.tsv.gz`/`candidate_pairs.tsv.gz` are the real E03 MVP output (the one that scored
0.942 on the public leaderboard) — that historical upload's actual files live here, separate from
whatever is currently sitting in `output/`. One upload used out of 15 total / 5-per-day.

## 13. Current results (see `experiments/log.csv` for full history)

| exp | change | block recall | cands/entity | F0.5 all | F0.5 US | F0.5 India | F0.5 singletons | F0.5 matched | US→India | India→US |
|---|---|---|---|---|---|---|---|---|---|---|
| E03 | MVP: blocking B03 + stage-1 LGBM + tuned decoder | 0.9185 | 42.5 | 0.94642 | 0.95648 | 0.93133 | 0.95802 | 0.94573 | — | — |
| E04 | post-merge blocking (B05) + EDA weights + sampling + wider decoder grids | 0.9534 | 42.6 | 0.96274 | 0.96811 | 0.95469 | 0.95456 | 0.96322 | 0.89097 | 0.94047 |
| E05 | stage-2 context + competition + triangle features | 0.9534 | 42.6 | 0.96519 | 0.97073 | 0.95688 | 0.95651 | 0.96570 | 0.88893 | 0.93812 |
| **E06** | **normalization v2 (alias split + French ets/bis-ter) + retrieval-evidence/ambiguity features** | 0.9523 | 42.6 | **0.96817** | 0.97349 | 0.96019 | 0.96544 | 0.96833 | **0.89623** | **0.94474** |

E06 is the current best: it improved overall F0.5, singleton F0.5 (+0.89pt — the alias/frequency
features specifically help distinguish singletons from false-lead near-duplicates), *and* both
transfer directions simultaneously, with only a small blocking-recall cost (alias text moving out of
`name_core` changes what a couple of key families hash). Public leaderboard: only one upload spent so
far, 0.942 on the much earlier E03 config (before E04's blocking/feature gains) — a fresh upload with
E06 or its stage-2-fixed successor hasn't happened yet.

## 14. Open problems (see `HANDOFF.md` "Known bugs / blockers" for the living version)

1. **CORRECTED — not actually a bug.** This section previously described "`models2` comes back empty
   on the full run" as an unresolved bug needing a root-cause fix. On closer check it isn't one:
   `train.ipynb`'s one full (non-eval-only) run to date explicitly passed `--no-stage2` on purpose (MVP
   path); E05/E06's dev OOF numbers (0.96519/0.96817) already prove stage 2 trains and scores correctly
   whenever it's actually enabled. The real, narrower fact: no full run has *ever* had stage 2 enabled —
   an unexercised path, not a broken one. Just don't pass `--no-stage2` on the next full run.
2. **S3 cache-sharing bucket not writable** — IAM role on the SageMaker notebook instance needs an S3
   policy attached before `cache/` artifacts can be shared across teammates' AWS accounts.
3. **Blocking-key measurement toggle unflipped** — `ER_NAME_KEY_MODE=features_only` (per teammate EDA
   in `docs/EDA_TEAMMATE.md §9`) hasn't been measured against the current default on the full pipeline.
4. **Decoder not re-tuned since E06** — the recorded t_match/t_empty (0.7/0.3) predates E06's feature
   changes; `decide.tune` should be rerun once stage 2 is fixed.
5. **Packaging incomplete** — README, methodology write-up, and final submission zip not yet done
   (`scripts/make_zip.py` exists and self-verifies its layout, just hasn't been run for a real
   submission).
6. **`output/` currently holds the empty-baseline files, not a real model run** — see §12. Anyone
   about to upload or build the submission zip should regenerate `output/` first and re-check.
7. **`01_run_mvp.ipynb` is hard-coded to the original MVP run** (`--exp-id E03`, `v1` cache tag
   assumptions) — needs updating before it can launch a fresh experiment unattended.

## 15. File-by-file quick index

| File | Responsibility |
|---|---|
| `src/config.py` | Every path, seed, sampling/blocking/model/decoder hyperparameter — single source of truth, env-var overridable |
| `src/io_utils.py` | TSV load/write, ground-truth pair loading, id-list <-> pair-frame conversion |
| `src/normalize.py` | Name/address normalization, alias splitting, legal-suffix/street canonicalization, phonetic skeleton |
| `src/translit.py` | Learned native-script -> Latin token table from train ground truth |
| `src/blocking.py` | Hashed multi-family, multi-channel candidate generation via sparse matmul + IDF weighting |
| `src/noise.py` | Perturbation-operator ("noise channel") pair features + learned generic-word vocabulary |
| `src/features.py` | All stage-1 pair features (fuzzy strings, set overlaps, frequency, blocking-derived, noise) |
| `src/sampling.py` | Stratified dev/train S1 entity sampling; training-row subsampling/weighting for negatives |
| `src/model.py` | GroupKFold LightGBM training (OOF) and full refits |
| `src/context.py` | Stage-2 features derived from stage-1 OOF scores: group rank/competition + triangle support |
| `src/decide.py` | Isotonic calibration, temperature scaling, one-to-one, threshold and expected-F0.5 decoders, OOF tuning |
| `src/evaluate.py` | Exact competition metric (macro F0.5) + per-country/singleton/matched breakdowns + blocking recall report |
| `src/run_pipeline.py` | End-to-end CLI entry point wiring every module together, caching, logging, transfer report, output writing |
| `src/eda.py` | Standalone EDA script (not part of the runtime pipeline) |
| `src/io_utils.py` (detail) | Fast pyarrow-backed TSV load (~10x pandas C parser on multi-GB files), ground-truth dict parsing, submission TSV writer (`write_lists`), numpy-based pair-to-lists grouping (avoids `ArrowMemoryError` from pandas `agg(list)` at tens of millions of rows) |
| `src/sampling.py` (detail) | `stratified_s1`: samples S1 entities per (country × match-count bin: 0/1-2/3-4/5+) so dev/train subsets keep every stratum's share. `training_rows`: keeps all positives + all negatives with `block_rank < TRAIN_KEEP_RANK`, subsamples the rest at `TRAIN_EASY_RATE` with inverse-probability weighting |
| `code/business_entity_resolution/notebooks/train.ipynb` | Cell-by-cell mirror of `run_pipeline.main()` for interactive dev/SageMaker use — same `src/` calls, same caches |
| `notebooks/00_dashboard.ipynb` | Read-only: plots `experiments/log.csv`, submission history, tails the newest run log |
| `notebooks/01_run_mvp.ipynb` | Read-only pre-flight checklist + background-launches a pipeline run + runs the validator |
| `notebooks/02_errors.ipynb` | Read-only-ish (retrains stage-1 OOF in-memory) false-positive/false-negative error browser with raw-record join |
| `scripts/make_empty_submission.py` | Writes the E00 all-empty baseline submission |
| `scripts/archive_submission.py` | Gzip-archives `output/` + git-tags the commit after a real leaderboard upload |
| `scripts/make_zip.py` | Builds and self-verifies the final `<team>_submission.zip` in the required layout |
| `scripts/france_check.py` | Read-only France-vs-US/India match-rate diagnostic on `output/matching_results.tsv` |
| `output/` | `matching_results.tsv` + `candidate_pairs.tsv` — **currently the empty E00 baseline, see §12** |
| `submissions/<tag>/` | Gzipped archived upload + `meta.json` (exp_id, public_lb, git commit, dirty flag, timestamp); gitignored, git-tagged |
