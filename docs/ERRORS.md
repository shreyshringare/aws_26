# Error Taxonomy — Prompt 7 (exp E05, ef05 decoder tau=0.8 lam=0.0)

Sampled 200 errors from the E05 OOF cache (`cache/v2_oof_scores_0.1.parquet`, 220,682 dev S1 entities,
9,399,349 scored candidate pairs): 100 false positives (predicted match, ground truth says no) and 100
false negatives, the latter split proportionally between **blocking misses** (true pair never reached the
candidate set — 35,617 of 763,739 dev positives, 4.66%, matching B05's `1 - 0.9534` recall gap exactly)
and **scored-but-rejected** pairs (candidate existed, model/decoder said no — 23,656 of 763,739).

Classification is heuristic (rule-based on script flag, alias regex, legal-suffix mismatch, numeric-token
overlap, name_core equality, address-token Jaccard, score proximity to 0.5) — not hand-labeled. The large
`other` bucket, especially in FP (47/100), reflects real classifier imprecision, not that half of errors
are unexplainable; see the meta-finding at the end.

## Counts

| class | FP (of 100) | FN (of 100) |
|---|---|---|
| blocking_miss | — | 60 |
| threshold | 2 | 18 |
| number_conflict | 14 | 8 |
| other | 47 | 5 |
| abbreviation_alias | 11 | 4 |
| native_script | 15 | 3 |
| address_normalization | 2 | 2 |
| name_collision | 9 | — |

## Class detail: examples, fix, ranked by expected gain

### 1. `blocking_miss` (FN) — largest single class, 4.66% of all true pairs

Examples:
- `Bangalore North Infratech` (India) vs `Mirariza` — no shared name tokens, address shares only house
  number fragment "B203"/"203". Genuinely hard: neither name nor address blocking channels have a
  common key.
- `Physical Therapy Premier Physicians, LLC` (US) vs `#Physicaltherapy` — the pool side is a
  concatenated/hashtag-style name. Our `nc` (concatenated-name) blocking family exists exactly for this
  case but didn't retrieve it — worth checking `nc`'s weight/top-k against this pattern specifically.
- `Hoskinson and Rodriguez LLC` vs `Hoskinson & Rordgbuez LLC` — heavy character-level typo
  ("Rordgbuez" vs "Rodriguez"), not an abbreviation or word-order issue. Token-based blocking keys
  cannot catch this; this is exactly Prompt 4B's char n-gram channel use case.

**Fix, ranked #1 (highest expected gain — this is the largest error class by far):** continue the
blocking work already in progress (Prompt 4 landed B05 at 0.9534 recall, up from 0.9185). Two concrete
sub-fixes surfaced by these examples specifically: (a) check `nc` family's top-k against concatenated/
hashtag-style names, (b) if recall plateaus, Prompt 4B's char 3-4-gram channel directly targets the
heavy-typo case.

### 2. `threshold` (FN 18, FP 2) — second largest, scores sitting near the decision boundary

Examples:
- `Shriji Online Private Limited` vs `Private Shriji Online Ltd` (p=0.657) — word-order swap + legal
  suffix variant, both present, scored just under a typical cutoff.
- `Helios Heritage Co` vs `helios heritage co` (p=0.408) — case-only difference plus empty pool address;
  scored surprisingly low for what looks like an easy positive.
- `Costa Pediatrics LLC` vs `Fluxvantageveo` (p=0.595) — near-identical address, completely different
  name. Ambiguous even for a human without more context.

**Fix, ranked #2:** the `Helios Heritage` case in particular ties directly to the E04b gap already queued
in `docs/NEXT_STEPS.md` item 9 — empty pool address scores as "totally different" (0 not NaN), likely
dragging an otherwise-easy match's score down. Re-tuning the ef05 decoder's `tau`/`lam` after stage-2
(already done for E05) helps at the margin but won't fix systematically underscored easy cases like this.

### 3. `number_conflict` (FP 14, FN 8)

Examples:
- FP: `Data Mobility Collective L.L.C.` "420 Main Street" vs "433 Main St" — different house numbers,
  same street, model scored it a match (p=0.869) anyway.
- FN: `Aurora Pony Co` "100 N Norene Circle" vs "113 North Norene Circle" (p=0.650) — same pattern,
  opposite outcome: correctly suppressed here, missed there. Inconsistent handling of the same error type.
- FN: `Mega Software Private Limited` — very low score (p=0.106) with both a number mismatch and address
  OCR noise ("Bhayaniskyline" vs "Bhayaniskylise") compounding.

**Fix, ranked #3:** this is exactly the near-miss number flag gap already queued in `docs/NEXT_STEPS.md`
item 9 — current features have exact-number-Jaccard and first-number-equal, but nothing that
distinguishes "numbers differ by a little" (likely same place, a typo) from "numbers differ a lot"
(likely different place). The FP/FN inconsistency above (420↔433 wrongly kept, 100↔113 correctly
dropped) suggests the model has no consistent signal here at all — a real feature gap, not noise.

### 4. `abbreviation_alias` (FP 11, FN 4)

Examples:
- FP: `Jennette Fontana, DPM` vs `JENNETTE FONTANA, DPM (CORPORATION)` — legal-suffix-only difference,
  empty pool address, scored as a match (correctly, arguably — but labeled FP by ground truth, worth a
  second look at whether this is a true FP or a label edge case).
- FN: `Jackson Keystone Btc Corp` vs `Jackson Keystone Btc` (p=0.101) — legal suffix dropped, scored very
  low despite otherwise-strong name match.
- FN: `Raja Developersprivate LLP` vs `Raja Developersprivate` (p=0.472) — same pattern.

**Fix, ranked #4:** `docs/PROMPTS.md` Prompt 8 already targets alias splitting (`dba`/`aka`/`fka`/
`formerly`) — these examples are legal-suffix-only mismatches rather than DBA-style aliases, a
related but distinct sub-case. Worth confirming Prompt 8's alias work also covers bare legal-suffix drops,
not just the `dba`-keyword pattern.

### 5. `native_script` (FP 15, FN 3) — India-specific, concentrated in South Indian scripts

Examples:
- FN: `Lakshmi Vision Enterprises Pvt Ltd` vs its Tamil-script twin (p=0.698) — scored moderately, not
  confidently.
- FN: `RZ Solutions Private Limited` vs Kannada-script version (p=0.003) — scored almost zero despite an
  otherwise-clean transliteration case (exact address number match).
- FN: `East Exports Pvt Ltd` vs Kannada-script version (p=0.565).

**Fix, ranked #5:** the `RZ Solutions` case (p=0.003, near-total miss) is the concerning one — suggests
the learned transliteration table may have weaker coverage for Kannada than for Devanagari/Hindi (the
majority of India's training data). Worth checking `translit.py`'s per-script token coverage instead of
treating "native script" as one uniform problem.

### 6. `name_collision` (FP 9) — mostly accent/diacritic near-duplicates, not true collisions

Examples:
- `Real Das Pvt Ltd` vs `Real Dás Pvt` — one accented character difference.
- `Tavue` vs `Távue` — same pattern.

**Fix, ranked #6 (lower priority, likely already near-correctly handled):** these read less like genuine
"same name, different business" collisions (the taxonomy's intended meaning) and more like normalization
correctly collapsing accent variants to the same `name_core`, with the ground truth's non-match label
being the actual surprise — possible label noise, or a real distinct-business-same-name coincidence
that only address should be disambiguating (both examples have empty pool addresses, so there's no
address signal to fall back on). Not clearly a normalization bug; flag for spot-check rather than a code
fix.

### 7. `address_normalization` (FP 2, FN 2) — smallest sampled class

Examples:
- FP: `HWB Coal Private Limited` vs `MHWB Coal (Private-Limited)` — extra leading "M", state abbreviation
  mismatch (Telangana vs "TG").
- FN: `One Properties` vs `M/s One Services` — "M/s" prefix, different address components present on
  each side.

**Fix, ranked #7:** small sample, low volume relative to other classes — not a priority fix on its own,
though the "M/s" prefix pattern is a cheap addition to the abbreviation dictionary alongside Prompt 8's
existing alias work.

### 8. `other` (FP 47, FN 5) — the meta-finding

This bucket is large enough that it dominates the FP sample, which says more about the classifier's
heuristics being too narrow than about there being one dominant unclassified failure mode. A few real
patterns visible on inspection, not covered by the taxonomy as given:
- **Trade-name swap at a shared address**: `EQD (india) Private Limited` vs `Solonyx` — identical
  address, completely unrelated names. This is the "no shared core token, trade-name swap" noise pattern
  already measured in our own `docs/EDA.md` §C (8.0% of pairs). Worth adding as an explicit 9th taxonomy
  class rather than leaving it in `other`.
- **Possible ground-truth-adjacent cases**: `Vibgyor Infraventures Private Limited` vs `Shri Vibgyor
  Infraventures Limited` — near-identical name and address (p=0.976), labeled FP. Worth a manual spot
  check of a handful of high-confidence `other`-class FPs to see if any are actual label noise rather
  than model error, before assuming every FP here needs a feature fix.

**Fix, ranked #8:** re-run this classification with a `trade_name_swap` class added (shared address,
disjoint name tokens) before drawing further conclusions from `other` — the current heuristic just
doesn't have a bucket for the biggest visible real pattern in that group.

## What NOT to conclude from this analysis

- Don't treat `other`'s size as "half of errors are mysterious" — it's a classifier gap, not a model gap.
- Don't fix `name_collision` — the examples suggest it may not be a real problem class here.
- `address_normalization`'s low sample count (2 and 2) means don't over-index on it vs the taxonomy's
  a-priori importance; the data just didn't surface much of it in this particular sample.

## Queued fixes

Added to `docs/IDEAS.md` Part B as queued items, ranked by this analysis:
1. Near-miss number flag (ties to existing `docs/NEXT_STEPS.md` item 9) — confirmed by real FP/FN
   inconsistency on the same number-mismatch pattern.
2. `nc` (concatenated-name) blocking family — check weight/top-k against hashtag/concatenated-name misses.
3. Kannada-specific transliteration coverage check in `translit.py`.
4. Add `trade_name_swap` (shared address, disjoint name) as an explicit taxonomy class and re-sample.
5. Bare legal-suffix-drop as a sub-case of Prompt 8's alias work, not just `dba`-keyword aliases.
6. Spot-check a handful of high-confidence `other`-class FPs for label noise before further feature work.
