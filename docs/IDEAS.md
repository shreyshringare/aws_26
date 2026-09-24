# Ideas: What We Are Building and What We Are Brainstorming

Amazon ML Challenge 2026 · Business Entity Resolution · living document (last updated 2026-09-25, Day 1).

- **Part A** is the committed solution: what the pipeline does and why.
- **Part B** is the idea backlog. Each idea gets a status, the evidence for or against it, and what would make us adopt it.

Related: `docs/STRATEGY.md` (strategy + compliance audit), `docs/EDA.md` (measured data facts), `docs/superpowers/specs/…` (design spec), `docs/superpowers/plans/…` (task plan), `HANDOFF.md` (shift state).

---

## Part A — The idea we are implementing

### A.1 The problem in one paragraph
Three sources of business records (`name`, `address`, `country`) with no shared key. Source 1 (S1) is a clean, deduplicated list. Sources 2 and 3 (S2/S3) are noisy copies of the same businesses, plus about 26% records that match nothing. For every S1 business, output the S2/S3 records that are the same business (0, 1 or many). The score is **macro F0.5 per S1 business**: per business F0.5 = 1.25·TP / (k + 0.25·G), then averaged. Precision counts double. A business with no true match scores 1 only if we predict nothing. The test set adds **France** (15% of test S1), which never appears in train.

### A.2 The central idea
> Don't treat matching as millions of independent yes/no questions. Treat each S1 business as a **small decision problem over a shortlist of candidates**. Score candidates with features that understand *how* the data was corrupted. Then decide *jointly*: candidates support each other, each record has one owner, and the chosen set maximises the metric we're graded on.

That idea breaks into five stages:

| # | Stage | What it does | Why (evidence from EDA) |
|---|---|---|---|
| 1 | **Normalize** | Canonicalise abbreviations (Road→rd, R.→r, Pvt→pvt), split legal suffixes, map US states, pull out landmarks, extract house numbers and street keys, transliterate Indian scripts with a **table learned from train pairs** | legal suffix missing on one side in 21.8% of pairs; 11–15% of pool names in native script |
| 2 | **Shortlist (blocking)** | Per country label: rare shared keys weighted by rarity (IDF) via sparse matrix products; key families = name words, name word pairs, initials, **phonetic skeleton**, **concatenated name** (for `name.com` style), address words, house#+street. **Multi-channel**: separate name / address / combined top-K slots, then union | 10M pool records; E01 v1 kept 90.2% of true pairs → upgrade to ≥ 97% in progress |
| 3 | **Score pairs (stage 1)** | LightGBM on string similarities (rapidfuzz, Jaccard, Jaro-Winkler) + **noise-channel features** + record statistics. Country is never a feature | noise operators measured: digit drop 5.5%, typo 4.7%, reorder 3.4%, trade-name swap 8.0% |
| 4 | **Group pass (stage 2)** | Features from stage-1 out-of-fold scores: rank in group, gap to best, competition from other S1s, **triangle support** (how much a candidate resembles the group's other strong candidates) → LightGBM #2 | mean 3.5 matches per business spread across S2 and S3; true matches resemble each other |
| 5 | **Decide** | Isotonic calibration → **global one-to-one** (each record to its best S1 only) → per business, choose the top-k (k may be 0) that **maximises expected F0.5** (Monte Carlo). A tuned threshold + empty gate is the fallback, and the one that scores better on OOF wins | one-to-one holds exactly (0 of 7.6M shared); singletons 5.6% |

### A.3 Our three differentiators
1. **Collective, metric-aware decoding.** Triangle support + one record/one owner + expected-F0.5 subset choice. Most entries use one global threshold on independent pair scores.
2. **Noise-channel features.** Each copy-error pattern of the data generator becomes an explicit feature: `num_digit_del`, `edit1_tokens`, `legal_moved`, `name_generic_only`, `addr_only` (trade-name swap), `accent_only`, `addr_norm_only`. The model learns which differences are harmless.
3. **Script-agnostic matching without external data.** A learned native→Latin word table (6,585 tokens) + a phonetic consonant skeleton ("मार्केटिंग" → `mrktng` = "marketing").

### A.4 How we validate
- **Dev loop:** 10% of train S1, searched against the *full* train pool. 5-fold GroupKFold by S1 business. Exact metric. Breakdown: all / per country / no-match / matched, plus blocking recall and candidates per business.
- **France proxy:** train on US → score India, and India → US. EDA warning: France's unseen-word rate is 45% vs 28–36% in the proxy, so the proxy is **optimistic** for France.
- **Rule for accepting a change:** it improves dev F0.5 **and** doesn't hurt transfer. The final submission is chosen by local numbers, not the public leaderboard.

### A.5 Hard constraints the idea respects
No external data or lookups · LightGBM (MIT, a few MB) · country used only to group comparisons · `candidate_pairs.tsv` = exactly what the model scores · validator before every upload · ≤ 5 uploads/day, each archived · docstring on every function · the zip reproduces outputs with one command.

### A.6 Status (Day 1)
| Stage | Status |
|---|---|
| Normalize + learned transliteration | ✅ done, tested |
| EDA v2 (measured) | ✅ `docs/EDA.md` |
| Blocking | 🔄 v1 recall 0.902; upgrade (phonetic / concatenated / pair keys, multi-channel) being measured |
| Stage-1 features, model, decoders | ✅ coded + unit-tested; first real run next |
| Stage-2 group pass | ⏭ Day 2 |
| Packaging (zip, methodology doc) | ⏭ Day 3 |

---

## Part B — Brainstorm backlog

Status legend: **✅ adopted** (in the pipeline) · **🔄 in progress** · **⏭ queued** (decided, not built) · **🧪 candidate** (worth an experiment if time allows) · **❌ rejected** (with reason).
One experiment per idea, logged in `experiments/log.csv`. Keep an idea only if dev F0.5 improves without hurting transfer.

### B.1 Shortlisting (blocking) — raises the ceiling on recall
| Idea | Status | Rationale / evidence | Adopt if |
|---|---|---|---|
| IDF-weighted rare-key blocking via sparse matmul, per country | ✅ | scales to 10M; E01 = 0.902 recall | — |
| Phonetic skeleton key (`np`) | 🔄 | covers native-script misses and some typos | recall ↑ |
| Concatenated-name key (`nc`) for `nexloixadvisory.com` | 🔄 | domain-style names share no tokens | recall ↑ |
| Unordered name-word pairs (`n2`) | 🔄 | survives a corrupted middle word and reordering (3.4%) | recall ↑ |
| Multi-channel top-K (name / address / combined) | 🔄 | name-only matches (empty pool address) were crowded out | recall ↑ at ≤ 60 cands |
| Re-weight families (raise `a`, `ab`; lower `nm`; drop `p`) | 🔄 | EDA §F: 98.4% of pairs share a surviving key; `a`/`ab` give unique recall; postcode is dead | recall ↑ |
| Character 3-gram key on name core | 🧪 | catches heavy OCR corruption ("Bhcfcle"); expensive | residual misses dominated by typos |
| Second-hop blocking via S2↔S3 links | 🧪 | if S3 record y strongly matches pool record x already shortlisted, add y (transitivity) | recall stuck below target |
| Adaptive K per business (bigger K when scores are flat) | 🧪 | a flat score profile means ambiguity | cands/entity budget allows |

### B.2 Normalization
| Idea | Status | Rationale / evidence | Adopt if |
|---|---|---|---|
| Hand-written dictionaries (legal, street, US states, FR words) | ✅ | listed in the methodology doc | — |
| Learned native→Latin table (position-aligned train pairs) | ✅ | 6,585 tokens | — |
| **Alias split** (`dba`, `aka`, `fka`, `formerly`, `doing business as`) → `name_alias`, best-of similarity | ⏭ | EDA: these words surfaced in the learned generic vocab | dev ↑ |
| French: `etablissements/ets` legal-like; skip `bis/ter` in street key; seed FR generic words (ecole, amicale, groupe…) | ⏭ | EDA §E: 45% France OOV; top missing FR tokens listed | no regression on US/IN |
| Better transliteration fit (align by edit distance when token counts differ; also fit on addresses) | 🧪 | the table only learns from equal-length pairs | native-script misses remain large |
| Generic vocab requires ≥ 3 chars | ⏭ | drops noise tokens (`a`, `jy`) | — |

### B.3 Features
| Idea | Status | Rationale / evidence | Adopt if |
|---|---|---|---|
| String similarities (rapidfuzz ratio/token-set/partial, Jaro-Winkler, Jaccard) | ✅ | standard, cheap, language-agnostic | — |
| Noise-channel features (9) | ✅ | measured operator frequencies (EDA §C) | — |
| Per-channel blocking scores (`f_score_name`, `f_score_addr`) | 🔄 | comes with multi-channel blocking | — |
| Drop postcode features | ⏭ | ≤ 1.9% coverage | — |
| Address-shape features (token count, has landmark, has number) | ⏭ | US vs India differ strongly; shape transfers, country doesn't | transfer holds |
| Name-frequency / ambiguity features (how many S1s share this name) | ⏭ | 48–54% of S1 share their exact name | dev ↑ |
| Phonetic-skeleton similarity feature | 🧪 | extends script-agnostic matching into scoring | native-script errors remain |
| Char n-gram TF-IDF cosine (name, address) | 🧪 | robust to typos and unseen FR words | France proxy ↑ |
| Small multilingual embedding similarity (multilingual-e5-small, MIT) | 🧪 | semantic + cross-script; needs GPU/time | only if CPU budget allows and transfer ↑ |

### B.4 Training data picking
| Idea | Status | Rationale / evidence | Adopt if |
|---|---|---|---|
| Dev/train businesses stratified by **country × match-count bin** (0 / 1–2 / 3–4 / 5+) | ⏭ | EDA: 71% of businesses have 2–5 matches; singletons 5.6% | — |
| **Hard-negative-aware rows:** all positives + all negatives in block ranks < 10 + 25% of the rest (weight 4) | ⏭ | ~2× faster training → 2–3× more businesses for the same compute | dev F0.5 ≥ baseline |
| Emphasise same-name / different-address negatives | 🧪 | EDA §D: 60.5% of S1 have such collision negatives | FP analysis shows name-collision merges |
| Train on more businesses on SageMaker (TRAIN_FRAC 0.15 → 0.4) | 🧪 | more variety; memory OK on 128 GB | learning curve still rising |

### B.5 Model
| Idea | Status | Rationale / evidence | Adopt if |
|---|---|---|---|
| LightGBM binary, GroupKFold by S1, early stopping | ✅ | fast, MIT, strong on tabular features | — |
| Stage 2 (group context, competition, triangle support) | ⏭ | differentiator #1a | dev ↑, transfer holds |
| LambdaRank objective (rank within an S1 group) | 🧪 | aligns with "pick top-k per business" | beats binary on dev F0.5 |
| Separate heads for S2 vs S3 candidates | 🧪 | sources may have different noise | feature importance shows strong `is_s3` interactions |
| Seed / fold ensembling | 🧪 | cheap variance reduction | Day 3 if time allows |

### B.6 Decision layer
| Idea | Status | Rationale / evidence | Adopt if |
|---|---|---|---|
| Isotonic calibration on OOF | ✅ | expected-F0.5 needs calibrated probabilities | — |
| Global one-to-one assignment | ✅ | exact in train | — |
| Expected-F0.5 subset selection (Monte Carlo, includes "none") | ✅ | differentiator #1b | beats threshold on OOF |
| Threshold + empty-gate baseline | ✅ | fallback | — |
| Joint S2/S3 cluster decision (pick consistent clusters, not just top-k) | 🧪 | uses S2↔S3 similarity at decision time | stage-2 triangle features aren't enough |
| Cardinality prior (predicted number of matches per business) | 🧪 | EDA shows a tight count distribution (p90 = 6) | EF0.5 over/under-selects |

### B.7 France-specific (unseen country)
| Idea | Status | Rationale / evidence | Adopt if |
|---|---|---|---|
| Country never a feature; universal dictionaries incl. French words | ✅ | rule + transfer | — |
| Rarity and frequency statistics computed on the **test** pool per country | ✅ | unlabeled test data, no external data | — |
| Pseudo-labelling France (retrain with very-high-confidence test matches) | 🧪 | adapts to 45% unseen vocab | the US→IN pseudo-label simulation improves transfer |
| Diagnostic upload: same file with France rows emptied vs predicted | 🧪 | measures France contribution on the public LB | we have a spare upload on Day 2 |
| Adversarial validation (train vs test France) to find shifted features | 🧪 | flags features that behave differently | time allows |

### B.8 Rejected ideas
| Idea | Why rejected |
|---|---|
| LLM as the matcher | slow at 70M pairs; CLAUDE.md says models are features at most; licence/size limits |
| Geocoding / business registries / web lookups | explicitly prohibited → disqualification |
| Country one-hot or per-country models | breaks on France; explicitly forbidden |
| All-pairs comparison | 1.7M × 10M infeasible |
| Pairwise-F1 threshold tuning | wrong metric; we tune on the exact macro F0.5 |
| Top-1 match per business | caps recall at ~1/3.5 (mean 3.46 matches) |

---

## How to add an idea
Add a row in the right table with status 🧪, one line of rationale (ideally an EDA number), and a concrete "adopt if" test. When you run it, log an experiment row and update the status here and in `HANDOFF.md`.
