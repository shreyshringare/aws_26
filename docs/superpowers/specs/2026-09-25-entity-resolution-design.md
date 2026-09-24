# Design: Business Entity Resolution — Amazon ML Challenge 2026

Date: 2026-09-25 · Status: draft for team review
Governing docs: problem statement PDF, guidelines PDF, `Documentation_template.md`, `CLAUDE.md`.

## 1. Goal and success criteria
For every test S1 entity, output the S2/S3 records that are the same business, maximising
macro F0.5 per S1 entity (private leaderboard decides). Success criteria:
- Local OOF macro F0.5 is the primary metric; the US→IN and IN→US transfer scores act as the France proxy.
- Blocking pair recall ≥ 0.97 with ≤ 40 candidates per S1 entity.
- Every upload passes `utils/validate_submission.py`; the final zip matches the required structure.
- The full pipeline reruns from raw data with one command.

## 2. EDA facts the design relies on (train)
| Fact | Value | Design consequence |
|---|---|---|
| Scale | S1 2.2M, S2 5.0M, S3 5.3M (test: 1.73M / 4.9M / 5.1M) | sparse/vectorised blocking, per-country chunks, SageMaker 128 GB |
| Singleton rate | 5.6% (same in US and IN) | recall matters; empty prediction chosen by expected-F0.5, not a fixed gate |
| Matches per entity | mean ~3.5, max 11; S2 and S3 both 0–6 | multi-match subset selection |
| One-to-one | 0 of 7.6M pool ids linked to >1 S1 | global one-to-one assignment is exact |
| Cross-country pairs | 0 | block within country label |
| Unmatched pool records | ~26% | distractors; need calibrated probabilities |
| Native-script names | 11–15% of S2/S3 names | transliteration table learned from train pairs |
| Test France share | 259k of 1.73M S1 (15%) | country-agnostic features; FR abbreviation dictionary |
| Observed noise | house-number digit drop (3424→424), suffix moved to front, generic words added, accents, TX↔Texas, typos, empty address, trade-name swap | noise-channel features (§7.2) |

## 3. Compliance constraints (non-negotiable)
- Only files under `dataset/`. Hand-written dictionaries are allowed and are listed in the methodology appendix.
- Models: LightGBM (MIT) only. No LLM or embedding model in v1.
- `country` is used only as a blocking group key over whatever labels exist. Never as a feature, filter or one-hot.
- Output: exact headers, one row per test S1 in `test_source1` order, S2-/S3- ids only, no duplicates, matches ⊆ candidates.
- `candidate_pairs.tsv` is exactly the pair set the model scores (the final blocking stage).
- Every function has a docstring. Versions are pinned. Seed = 42.
- Methodology doc: `Documentation_template.md` filled in. Core sections ≤ 2 pages (guidelines); detail goes in the Appendix.

## 4. Architecture
```
raw TSV ─▶ normalize ─▶ blocking ─▶ features(stage1) ─▶ LGBM-1 (OOF) ─▶ context+triangle features ─▶ LGBM-2 (OOF)
                                                                                                        │
                                  output TSVs ◀─ expected-F0.5 subset selection ◀─ global one-to-one ◀──┘
```
Each stage reads and writes parquet in `cache/`, following these contracts:

| Stage | File | Columns |
|---|---|---|
| normalize | `norm_{split}_{s1,pool}.parquet` | rid, entity_id, country, name_norm, name_core, legal, addr_norm, addr_tokens, nums, postcode, script_flag |
| blocking | `cands_{split}.parquet` | s1_rid, pool_rid, block_score, block_rank |
| features | `feats1_{split}.parquet` | s1_rid, pool_rid, f_* |
| model | `scores_{split}.parquet` | s1_rid, pool_rid, p1, p2 (OOF on train) |
| decode | `output/*.tsv` | submission format |

The modules are the ones listed in CLAUDE.md, plus `translit.py` (learned table), `noise.py` (noise-channel features) and `decide.py` (§8, collective decoding).

## 5. Normalization (`normalize.py`, `translit.py`)
- **Common cleaning:** NFKC, then transliterate non-ASCII tokens (learned table first, `unidecode` fallback), lowercase, delete dots (`L.L.C.`→`llc`, `R.`→`r`), `&`→`and`, remaining punctuation → space, collapse spaces, strip junk tokens (`--`, `<<`, `null`).
- **Name:** `legal` holds the canonical legal tokens found anywhere in the name (inc/incorporated, llc, corp→corporation, co→company, ltd→limited, pvt→private, llp, plc, sarl, sas, sasu, sa, eurl, sci, snc, opc…). `name_core` = the name without legal tokens and stopwords (and, the, of, de, du, des, la, le, les, et).
- **Address:** token dictionary (rd→road, st→street, ave/av→avenue, blvd/bd→boulevard, dr→drive, ln→lane, ct→court, pkwy→parkway, hwy→highway, r→rue, all→allee, imp→impasse, ch→chemin, apt→apartment, ste→suite, fl→floor, no/hno/h/door/unit dropped). US state full name → 2-letter code. Landmark phrases (`near|opp|behind|beside …`) are moved to a separate field. `nums` = numeric tokens in order. `postcode` = the first 5–6 digit token.
- **Learned transliteration (`translit.py`):** from train matched pairs where the pool name or address is non-Latin and S1 is Latin, align tokens by position when the token counts are equal. Keep the majority Latin token per native token when support ≥ 3 and share ≥ 0.6. The table is fitted on train only and saved as a TSV artifact.

## 6. Blocking (`blocking.py`)
**Method:** per country label, build IDF-weighted sparse key vectors for S1 and the pool, then compute `S1_chunk · poolᵀ` with scipy (chunks of 20k rows, parallel over cores) and keep the top-K per S1 row by score.
- Key families (config `BLOCK_W`): `n:` name_core tokens; `nb:` adjacent name_core token bigrams; `a:` address tokens with numbers and state removed; `ab:` house-number + first street token; `p:` postcode; `nm:` sorted name-core initials + first token (catches typos).
- A key is dropped when its pool document frequency > `BLOCK_MAX_DF` (3000).
- `K = 40`. Tune K and the weights for recall on the dev sample. Report pair recall, entity-full-recall and candidates/entity after every change.
- Output: `cands` parquet, which is also the source of `candidate_pairs.tsv`.

## 7. Features and models
### 7.1 Stage-1 pair features (`features.py`)
- **Name:** rapidfuzz ratio, token_set, token_sort, partial_ratio, Jaro-Winkler on name_norm and name_core; token Jaccard; IDF-weighted token overlap; legal agreement (both empty / equal / differ / one missing); length diff.
- **Address:** ratio, token_set, partial on addr_norm; token Jaccard; nums Jaccard; first-number equal; postcode equal / missing-either; state/city token overlap.
- **Other:** block_score, block_rank, is_s3, name frequency (count of name_core in the pool, log), native-script flag on either side, empty-address flags.
- Computed with rapidfuzz `process.cpdist` (vectorised, multithreaded).

### 7.2 Noise-channel features (`noise.py`) — differentiator #2
Features that say *why* two strings differ, learned from the noise operators seen in train:
- `num_digit_deletion`: one house number equals the other with one digit dropped (3424/424, 1201/120).
- `num_equal`, `num_missing_one_side`.
- `name_diff_generic_only`: the symmetric difference of name_core tokens is entirely made of "generic add-on" words. That vocabulary is learned from train: tokens that are over-represented in pool names relative to their matched S1 names (e.g. services, care, company, group).
- `name_legal_moved`: the legal tokens are equal but in a different position.
- `name_edit1_tokens`: count of token pairs at Damerau distance 1 (typos, "Vlley"/"Valley").
- `state_expanded`: states are equal after mapping.
- `addr_only_match`: address strongly equal while names share no tokens (the trade-name swap case).
- `accent_only_diff`: equal after stripping accents but different before.

### 7.3 Stage-1 model
LightGBM binary (config `LGB_PARAMS`), 5-fold GroupKFold by S1 entity, early stopping on the fold's validation set. Produces OOF `p1` for train and the fold-averaged `p1` for test.

### 7.4 Stage-2 context + triangle features — differentiator #1a
Computed from OOF `p1` only:
- **Group context:** rank of p1 in the S1 group, gap to the group max, count of p1 > 0.5 in the group, group size.
- **Competition:** the candidate's best p1 against any *other* S1, and the gap to it (the one-to-one signal).
- **Triangle support:** for candidate i in group g, `support_i = Σ_{j≠i in g} p1_j · sim(i, j)`, where `sim` = mean(name token_set, address token_set) between the two pool records. Also `max_sim_to_top` = similarity of i to the group's highest-p1 candidate from the *other* source, and `n_supporters` = count of j with p1_j > 0.5 and sim > 0.8. Wrong candidates rarely get cross-source support.
- **Stage-2 model:** LightGBM on stage-1 features + these. Same folds as stage 1, producing OOF `p2`. Keep stage 2 only if OOF F0.5 improves *and* transfer does not drop.

## 8. Decoding (`decide.py`) — differentiator #1b
1. **Calibration:** isotonic regression of p2 against the label, fitted on OOF, then applied to test.
2. **Global one-to-one:** for each pool record, keep it only in the S1 group where its calibrated p is highest. Zero it elsewhere. (Exact, because one-to-one holds in train.)
3. **Expected-F0.5 subset selection per S1:** sort the candidates by p (descending). For k = 0..min(n, 12), estimate
   `E[F_k] = E[1.25·TP_k / (k + 0.25·G)]` with 512 Monte-Carlo draws of independent Bernoulli(p_i), plus `G += m` for expected matches lost in blocking (m = mean missed per entity from the OOF blocking report). k = 0 scores `P(G = 0)`. Choose the argmax k.
4. **Knobs tuned on OOF with the exact scorer:** a probability temperature τ (p ← p^τ) and a minimum-probability floor. Prefer flat optima.
Baseline for comparison: a global threshold plus an empty gate. Keep whichever scores higher on OOF.

## 9. Validation
- **Dev loop (laptop, ≤ 20 min):** a stratified 10% sample of train S1 per country, blocked against the *full* pool of that country. 5-fold GroupKFold inside it.
- **Full run (SageMaker `ml.r5.4xlarge`):** all train S1 for OOF + the test inference.
- Every evaluation prints: all / per country / singletons / matched, blocking recall, candidates/entity. It also prints transfer US→IN and IN→US (train on one country, score the other). A change is rejected if overall improves but transfer drops.
- `experiments/log.csv`: one row per experiment (columns as in CLAUDE.md).

## 10. Testing (`tests/`, pytest)
- The metric reproduces 0.714 on the PS example, and the singleton rules hold (empty/empty = 1, empty/non-empty = 0).
- Normalization examples: `L.L.C.`→`llc`, `R. DE DIEPPE`→`rue de dieppe`, `Texas`→`tx`, `Pvt. X Ltd.`→core `x`, legal {private, limited}.
- `num_digit_deletion("3424", "424") == 1`.
- Decode toy case: two S1s competing for one pool id → assigned to the higher p; an all-low-p group → empty.
- Writer: headers, order, empty second column, no duplicates. Then the official validator on a toy output.

## 11. Operations (shift relay)
- Private GitHub repo; the root is `student_resource/`. `dataset/`, `output/`, `cache/` are gitignored.
- `HANDOFF.md` is updated at every shift end: done / next / best OOF / blocking recall / public LB / uploads used today / running jobs / instance state.
- Only the shift driver uploads. The validator runs first. Git tag `sub-dN-MM` per upload.
- Long runs start at shift end. The instance is stopped when idle.

## 12. Deliverables
- `output/matching_results.tsv`, `output/candidate_pairs.tsv`
- `code/business_entity_resolution/{src/, README.md, requirements.txt}`
- `Documentation_template.md` filled in: core ≤ 2 pages; Appendix = dictionaries, learned-table description, experiment log, error analysis
- `scripts/make_zip.py` produces `<team>_submission.zip` with the exact structure from the PS

## 13. Out of scope (v1)
Embedding models, a GPU, LLMs, France pseudo-labelling (possible day-2 stretch, only if the transfer proxy improves).
