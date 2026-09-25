# Improvement Plan — from teammate EDA cross-validation

Source: independent EDA by a teammate (full dataset, deterministic blocking benchmark + normalization +
ground-truth topology). Numbers cross-validated against our own `docs/EDA.md` and found consistent
(singleton rate 5.585% vs our 5.6%, mean matches/S1 3.461 vs our ~3.5, max 11, France ~15% of test/absent
from train — all match). Treated as real evidence, not just a second opinion. Full report content is in
chat history (25 Sep, ~14:50 IST); raw artifacts (`hard_negatives.csv`, `positive_pairs.tsv.gz`,
`false_collision_examples.csv`, `singleton_candidate_load_all.csv.gz`, `analysis/` scripts) are NOT yet
in this repo — see item 5.

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

## 4. Resolve the normalization-as-identity-key disagreement (needs discussion, not a silent change)

**Finding:** Teammate recommends NFKC + casefold + punctuation/whitespace-only for blocking *identity*
keys, treating abbreviation expansion and token-sorting as scoring features only — because abbreviation
normalization grows their collision groups 226→429 (measured on their 35K-row sample). Our `blocking.py`
does the opposite: abbreviation-normalized `name_core`/`legal`-split tokens are blocking key families
(`n`, `nb`, `nm`, `np`, `nc`, `n2`) directly.

**This is a real architectural difference, not obviously wrong on our side** — our blocking weights keys
by IDF rarity and drops keys with document frequency > `BLOCK_MAX_DF=2000` (`config.py`), which is a
different mitigation for the same collision-inflation problem they're describing for naive exact-match
blocking. But it's worth confirming this actually holds at our current recall level rather than assuming.

**Action:** Don't change normalize.py or blocking key families based on this alone. Instead:
1. Ask the teammate exactly what blocking method they measured the 226→429 collision-group growth on
   (naive exact-match vs. our IDF-weighted union) — if it's the former, it may not transfer.
2. If there's time after the B04-B06 blocking work, a cheap check: does removing abbreviation-normalized
   families from the blocking union change recall/cands-per-entity at same config? That's answerable in
   one experiment run.

**Owner/next step:** Sync with the teammate directly before touching `normalize.py` or `config.BLOCK_W`.

---

## 5. Get the raw artifacts into the shared bucket

Everything above except item 1 is blocked on actually having the teammate's generated files, not just
the summary numbers. `aws-hack-sagemaker` (the shared S3 bucket) already exists — ask them to
`aws s3 sync analysis/ s3://aws-hack-sagemaker/eda-teammate/` (or similar) once their bucket-policy access
is confirmed working.

---

## Explicitly not adopting

- Any of their deterministic blocking strategies as a replacement for our sparse IDF union — already
  shown to be far less efficient at comparable recall (item 1).
- Reworking normalize.py identity keys immediately (item 4) — discussion first.
