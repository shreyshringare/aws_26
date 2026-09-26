# Research note — Joint S2↔S3 cluster consistency at decision time (IDEAS.md B.6, 🧪)

Scope: research only, no code changed. Answers "how would we actually build this, is it worth it,
and in what order" for the item flagged in `docs/IDEAS.md` B.6: *"Joint S2/S3 cluster decision (pick
consistent clusters, not just top-k) — uses S2↔S3 similarity at decision time — stage-2 triangle
features aren't enough."*

## 1. Why triangle support (already built) isn't the same thing

`src/context.py::triangle_features` already computes, per candidate `i` in an S1 group, support from
the group's top-`TRIANGLE_TOPM=5` other candidates by stage-1 OOF score `p1`: a weighted-similarity
sum (`f2_support`), a supporter count (`f2_n_supporters`), and — the closest existing thing to this
idea — `f2_sim_top_other_src`, i-'s similarity to its best *opposite-source* supporter. Three real gaps
remain between that and "joint cluster consistency at decision time":

1. **It's a feature, not a decision.** All of it feeds stage-2's LightGBM as inputs to a per-pair
   probability `p2`. The *decoder* (`decide.select_expected_f05`) then treats every candidate's
   inclusion as an **independent** Bernoulli draw (`hits = rng.random(...) < pb`) when simulating
   expected F0.5. Two candidates that mutually corroborate each other (an S2 record and an S3 record
   that resemble each other AND both resemble S1) are, in reality, correlated — if one is a true match
   the other is more likely to be too — but the current Monte Carlo throws that correlation away at
   exactly the step that's supposed to reason about *sets*, not just scores.
2. **Supporters aren't source-restricted.** `f2_support`/`f2_n_supporters` pool same-source and
   opposite-source supporters together; only the single `f2_sim_top_other_src` scalar isolates
   cross-source evidence, and it only reports the *best* one, not group/cluster structure (size,
   whether a consistent cluster spans both sources, how many corroborating members exist).
3. **No explicit grouping survives to decode time.** There's no notion of "these 2-3 candidates are
   plausibly the same real match cluster" anywhere the decoder can see — just per-candidate scalars.

## 2. Why this should actually help (evidence, not vibes)

- `docs/EDA.md` §B: mean 3.46 true matches per S1 entity, 71% of entities have 2-5, spread across both
  S2 and S3 (both sources sit at ~25-27% unmatched-pool-record rate, i.e. neither source is where all
  the "real" matches concentrate) — most non-singleton S1 entities *do* have a genuine multi-candidate,
  cross-source cluster to exploit, not just a single best match.
- `docs/IDEAS.md` A.2 stage 4 rationale, already validated by triangle features improving E05 over E04:
  "true matches resemble each other." The premise this idea extends is already measured to be true, not
  speculative.
- `docs/ERRORS.md` has concrete cases this would plausibly fix:
  - **`threshold` class**: `Costa Pediatrics LLC` vs `Fluxvantageveo` (p=0.595, near-identical address,
    unrelated name) — genuinely ambiguous *alone*. If the correct match for that S1 entity is a
    different, well-corroborated cross-source pair, joint reasoning should suppress this one; if this
    IS the only candidate with any address support, joint reasoning changes nothing (correctly).
  - **`number_conflict` FP/FN inconsistency**: `420 Main Street` vs `433 Main St` kept (p=0.869, wrong)
    while `100 N Norene Circle` vs `113 North Norene Circle` dropped (p=0.650, wrong the other way) —
    the model has no consistent standalone signal here; whether a *different* candidate in the same S1
    group has strong cross-source corroboration is exactly the kind of side information that could break
    both ties correctly instead of inconsistently.
  - **`other`/`name_collision` near-duplicates** (accent-only, trade-name-swap-at-shared-address): these
    are precision problems on individual pairs; corroboration structure is orthogonal information the
    per-pair features can't see.

## 3. Design options, cheap → expensive

### Option A — Cluster-level features only (extends context.py, no decoder change)

Per S1 group, build a similarity graph over its (OOF-scored) candidates: edge `(i, j)` exists if
`sim(i, j) ≥ threshold` using the same name+address blended `token_set_ratio` metric
`triangle_features` already computes, **restricted to cross-source pairs** (`is_s3[i] != is_s3[j]`) —
reuse the "sup" threshold precedent (`sim > 0.8`) rather than inventing a new one. Connected components
of that graph (union-find, cheap: candidate counts per group are ~40 after blocking) become clusters.
Add, per candidate, its cluster's: `f2_comp_size`, `f2_comp_n_sources` (1 or 2 — "does this cluster
have cross-source corroboration at all"), `f2_comp_mean_p`, `f2_comp_max_other_p` (best *other* member's
p, i.e. this candidate's own corroboration strength distinct from `f2_sim_top_other_src`'s pure
similarity). Feed into stage-2 LightGBM alongside existing `f2_*` features.

Cost: negligible (same order as existing triangle_features). Leakage: none beyond what
`triangle_features` already accepts (built from OOF `p1`, same discipline). Risk: may be redundant with
`f2_sim_top_other_src`/`f2_support` — a real possibility, testable directly via feature importance dump
(`ER_DUMP_IMPORTANCE=1`, already wired in `run_pipeline.py`) after one experiment.

### Option B — Decision-time correlated selection (decide.py change)

Replace `select_expected_f05`'s independent-Bernoulli Monte Carlo with a **two-level draw** that
preserves each candidate's calibrated marginal probability but induces positive correlation among
cluster-mates (built from the same graph as Option A, computed once per group, reused across all
`n_draws` simulations — no extra asymptotic cost):

```
for each S1 group:
    components = connected_components(cross_source_sim_graph(candidates, threshold=0.8))
    for each component c with members {i1..ik} and probabilities {p_i1..p_ik}:
        q_c = max(p_i1..p_ik)                      # cluster "is real" latent rate
        a_i = min(1.0, p_i / q_c)  for each i in c  # conditional rate given cluster is real

for each Monte Carlo draw:
    for each component c:
        z_c ~ Bernoulli(q_c)                        # one shared draw per cluster
        for each member i in c:
            hit_i = z_c AND (Bernoulli(a_i))         # correlated: all members gated by the same z_c
```

Singleton components (`k=1`, no cross-source corroboration — still the common case, especially for the
5.6% singleton entities) reduce exactly to `q_c = p_i`, `a_i = 1`, `hit_i = Bernoulli(p_i)` — **identical
to current behavior**. This is the key safety property: the mechanism only changes anything where a
corroborated cluster actually exists, so it can't make singleton-heavy or weakly-blocked entities worse
by construction, only entities that already have multi-candidate cross-source structure.

Every downstream step (`tp = cumsum(hits)`, expected F0.5 over `kk` values, the `k=0` comparison) is
unchanged — only how `hits` gets sampled changes. Marginal probabilities are provably preserved
(`E[hit_i] = q_c * a_i = q_c * min(1, p_i/q_c) = p_i` when `q_c ≥ p_i`, true by construction since
`q_c = max` over the component), so this doesn't silently bias the empty-gate math (`e0`) either — it
still uses the same per-candidate `pb` for the independent-equivalent parts.

Cost: same asymptotic complexity as today's decoder (`O(n_draws × kmax)` per group); the component
bookkeeping is `O(kmax)` per group, computed once, not per draw. Should be implementable without a
measurable slowdown vs. today's `select_expected_f05`.

### Option C — Full combinatorial joint selection (not recommended to build first)

Instead of top-k by marginal probability, exactly (or near-exactly, via component-level dynamic
programming) search over which *components* to include and how many members of each, maximizing exact
expected F0.5 rather than a Monte Carlo approximation. Feasible in principle since `kmax=12` bounds
group size for this decision, but adds real implementation complexity (need to handle partial-component
selection, i.e. taking 1 of 2 members in a cluster) for a benefit that Option B likely captures most of.
Only worth revisiting if Option B measurably helps but under-fits some structure (e.g. large clusters
where "take all vs. take best-k" matters).

## 4. Recommended build order (one-change-per-experiment discipline, per CLAUDE.md)

1. **Exp J1 — Option A only.** Add `f2_comp_*` features to `context.py`, retrain stage 2, compare
   against the current best (E06) on OOF F0.5 (all/US/India/singletons/matched) and transfer. Cheap,
   low-risk, tells us whether cluster structure carries signal *beyond* what `f2_sim_top_other_src`
   already gives — check via `ER_DUMP_IMPORTANCE` feature ranking, not just the aggregate F0.5 delta.
2. **Exp J2 — Option B, gated on J1.** Implement the correlated-draw decoder as a new decode mode (add
   it as a `decide.tune` grid dimension, e.g. `joint: True/False`, exactly how `one_to_one` is already
   gridded), so it's only kept if it beats the current best decoder *on OOF* — same discipline as every
   other decoder choice in this codebase. Must also pass the transfer-robustness bar CLAUDE.md sets
   (don't accept if US↔India transfer regresses beyond the ~0.002 tolerance already precedented for
   E05's stage-2 adoption).
3. Do not build Option C unless J2 lands and a specific under-fit pattern (large corroborated clusters
   handled worse than small ones) shows up in a follow-up error-taxonomy pass (`docs/ERRORS.md`-style).

## 5. Risks / things that would make this a bad idea in practice

- **Redundancy with existing stage-2 features.** `f2_support`/`f2_n_supporters`/`f2_sim_top_other_src`
  already give the model *some* cluster-awareness; Option A might add little. Must check feature
  importance, not just assume novelty = value.
- **Double-correction risk in Option B.** If stage-2's `p2` already partially "bakes in" corroboration
  (because it was trained with triangle features), applying correlated sampling on top of an already
  corroboration-aware `p2` could over-cluster (push k up too aggressively for weakly-corroborated
  groups). This is exactly why J2 must be tuned/compared on OOF like every other decoder choice, not
  assumed to help.
- **Threshold reuse (`sim > 0.8`) is inherited, not re-validated** for this specific use — worth a small
  grid over the component-similarity threshold in the same experiment rather than trusting the borrowed
  constant.
- **No new leakage vector** as long as component structure is built from OOF `p1`/`p2` only (same rule
  `triangle_features` already follows) — but worth stating explicitly in the experiment's `change` note
  in `experiments/log.csv` since it's easy to accidentally use in-fold scores when refactoring.
- **France risk is neutral-to-positive, not a new exposure.** The mechanism never reads country and
  degrades to current behavior for uncorroborated (singleton-component) entities, so it shouldn't
  uniquely hurt France — if anything, France's 15% share and unseen vocabulary make per-pair scores
  noisier, so a mechanism that leans on *structural* (cross-record) corroboration rather than purely
  lexical similarity could be relatively more valuable there. This should be checked directly in J2's
  transfer report, not assumed.

## 6. Bottom line

Worth building, in the order above. Option A is nearly free and should run regardless of whether B ever
lands — it's a direct, low-risk test of whether cluster structure adds signal on top of what triangle
support already captures. Option B is the actual "decision-time" mechanism the idea calls for, has a
clean degrade-to-current-behavior property for the common singleton/uncorroborated case, and reuses the
existing OOF-tuning discipline (`decide.tune`) to gate acceptance — the main real risk is that stage-2
already captures most of the value and B/A land as a small or null delta, which is a legitimate
experimental outcome, not a reason to skip measuring it given how novel `docs/IDEAS.md` itself flags
this as being.
