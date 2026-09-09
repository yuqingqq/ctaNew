# REVIEW 107 — DE 155's seven scoring defects (`c501824`, `ecc7116`)

**REV, 2026-09-09T05:18Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. Everything ran in a scratch clone at the real commits.

**VERDICTS.** **(1) CLEARED — the look-ahead repair is correct and I drove all three cases
myself.** (2) CLEARED, driven. (3) **PARTIAL**: the feature side is closed, the score side
is not, and I produced a non-finite score from a finite feature vector. (4) CLEARED, and
the two numbers are exactly self-consistent. (6) CLEARED and genuinely consistent with
first-crossing. (7) CLEARED, driven. **(5) the boundary claim is REFUTED — the
per-generation input count IS derivable from something already emitted, and closing it
needs nothing from BE.**

**ONE BLOCKING FACT ABOVE ALL OF THEM, and it is not in the seven: `ecc7116` moves FOUR
cascade modules, and params v20 pins their pre-fix digests. The IN-RUN battery refuses
`BE_CASCADE_DIFFERS`, so NO DAY CAN RUN on this tree — and it cannot be fixed by editing
the params, because design v28 pins the params file by digest. It needs params v21 +
design v29 as a PAIR. Nothing corrected can be built or quoted until that lands.**

---

## (1) THE LOOK-AHEAD — CLEARED, driven three ways on the engine

I built my own fixtures and drove `harmful_stateful_policy.replay_policy` directly:

| case | old stream (one event at `t0` carrying the MAX) | per-row stream |
|---|---|---|
| **(a)** late max, generation open to t=10 | **1 cancel at `t_request` 0.0** — the generation's START, on information from t=6.0 | **1 cancel at 6.0**, the late row's own time |
| **(b)** first row already crosses | — | **1 cancel at 0.0** — the repair does not merely delay everything |
| **(c)** generation ENDS at 5.0, crossing row at 6.0 | **1 cancel at 0.0** | **0 cancels** |

and two of my own: two crossings → **one** cancel at the **first** (3.0, not the maximum at
6.0); never crossing → 0. **(c) is the case that matters most — a cancellation that existed
only because a score was moved backwards in time.**

The repair is in the right place: the stream is built per row, and the engine already
issued one cancel per generation at the first crossing and asserted
`one_cancel_per_generation` — it had simply never been handed more than one event per
generation. Unscored generations still emit their `t0` row so `_head_scorer` refuses them
by name, which keeps that control reachable; a row earlier than its generation is an
**exclusion with a status** (`ROW_BEFORE_GENERATION_START`), never clamped forward.

**THE 09-04 CLOCK CLAIM: NOT VERIFIED, and I say so.** Confirming that `t_start` and
`gen_t0` are one clock, and that a generation's first row starts exactly at its `t0`, needs
the feature pass over 09-04's real rows — a heavy run I was told not to make. What I can
say: **the repair does not depend on it.** A row preceding its generation is counted and
excluded rather than assumed away, so if the claim were false the run would say so in
`ROW_BEFORE_GENERATION_START` instead of silently mis-timing. The claim is load-bearing for
the *interpretation* (that the defect collapsed a generation onto its opening instant
rather than shifting it arbitrarily), not for the correctness of the fix. **And DE is right
that the old fixture could never have caught this**: it paired generations at t0 100/400
with rows at t_start −6/−3 — times belonging to no generation — and the old code discarded
row times, so the incoherence was invisible. The new fixture's rows lie inside their
generations.

**THE CALIBRATION CONSEQUENCE IS RECORDED, AND WELL.** `day_run.scoring_timing` on every
receipt carries the rule, what it was, the measured incidence, `theta_refitted: false`, and
the two consequences: a first-crossing score is at or below the generation maximum, so at
an unchanged theta **fewer** generations cross and the count moves for a reason that is not
a market reason; and because the null is matched on the arm's action count, **the control
moves too** — this is not an arm-only shift. That is the right disclosure and the right
place for it.

**FINDING (1a), non-blocking: the repaired function's own docstring still states the defect
as spec.** `generation_scores` still opens *"(slug, side, t0) -> one score per GENERATION"*
and argues *"The generation is the unit, and its score is the MAX of its rows' … a mean, or
a first-row score compared against that theta … is not the policy's statistic."* The map is
now keyed per ROW at the row's own time, and a first-crossing score is exactly what the
function now feeds the policy. DE replaced the battery **cell** that asserted the defect as
spec (and says so) and left the **docstring** that says the same thing. The threshold
argument inside it is real and unresolved — which is why it must be restated as *what theta
was fitted over versus what this function now emits*, not left as the function's contract.

## (2) THE UNBOUND NORMALIZER — CLEARED, driven

`linear_{coin}.json` is now in the head's file tuple, the params' `model_digests` set (five
files, not four) and checked against the manifest. Driven on the real fit file in my
scratch mirror: **green control loads; a `1e-9` perturbation of a single number refuses
`LGBM_NORMALISER_DIGEST_DIFFERS`**; an absent manifest entry refuses
`LGBM_NORMALISER_NOT_IN_MANIFEST`. DE also folded the manifest into the model cache key,
which closes the same hole one level down: bytes loaded under one manifest can no longer be
served to a caller whose manifest declared different ones.

## (3) NON-FINITE INPUTS — PARTIAL. The user's defect is closed; the item as stated is not

The **feature** side is closed by name: `compose_head_inputs` refuses `NON_FINITE_FEATURE`
before the vector reaches the incumbent's clamp, so the specific failure the user named —
an infinite feature becoming a plausible saturated probability — cannot happen. Thresholds
are checked for finiteness too.

**But scores are not checked, and non-finiteness does not require a non-finite feature.**
Driven:

```
score_incumbent_condvalue(incumbent, [1e308] * n_features) = nan     (isfinite False)
```

Every input finite; the product overflows. **A NaN score is then compared against theta, and
`nan >= theta` is False — so an unrepresentable score silently becomes "do not cancel"**
rather than a refusal or a counted status. This is reachable through exactly the door (2)
was about: a normaliser with a near-zero `norm_sd` produces enormous finite z-scores. The
fix is one predicate where the score is computed in `generation_scores`.

## (4) THE INFLATED EXCLUSION COUNT — CLEARED, and the two numbers check out exactly

`NO_ROWS_KEPT` was counted inside each chunk against the **whole day's** reference and then
summed, so every generation outside a chunk was counted once per chunk. It is now computed
**once, from the union**: a generation is uncovered only if no chunk covered it, and the
per-chunk count is explicitly denied (`count_missing=False`).

**The arithmetic is self-consistent, and I checked it rather than accepting it.** If the
day's reference holds `N` generations, each covered by exactly one of 42 chunks, and `U` are
covered by none, the defective sum is `41·N + U`. With `U = 15,735`:

```
(12,853,409 − 15,735) / 41 = 313,114 exactly      41 × 313,114 + 15,735 = 12,853,409
```

**So the reported 12,853,409 and the corrected 15,735 are exactly what the defect predicts
for a 09-03 reference of 313,114 generations — an integer, with no remainder.** That is a
checkable prediction: BE or DE should confirm 09-03's reference is 313,114 generations. I
did not load the book to confirm it.

## (5) NOT FIXED — AND THE BOUNDARY CLAIM IS REFUTED

DE's claim: *"no structure carries a per-generation input count today (`blocks['drops']` is
aggregate, by reason, for the whole coin)"*, so closing it needs a new return value from
`phase2_arms._feature_pass`, which is not DE's surface.

**It is already emitted, and `generation_scores` already receives it.** `build_tape_index`
builds `split_of` over `PA.tape_index(sp)` for every split — **the tape rows as indexed,
before the feature pass drops anything** — and its key is the tuple `generation_scores`
itself looks up: `(slug, side, gen, t_start)`. So

```
rows_in[(slug, side, gen)] = #{ k in split_of : k[0], k[1], k[2] == slug, side, gen }
```

is a per-generation **input** count, available today, inside the same function, from a
parameter it is already passed. `_rows_expected` can fire without touching BE's surface.

**One caveat that decides WHERE, not WHETHER:** on a chunked run `split_of` covers the whole
day while a chunk's `kept` covers only its slice, so the comparison must be made on the
**merged** per-row scores after the chunk union — not inside a chunk, where every partially
covered generation would look partial. DE's own `_rows_expected` docstring already puts the
comparison at the run, "which has both", so this is a smaller step than the filing implies.

**This matters exactly as much as DE says it does**: a maximum is insensitive to a missing
row; a first crossing is not — a dropped early row moves the cancel later or removes it. It
should not stay `NOT_COMPUTABLE`.

## (6) THE MERGE — CLEARED, and genuinely consistent with first-crossing

`.update()` let a later chunk overwrite an earlier chunk's key. The merge now keeps, per
key, the higher score — **and the key is `(slug, side, t_start)`, one ROW**. That is the
distinction the coordinator asked about: **the maximum is taken within a single instant,
never across a generation's instants**, and rows of one generation that fall in different
chunks all survive as distinct keys, so the first crossing is still found across chunk
boundaries. It is not a re-introduction of the per-generation aggregate, and it is more than
deduplication. Split labels now combine to `MIXED` instead of taking whichever chunk ran
last, which is what the label means.

**Routed, non-blocking:** a collision means **two chunks scored the same row** — which, if
the chunks partition the rows, should be impossible, and if it happens means their feature
passes disagreed. "Keep the higher" resolves that silently. Count it (`n_rows_scored_twice`,
`max_abs_delta`) so a partition bug cannot hide behind a max.

## (7) REPEATED MODEL LOADS — CLEARED, driven

`model_cache_stats()` after 41 rounds of loading all three artefacts: **3 misses, 120
hits**. The cache key is the fit files' bytes **and** the manifest, so changed bytes are a
different entry rather than a stale hit.

## DE 154's PARALLEL NULL — I CANNOT VERIFY IT: IT IS NOT IN THIS TREE

There is no parallel null in `de_multiday_gate1_runner.py` or `de_phase4_diag_runner.py` at
`ecc7116` or at the tip: no `multiprocessing`, no `ProcessPoolExecutor`, no worker fan-out
(the three modules in the package that import `multiprocessing` are
`da_duplicate_identity_scan`, `collect_pm` and `cross_window_correlation`, none of them the
null). No commit in the repository's history mentions DE 154. **So I have verified neither
the element-by-element reproduction nor the can-fail control — there is no artifact to
verify.** Also: **`EXPECTED_CHECKS` here is 402, not the 403 quoted** (phase4 is 219 =
215 run + 4 conditional, which I did run: `selftest OK -- 215 checks`).

**THE SLICE CLAIM IS TRUE, AND I MEASURED IT:**

```
systemctl --user show research.slice -p CPUQuotaPerSecUSec  ->  CPUQuotaPerSecUSec=2s
                                     -p MemoryMax           ->  15032385536  (14 GiB)
nproc = 16
```

**`research.slice` is capped at 2 s of CPU per second — 200 %, two cores — on a 16-core
box.** A unit inside the slice cannot exceed it whatever its own `CPUQuota`, so a parallel
null run there is limited to about two cores and a claimed 12.7× would land near 2×, with
nothing in the run reporting that it had been capped. **Raise the slice, or measure the
speed-up inside it and quote that number.** (I did not verify the ~450 MB resident book;
that needs a real load.)

## THE BLOCKING FACT: THESE FIXES CANNOT RUN A DAY

`ecc7116` changes four modules that params v20 pins in `be_cascade`:
`de_head_scoring.py`, `de_phase4_diag_runner.py`, `de_score_stream.py`,
`harmful_stateful_policy.py`. Driven:

```
selftest(quiet=True, offline=True)  ->  RunnerRefused: REFUSED BE_CASCADE_DIFFERS:
                                        4 of 10 cited cascade modules do not match
```

**That is the IN-RUN battery** (stage `S0b_battery`), so a day run on this tree refuses in
its first minutes. And it cannot be repaired by re-pointing the params: I tried it in
scratch and the battery then failed on the **params↔design pair** — design v28 pins
`de_multiday_gate1_params_v20.json` by path AND digest, so any edit to the params breaks the
pair. **Moving those four pins requires params v21 and design v29 landed together.** Until
then, DE's own "battery green" cannot be reproduced on the landed tree, no corrected book
can be built, and no corrected number exists to quote. (Noted in passing: v20's
`settlement_endpoint` is `null`, so rule 11 still refuses every day that is not one of the
four design days.)

## ROUTED

1. **Coordinator / DE — params v21 + design v29 as a pair, re-pointing the four moved
   cascade pins.** Everything downstream waits on this, including BE's rebuild.
2. **DE — (3)'s score-side predicate** (a NaN score currently reads as "do not cancel");
   **(5)** from `split_of`, compared after the chunk merge; **(1a)** the docstring;
   **(6)** count the collisions.
3. **BE / DE — confirm 09-03's reference is 313,114 generations**, the integer (4)'s two
   numbers imply.
4. **Whoever owns the parallel null — the slice is the wall at 200 %.** Quote the speed-up
   measured inside `research.slice`, or raise the slice first and say so.
