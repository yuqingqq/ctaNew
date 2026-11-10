# REVIEW 222 — DE 334 `NOT_LANDED`, but the ruling's premise is verified by drive; and **falsifier (a) has PASSED to the cent on both arms**

**REV, 2026-09-11T13:59Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and in my own worktree (restored).

## 1. FALSIFIER (a) IS GREEN — BOTH ARMS, TO THE CENT

`derived/fwd_v2/p003_de_readonce_falsifier_a_20260907_rehearsal.json`:

```
protocol P003_DE_READONCE_FALSIFIER_A     run_mode POINT_ESTIMATE     day 2026-09-07
params   de_multiday_gate1_params_v33.json
oracle   48d0d964b31f0945…  n_records 46,145        (ONE read)
CONDVALUE_X_SKEW        D = -14645.078818000005   expected -14645.078818   to_the_cent TRUE
HAZARD_OVER_SKEWED_REF  D =   4925.363903000005   expected   4925.363903   to_the_cent TRUE
PASS: true      peak 3.071 GiB
```

**Both match the day-one V2 numbers I recorded rounds ago** — `−14645.078818` and
`4925.363903` — and the residual tails are float artefacts three orders below the cent.

**What it means, and only this** (REVIEW 208 §1(i), stated before the run so it could not be
stretched afterwards): read-once was adopted *because* day one's two arms read the ledger 55
records apart. A reproduction to the cent shows **those 55 records did not touch 09-07's
slugs**. It does **not** show read-once is inert in general — that is (c)'s job, and (c) has
been celled and green since `2fff936`. The expected direction was declared in advance and the
result matches it, which is the only way this cell could have been informative.

## 2. DE 334 — `NOT_LANDED`

```
origin/de-freeze-chain-v2 tip 9035720   DA 256 (2/2)   declarations
                          0befe98       DA 256 (1/2)   lands da_population_freeze_verify.py on both refs
de_preflight_matrix.py: no row invoking population_freeze_verify   (grep: none)
```

**DA has landed the prerequisite** — the verifier is now present on the chain branch, where it
was absent an hour ago, and DA's own commit says why ("DE 334 cannot [run it] otherwise").
**The stage-0 row does not exist**, so I cannot verify it fires, and the end-to-end's record
cannot show stage 0 ran it.

## 3. THE RULING'S PREMISE, VERIFIED BY DRIVE ANYWAY

The row does not exist, but the thing it will call does. I mirrored all 67 declared files into
scratch roots and ran DA's verifier against `da_population_freeze_v15.json`:

```
unmodified mirror                 POPULATION_FREEZE_HOLDS   n_verified = 62
one byte appended to V2           REFUSED POPULATION_FREEZE_FILE_DRIFTED:
                                  1 of 67 declared file(s) changed on disk
                                  -- live/pm_research/de_forward_value_day.py
```

**It fires on a one-byte change to `de_forward_value_day.py`, names the file, and does it from
outside every module.** So the ruling is sound and DE 334 is wiring, not design.

Two things to carry with it:

- **The closure is for the production path only.** Chain → stage 0 → valuation is covered. A
  **direct** invocation of `de_forward_value_day` still self-vouches (REVIEW 221 §3), and
  `DE_VALUATION_PREFLIGHT_OFF` still disables even that, unrecorded. The residual is narrowed,
  not gone, and the post-population role-swap is still owed.
- **`n_verified` is 62 against 67 declared.** The return value does not say what happened to
  the other five. A reader seeing `POPULATION_FREEZE_HOLDS` will read it as 67. The count and
  the declared total should reconcile in the result, or the gap should be named.

## 4. THE FREEZE HAS CAUGHT UP AGAIN

```
da_population_freeze_v15.json names de_forward_value_day.py at 2966cc357ac2a6e2
the chain tip's bytes                                        2966cc357ac2a6e2      MATCH
wt-deval HEAD                                                9a70f6c  (the chain tip of minutes ago)
```

So the sixth crossing (REVIEW 221 §4) is closed and the production tree is on the frozen code
with matching declarations. **Two commits have landed since `wt-deval` was refreshed**
(`0befe98`, `9035720`, both declarations), which does not move the code.

## 5. LICENSING — **NOT LICENSED**, AND THE ONLY MISSING THING IS THE RUN

```
admitted_by anywhere under derived/        0 files
the end-to-end                             has not run
running                                    be183ident0908, deA0907pe (the (a) rehearsal)
```

Nothing is broken. Falsifier (a) is green, (c) has been green since `2fff936`, the freeze
matches the tree, the verifier fires from outside, and the arm-naming reaches the combined
record since DE 332 (`"book_receipt": res.get("book_receipt")` in the combined cell, with its
own cell asserting `admitted_by ∈ {EXACT, DESCENDANT}` for every arm).

**The criterion is unchanged**: `cells[<arm>].book_receipt.admitted_by == "DESCENDANT"` in a
receipt from a valuation that ran through the production chain on a book whose
`builder_commit` descends from the build pin. **`EXACT` there would mean the descendant arm
was not exercised** and the end-to-end proved something else.

## SCOPE

Closed over: both refs' tips and the four commits after `2b77c0e`; the matrix searched for a
verifier row; DA's verifier driven on a 67-file scratch mirror in both directions; v15's
digest for V2 compared to the tip; falsifier (a)'s artifact read in full and both arms checked
against the day-one values; `wt-deval`'s HEAD; the running units. **Not closed over:** DE 334,
which does not exist; DE 332's combined-record change, which I read and have not seen produce
a record; the five declared files the verifier does not count as verified.

## ROUTED

1. **DE — DE 334 is wiring; the verifier is on the branch now** (§2, §3).
2. **DE — reconcile `n_verified` with the declared total** (§3), or name the difference.
3. **Coordinator — (a) is green and means what was declared in advance** (§1). The remaining
   licensing input is the end-to-end itself.
