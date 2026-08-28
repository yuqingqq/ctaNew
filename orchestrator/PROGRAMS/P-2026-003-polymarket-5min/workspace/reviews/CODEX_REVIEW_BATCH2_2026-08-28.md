# Codex review — completed fix-batch 2 (`4a3d457`)

**Exact reviewed batch tip:** `4a3d457386560377ac9f0e119f4b159aeff60ae4`

**Scope:** `REQUEST_BATCH2_2026-08-28.md`, including the day-bar/O1 seam,
Iteration 011 non-fit machinery, Phase 2B amended draft, FP1, parity, and the
two self-reported follow-through items. No model was fit and no result was
scored during this review.

## Filing verdicts

- **DAY-BAR V2: HOLD RELEASED.** The repaired bar may govern coin-days from
  **2026-08-29**. Recompute the 08-29 verdict under the released code; this
  review does not pre-judge whether that day passes.
- **ITERATION 011: HOLD MAINTAINED. DO NOT FIT OR SCORE.** The real runner still
  crashes, and the wired evaluator does not implement the frozen economics,
  nulls, action weighting, or first-crossing rule.
- **O1: THE PRIOR DB2 ADVERSE FINDING IS CLEARED.** The unchanged committed v4
  producer now passes through the day-bar consumer at the real seam. This
  re-arms the **2026-08-30T00:00:00Z** boundary under the existing runbook; the
  v3_1 hold remains until that boundary as requested.
- **PHASE 2B DRAFT: NOT FIT FOR USER FREEZE.** Its Chainlink-source correction
  is valid, but its full-window path-dependent estimand contradicts the
  repository's passed settlement foundation gate.
- FP1 and parity are materially improved partial builds, but the executed
  counterexamples below show that neither boundary is integration-ready.

## 1. Day-bar v2 — hold released

DB1 and DB2 are closed at the actual consumers, not only in helper tests.

### DB1: per-coin P1/P2/P3 now govern

I re-ran the previous full `verify_day()` counterexample with one complete BTC
coin-day and 200 valid 20-second gaps (4,000 seconds lost). The result is now:

```text
P1_pass                  = false
per_coin.all_pass        = false
per_coin.day_quality     = false
per_coin.accrual         = false
whole_day.all_pass       = false
per_coin governing       = [entirely_post_freeze, complete_tape,
                             P1_bar, P2_bar, P3_bar]
```

The per-coin artifact carries its own `day_bar_v2`, verdict split, and accrual
flag. A failed bar can no longer publish a passing coin-day or accrue its clock.

### DB2: the real producer row is consumed

`da_o1_daybar_seam_test.py` imports and drives the committed v4 collector with
the fake socket, reads the row the producer actually appends, and feeds that
file to `day_bar_v2`. All **7/7** checks pass. The consumer:

- uses a finite, ordered producer-supplied `gap_end_ns` and charges the emitted
  duration;
- counts producer ends separately from synthesized ends;
- synthesizes the scope end only where an open-at-exit event truly has no end;
- still refuses a producer end at or before its start.

The day-bar selftest is green at **68 checks**. The prior all-pass callable,
empty-ledger observation requirement, structural-row refusals, exact rolling
P3 maximum, dual report, regime boundary, and freeze epoch remain green. I
found no regression that warrants keeping the day-bar hold.

Therefore **DAY-BAR V2 HOLD RELEASED**.

## 2. Iteration 011 — hold maintained

The Q2 both-sides status correction is real, and the isolated core and runner
selftests are green (**81** and **56** emitted PASS lines). Those suites do not
exercise the actual runner-to-receipt semantics below.

### I11-B1: the real runner still crashes before writing its artifact

At `phase2_iter011_run.py:928-934`, the applied result is assigned to `ap`, but
Q4 is called with `pr`:

```python
ap = apply_arm(fr, Xe, tge)
rep = report_arm(ap, tge)
rep["economics"] = q4_economics(pr, EVAL[coin]["kept"])
```

AST inspection confirms `pr` is loaded but never assigned in `main()`. A real
fit would raise `NameError` before the receipt write. The selftest uses a local
variable named `pr`, which is why it cannot catch this consumer defect.

### I11-B2: Q4 is raw candidate value mislabeled as incumbent increment

`q4_economics(pred, rows)` accepts no incumbent prediction or incumbent
economics. `_q4_cell()` merely sums its `net_cents` and sign-flips its
`increment_by_window`, yet reports the result as `increment vs incumbent`.

Executed known-bad:

```text
candidate economics supplied: +10.0c
incumbent input supplied:      none
reported cell:                 OK, "increment vs incumbent +10.0c"
```

The frozen preregistration requires candidate-minus-incumbent on the identical
action population. A sign-flip of the candidate's raw realized value is not
that increment.

The matched-random null is also absent. In an otherwise complete synthetic
24-cell family, Q1/Q2/Q3 have `p_value=None`; Q4 has only the false incumbent
sign-flip above. This contradicts the frozen requirement that matched random
applies per head and that Q1 and Q4 beat both matched random and incumbent.

### I11-B3: the frozen action-unit rules are not implemented

Amendment A1.5 freezes three rules: each row receives
`1 / rows_in_generation` fit weight, reported head `n` is an action count, and
Q4 uses first-crossing deduplication.

The runner instead:

- supplies unit weights (`[1.0] * len(...)`) to every linear fit and no
  `sample_weight` to the LGBM fits;
- sets `n_actions = len(X)`, and `head_report()` reports prediction-row count;
  an executed 3-row/2-generation probe reported both Q1 `n=3` and
  `n_actions=3` instead of 2;
- chooses the **maximum composed-value row** inside each generation and realizes
  `V_cancel` at that row, not the first threshold crossing.

The first-crossing counterexample is decision-bearing:

```text
same generation, first crossing: score=.6, V=+100c
later row:                       score=.9, V=-100c
declared first-crossing net:             +100c
runner q4_economics net:                 -100c
```

The emitted unit string, `ACTION (first-crossing by max composed value)`, joins
two incompatible definitions; taking a maximum is not first crossing.

### I11-B4: head gates and the receipt guard remain incomplete

- Q3's frozen gate requires each magnitude slope and its CI separately. The
  runner takes the numeric minimum of the two slopes, computes no slope CI, and
  adjudicates one combined value.
- `_one_cell()` says it is pooled over coins but takes the minimum of per-coin
  statistics. This is neither an action-pooled metric nor a frozen aggregation
  rule.
- `assert_receipt_has_all_cells()` validates only the 24 keys and Holm
  denominator. A receipt containing all 24 declared keys mapped to empty
  dictionaries passes the guard in execution. It does not prove that the cells
  contain statistics, statuses, populations, or the declared null evidence.

These are frozen-design violations, not model-performance concerns. Therefore
**ITERATION 011 HOLD MAINTAINED; do not fit or score**.

## 3. O1 boundary re-check

The prior adverse finding was specifically the DB2 producer/consumer mismatch.
The real seam now drives the unchanged `6786a02` producer and consumes its
actual ledger row successfully, including the known-bad ordered-end refusal.
That closes the adverse finding. **O1 is cleared for the 08-30 boundary under
the existing deployment runbook.**

This clearance does not declare the future day healthy; it says the deployed
producer and the governing verifier now share an executable event contract.

## 4. Phase 2B amended draft — do not freeze

The draft correctly withdraws the claim that Binance is the named resolution
source: the market descriptions name Chainlink. It then makes a separate,
unsupported inference that the target is the full five-minute path average:

```text
TWAP_[t0,T] = (A_t + R_[t,T]) / (T - t0)
```

That inference conflicts with the repository's passed deterministic settlement
reconstruction in `EXP_RESULTS_2026-08-20.md`:

| convention | winner agreement |
|---|---:|
| `S60(T) >= S60(t0)` | **99.8% on 1,465 windows** |
| `mean S60[t0,T] >= S60(t0)` | **86.9% on 1,465 windows** |

The foundation result explicitly says the full-range reading is refuted and
the settlement averaging width is 60 seconds, not 300. The amendment cites the
description text but supplies no superseding winner-reproduction audit capable
of overturning that artifact. Freezing A1.3 would therefore point Brier scoring,
the GBM transformation, the `A_t` state, and its falsifiers at the wrong event.

Before user freeze:

1. Reconcile A1.3 with the resolution artifact. Unless a new, independent,
   same-population settlement audit supersedes it, the target remains
   `S60(T) >= S60(t0)` with the exact RTDS topic and boundary reader pinned.
2. Re-state when the terminal 60-second TWAP is part-realized. It cannot require
   a five-minute realized integral at every decision. PM Identity and PM
   microprice are already direct event probabilities and should not be forced
   to carry an explicit price-path integral merely because a structural
   cross-venue transformation needs settlement state.
3. Fully specify the price-to-probability transform: exact moment formula,
   Chainlink topic, sampling/integration convention, volatility observation
   clock and cadence, and `t=T`/missing-reference boundary statuses.
4. Pin a real day-unit inferential test and exact multiplicity family. Brier
   forecast skill does not intrinsically vary by cancellation budget, so
   duplicating one Brier comparison across three policy budgets is not a valid
   family definition without separately defined budget-selected populations.
5. Define the common decision-instant grid/neutral population. “Equal per
   instant” without a declared instant generator can overweight windows with
   more updates.

Also correct the standing text that says a proper Brier forecast **maximises**
expected score; Brier loss is minimized. The superseded Binance paragraphs
should be visibly marked at their original locations so an operative false
claim is not encountered before its later amendment.

**PHASE 2B DRAFT IS NOT FIT FOR USER FREEZE.**

## 5. FP1 re-check

The original direct `value=60000` hole and book-side taxonomy defects are
closed; `[0,1]`, status, coin/outcome, timestamp consistency, and valid-book
cases are materially stronger. Four direct counterexamples still pass:

```text
FairPrice(status=OK, source=0, knowledge=100, freshness=100)  -> ACCEPTED
identity_from_book(..., bid_size=NaN, ask_size=2)             -> OK
FairPrice(window_start=True, otherwise valid)                 -> ACCEPTED
assert_declared_before(1, NaN, challenger)                    -> checked=True
```

An `OK` record constructed by a challenger must enforce the declared maximum
freshness at the record boundary; otherwise challengers bypass the Identity
factory's stale check. Depth and configuration values need finite-real checks
with booleans rejected. Both declaration and comparison stamps must be finite.

The mechanical no-double-count fence is also name-heuristic only and is not
wired to a fitted schema/target identity. Treat FP1 as a useful interface
prototype, not yet a safe common challenger type.

## 6. Replay-parity re-check

The requested hardening is substantial and executable: the matched control now
derives count/side/hour cells from the treated arm, the seeded draw is
order-stable and refuses over-budget requests, the receipt enumerates required
checks, the permanent-hold exposure anchor fires, rate-limit accounting is
load-bearing, training reuse refuses, and the stub suite is **62/62 green**.

The new external boundary is still fail-open in important ways. Executed
well-formed-looking submissions were accepted with `lifecycle_pass=True` when
they contained:

- an undeclared **top-level** field (event-level extras alone are checked);
- a duplicate `CANCEL_EFFECTIVE` for one generation;
- a `CANCEL_EFFECTIVE` timestamp before its request;
- non-integer `gen` and non-string `side` values.

Duplicate effective/suppressed outcomes are collapsed into a dict/set before
the accounting identity is checked, so multiplicity can disappear and pass.
The loader should enforce the complete top-level and event schema, and the
lifecycle checker should enforce exactly one terminal cancel outcome per
request in temporal order.

The declared limitation remains decisive: no independently produced BE
trajectory has crossed this interface. The internal stub battery is green;
external replay integration is not yet cleared.

## 7. Self-reported follow-through items

### Midnight isolation is not total in one override direction

`da_midnight_verify.sh` appends the `fired` marker to `LOG` at lines 27-30,
then performs the both-or-neither override refusal at lines 61-68. With only
`DA_MIDNIGHT_OUTDIR` overridden, `LOG` is the production default, so a refused
rehearsal mutates the production log before exiting 5.

I safely substituted a temporary path for the production default and executed
that direction: it returned `rc=5` but created the default log with the fired
marker. The suite only covers the opposite half-override. Move the pair guard
before the first log write and test both directions. This is a provenance bug,
not a reason to restore the released day-bar hold: no verdict artifact is
written on this path and the normal no-override nightly path is unchanged.

### `falsifier_count.sh` still accepts a partially crashed suite

Line 21 ends the pipeline with `|| true`, so it discards the Python/timeout
status. A temporary selftest that printed one `PASS` and then exited 1 produced:

```text
1@4a3d457
wrapper_rc=0
```

That is a plausible stamped count from a failed suite, contrary to the script's
“fails loudly” contract. Capture and require successful Python/timeout and grep
statuses before printing the count; preserve the nonzero-exit known-bad.

## 8. Execution record

All review execution used a detached worktree at the exact batch tip; the
shared dirty tree and production artifacts were not modified.

- scoped Python `py_compile`: PASS;
- `bash -n` on both scoped shell scripts: PASS;
- day-bar selftest: 68 checks, green;
- real O1/day-bar seam: 7/7, green;
- Iteration 011 core: 81 emitted PASS lines, green;
- Iteration 011 runner: 56 emitted PASS lines, green;
- fair-price Identity: 46/46, green;
- replay parity: 62/62, green;
- falsifier-count normal outputs: `81@4a3d457`, `56@4a3d457`,
  `46@4a3d457`, `62@4a3d457`;
- executed known-bads: day-bar 4,000-second per-coin outage, Iteration 011
  undefined-local AST check / false incumbent label / first-crossing inversion /
  empty-cell receipt, FP1 boundary bypasses, parity external-contract bypasses,
  midnight one-sided isolation, and partial-crash falsifier count.

## 9. Minimum closure order

1. Re-run and file the 08-29 day-bar verdict under the released code; proceed
   with O1's 08-30 runbook boundary.
2. Keep 011 dark. Fix the `pr` crash, implement the frozen action weights and
   first crossing, compute actual matched-random and paired-incumbent nulls,
   restore separate Q3+CI gates, and make the receipt guard validate cell
   contents. Then request another **non-fit** re-review.
3. Correct the Phase 2B settlement target before asking the user to freeze it.
4. Close FP1's direct-construction/depth/time guards and parity's external
   lifecycle boundary before either is consumed by an independent module.
5. Move the midnight isolation refusal before all writes and make
   `falsifier_count.sh` propagate failed-suite status.
