# Codex review — fix-batch 1 (`e72dd4c`)

**Exact reviewed batch tip:** `e72dd4ccec1355aff8ce68fed2228f306da377ef`

**Narrow post-tip addendum reviewed:** `b3f082e` only for the Q2
one-side-unevaluable correction requested in R-250. It does not rescue any
unrelated finding at the pinned batch tip.

**Filing verdicts:**

- **DAY-BAR V2: HOLD MAINTAINED. It may not judge 2026-08-29 yet.**
- **ITERATION 011: HOLD MAINTAINED. Do not fit or score.**
- **O1: ADVERSE PRE-DEPLOY FINDING. Postpone deployment until the producer ↔
  day-bar seam is repaired and exercised end to end.**
- Fair-price Identity and replay-parity are useful partial builds, but neither
  is integration-ready. **Do not freeze the Phase 2B challenger draft as
  written.**

No model was fit and no result was scored during this review.

## 1. Day-bar v2 re-review

### Requested acceptance cases

| item | result | executed evidence |
|---|---|---|
| (a) bars affect `all_pass` | **PASS in the isolated callable; FAIL end to end** | Clean predicates + green bars returned `True`; one failed v2 bar returned `False`; the same failed bar under the old regime returned `True`. However, the actual per-coin verdict is still computed before P1/P2/P3 are attached, described below. |
| (b) elapsed empty ledger | **PASS** | `coverage_observed=False`, `None`, omitted, and a truthy malformed value refuse; explicit `True` evaluates and an observed empty day passes. |
| (c) open-at-exit and structural rows | **FAIL at the real producer seam** | Missing-end synthetic `gap_open_at_exit` is charged and malformed/reversed/non-finite rows refuse. But the O1 collector's real row carries `gap_end_ns`, and the verifier refuses that producer schema. |
| (d) CLI dual-report keys | **PASS** | A synthetic complete-day report ran through `main()`: exit 0, both COIN-LEVEL and per-slug fields rendered, no traceback. |
| (e) nightly freeze epoch | **PASS** | The launcher seam records `--freeze-epoch 1787897340`, the `b3f7f9f` epoch. `bash -n` is green. |

The superseded raw-count predicate is also correctly excluded from v2
composition while remaining governing under `count_bar_v1_frozen`, and the
exact P3 counterexample now reports 1,000 seconds and fails.

### Release blocker DB1 — the per-coin verdict still omits P1/P2/P3

`verify_day()` constructs each `per_coin[coin]["all_pass"]` before the v2 bars
are built. Under `day_bar_v2`, `governing_predicates()` removes the legacy gap
count, leaving only post-freeze and completeness in the per-coin verdict.
P1/P2/P3 are later appended only to the global predicate table and global
`all_pass`.

Executed complete-day counterexample:

```text
btc_bar_all_pass       = false
btc_per_coin_all_pass  = true
whole_day_all_pass     = false
btc governing preds    = [entirely_post_freeze, complete_tape]
```

The BTC ledger contained a valid 4,000-second outage, so P1/P2/P3 failed while
the published BTC coin-day said PASS. This is decision-bearing because the
programme explicitly consumes per-coin verdicts for per-coin clocks; the
whole-day false cannot repair a false per-coin true. The per-coin artifact
must include its own P1/P2/P3 predicates and its own quality/accrual split.

### Release blocker DB2 / O1 blocker — producer and consumer disagree on
`gap_open_at_exit`

The committed O1 producer emits:

```text
event = gap_open_at_exit
gap_start_ns = ...
gap_end_ns = task-exit timestamp
```

The day-bar validator rejects every `gap_open_at_exit` whose `gap_end_ns` is
not `None`. Feeding an exact producer-shaped row to `day_bar_v2()` produced:

```text
ValueError: REFUSED: 1 gap record(s) ... lack a usable interval
```

The day-bar test uses a hand-built missing-end event; the O1 test verifies the
producer's start stamp but never passes its emitted row through the day-bar.
Thus both suites are green while their integration always refuses as soon as
O1d fires.

Required closure: establish one event contract. Prefer validating and using a
finite, ordered producer-supplied end; synthesize an end only when it is truly
missing, with the chosen scope recorded. Add a seam test that drives the fake
O1 socket, reads its actual ledger row, and consumes that row with the day-bar.

Because of DB1 and DB2, **HOLD MAINTAINED**.

## 2. Iteration 011 non-fit re-review

The strict target construction, Option-1 `p_pos`/`p_neg` targets, row-aligned
all-action predictions, Q4 composition primitive, fixed-family assembly
primitive, output guards, and identity guards are materially improved. The
isolated libraries are not, however, connected into an executable evaluation.

### Release blocker I11-1 — the real runner crashes after its first fitted arm

`report_arm()` returns head keys `Q2_p_pos` and `Q2_p_neg`. The main run then
prints `h["Q2_sign"]`. Executing that exact consumer path on a valid report
raises:

```text
KeyError: 'Q2_sign'
```

This occurs before the output artifact is written, so a real development run
cannot complete.

### Release blocker I11-2 — the fixed 24-cell evaluator is never invoked

The runner's `main()` fits/applies the two arms and writes per-head descriptive
reports. It never calls `build_cell`, `sign_flip_null`, `assemble_family`, or
`cluster_disclosure`; budgets are metadata only. Q4 predictions are composed
inside `apply_arm()` but are discarded by `report_arm()`. There is therefore:

- no evaluated 2 × 4 × 3 family in the output;
- no matched-random or incumbent null result;
- no per-budget decision metric or Q4 cell;
- no permutation p-values, fixed-denominator Holm result, or cluster
  disclosure from the real run.

The helper's unit tests prove the assembler in isolation, not that the runner
uses it. Wire the evaluator into `main()` and add an output-level known-bad
that refuses a receipt lacking all 24 declared cells.

### I11-3 — one-class sign heads are labelled `OK`

At `e72dd4c`, a 120-action probe produced `p_pos AUC=1.0`, `p_neg AUC=None`,
`p_neg status=OK`, and a Q2 cell of `1.0`. The coordinator's first flag was
therefore real.

The narrow `b3f082e` recheck is **green for cell propagation**: the same probe
now yields `Q2_sign=None` and `Q2_cell_status=UNEVALUABLE`; its runner selftest
adds five passing controls. However, the underlying `Q2_p_neg` head still says
`status=OK` with `auc=None`. A classification head without both target classes
must itself report `UNEVALUABLE` (or another declared non-OK status), not rely
only on the aggregate cell to correct its meaning.

### Coordinator flag 2

At the exact batch tip, the amendment did not freeze how the two Option-1 AUCs
become the single Q2 statistic. Post-tip commit `c014399` records the user's
ruling: `min(AUC(p_pos), AUC(p_neg))`, and the `b3f082e` implementation matches
that choice. I treat this design-choice flag as closed post-tip. It does not
close I11-1, I11-2, or the remaining per-head status defect.

Therefore **ITERATION 011 HOLD MAINTAINED; no fit or score**.

## 3. O1 pre-deploy check

The committed fake-socket suite executed **10/10 green** at the pinned tip:
3/3 ping settings, distinct subscribe-unconfirmed cause plus reconnect,
never-connected start stamping, escalating cause-aware backoff, reset after a
delivered message, and a healthy-feed positive control.

The cross-module `gap_open_at_exit` incompatibility in DB2 is nevertheless an
adverse O1 finding under the request's deployment rule. **Postpone O1** until
the producer-shaped ledger event passes the day-bar seam test. This finding
does not require changing the held live tree before the repaired boundary
procedure.

## 4. Fair-price Identity and Phase 2B draft

The Identity selftest is **21/21 green**, and its timestamp/as-of, readiness,
book-shape, depth, freshness, complement, and status-tally checks are useful.
Two correctness gaps prevent challenger integration.

### FP1 — the typed record does not enforce its invariants

`FairPrice` has no validating constructor/post-init, and `read_as_of()` accepts
any record with `status="OK"` and a non-`None` value. Executed probes:

```text
FairPrice(value=60000, status=OK, estimator=BinanceBookTicker)
    -> read_as_of(...) == 60000

identity_from_book(best_bid=99, best_ask=100, ...)
    -> status=OK, value=99.5
```

Coin, outcome convention, probability bounds, finite value, and consistency of
the stored freshness are likewise bypassable. This matters immediately because
challengers are required to construct the same type. Enforce the invariants at
the record boundary and add direct-construction known-bads.

The implementation also does not yet supply the interface spec's mechanical
toxicity no-double-count fence or challenger-declared-after-comparison refusal.
It is an Identity factory, not yet the complete successor interface.

### FP2 — raw Binance `bookTicker` mid is not a probability

The draft names the raw BTCUSDT/ETHUSDT mid as a challenger while the estimand
and Brier score require `p = P(Y=1 | state)` in `[0,1]`. A dollar price cannot
enter the `FairPrice.value` field or Brier score directly. Before freeze, name
and pin the point-in-time transformation from underlying price to settlement
probability, including reference/strike and tie convention, horizon, any
volatility/calibration inputs, and their lookbacks/shifts.

The draft also leaves decision-bearing terms unfrozen: “systematically later”,
“materially below”, the budgets, minimum paired sample count, test statistic,
alpha, and per-window/action weighting. Pairing alone makes scored counts equal,
so availability must be measured against a declared common eligible universe.
The constant-lag falsifier is not universally true on arbitrary realised paths
and should be defined on a controlled synthetic fixture.

**Recommendation: do not freeze the Phase 2B draft as written.**

## 5. Replay-parity battery

The canonicalization/selftest layer is **18/18 green**: the disabled anchor,
one-cancel perturbation, no float tolerance, signed-zero declaration, NaN
refusal, infinite threshold, basic lifecycle examples, hash-seed determinism,
and empty-run refusal all behave as stated.

It is still a partial checker:

1. `matched_control(opps, cancels)` ignores `cancels`. Requests for 0, 1, 6,
   and 99 cancellations all produced 12. Both “treated” and “random” arms run
   the same cancel-everything function; there is no random selection and no
   budget match.
2. `battery()` returns only the disabled anchor and infinite-threshold result.
   Matched control, lifecycle, and determinism checks exist only in the
   checker's selftest, not in an evaluated battery receipt.
3. The spec's zero-repost/permanent-hold anchor, requested/effective/suppressed
   rate-limit accounting, and no-policy-generated-training-reuse guard are not
   implemented.
4. The battery generates every arm through its own `run_stub_arm`; it cannot
   yet accept and check independently produced BE arm trajectories.

Treat this as a useful canonical trajectory prototype, not as clearance for
the seven-arm replay integration.

## 6. Execution record and count correction

All commands below ran in detached worktrees; the shared held collector file
was not touched.

- exact commit identity at the primary worktree: `e72dd4c...`;
- `py_compile` on the scoped Python files: PASS;
- `bash -n da_midnight_verify.sh`: PASS;
- day-bar selftest: 63 checks passed;
- Iteration 011 core: 81 emitted PASS lines, green;
- Iteration 011 runner at `e72dd4c`: 34 emitted PASS lines, green;
- annotation merge: 11 emitted PASS lines, green;
- Iteration 011 runner at `b3f082e`: 39 emitted PASS lines, green (+5);
- fair-price Identity: 21 checks, green;
- replay parity: 18 checks, green;
- O1 fake-socket behavior: 10/10, green.

The R-248 counts of 38 runner and 15 annotation checks were not reproducible;
R-250 records the coordinator's count-provenance correction. This is a
register-side correction, not a new code defect, and none of the verdicts above
depends on the mistaken counts.

## 7. Minimum closure order

1. Repair the O1 `gap_open_at_exit` producer/consumer contract and add the real
   seam test; then re-run the O1 deployment check.
2. Put P1/P2/P3 and quality/accrual into each per-coin verdict; re-run the full
   day-bar counterexample. Only then can the day-bar hold release.
3. Fix the Iteration 011 `Q2_sign` print crash, propagate one-class status to
   the head, and wire the complete 24-cell evaluator/Q4/budgets/nulls into the
   actual output. Only then request fit clearance.
4. Harden the fair-price record and amend the Phase 2B protocol before freeze.
5. Turn parity from a self-generated stub demonstration into an external arm
   checker with a real matched-random control and the missing lifecycle guards.

