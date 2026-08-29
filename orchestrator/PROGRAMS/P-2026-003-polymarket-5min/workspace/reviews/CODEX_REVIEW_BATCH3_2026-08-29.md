# Codex review — completed fix-batch 3 (`a63d717`)

**Exact reviewed batch tip:** `a63d7175c95c916e98319f4326e5d796d824bb88`

**Scope:** `REQUEST_BATCH3_2026-08-28.md`, the frozen Iteration 011
preregistration and A1 amendment, the corrected Phase 2B/A2 package, O1
relevance, the real inert trajectories, and a fresh sweep of the truncation
class. I fit and scored nothing. Review execution used a detached clean
worktree; the shared worktree and live collector state were not edited.

## Filing verdicts

- **ITERATION 011: HOLD MAINTAINED. DO NOT FIT OR SCORE.** Many named fixes are
  real, and both suites plus the dry-run now complete, but the actual runner
  still does not load or apply the incumbent. Q1 never performs its frozen
  incumbent comparison, Q4 reaches the receipt as candidate-only economics,
  and the action-unit and joint-gate semantics remain unsound.
- **PHASE 2B: NOT FIT FOR USER FREEZE.** Amendment A2's endpoint target is now
  supported by strong fresh evidence, but the claimed frozen-equivalence
  instrument is not frozen or exact, the challenger function accepts changes
  to frozen estimator inputs under the same identity, and the protocol still
  leaves the decision grid, inferential null, and multiplicity family
  unresolved.
- **O1: NO O1-RELEVANT ADVERSE FINDING.** Nothing in this filing changes the
  collector/day-bar contract cleared in batch 2. The O1 seam is 7/7 green and
  the forward-day verifier is green. **This review does not require postponing
  the planned 08-30 boundary arming.** It does not pre-judge the future day's
  health.
- **REAL INERT TRAJECTORY:** the committed receipt and both 61 MB artifacts are
  authentic, loader-accepted, and support the narrow inert-parity claim. They
  do **not** yet support real temporal/lifecycle semantics: the producer orders
  all windows on window-relative `t_start`, not the absolute governing instant
  `t0 + t_start`.
- **TRUNCATION SWEEP:** the class remains open. Three result-bearing streaming
  consumers accept a JSON array truncated after a complete row as if it ended
  normally, and the state verifier's production streaming path still checks
  row layout only in its first 400 rows.

## 1. Iteration 011 — hold maintained

### Closures that re-executed successfully

The round is materially better than batch 2:

- `phase2_iter011.py --selftest`: green, 102 emitted PASS lines;
- `phase2_iter011_run.py --selftest`: green, 98 emitted PASS lines;
- `phase2_iter011_run.py --dry-run`: exits 0 and writes a 24-cell receipt;
- the old undefined `pr`, Q2 one-side carry, first-crossing inversion,
  relative-hour stratum, two-sided matched-random/sign-flip directions, hollow
  receipt contents, and missing `sample_weight` call sites are closed in the
  tested helpers;
- the committed incumbent loader verifies hashes and its isolated apply tests
  respond to feature changes.

Those are capability closures. The production composition still does not use
several of them.

### I11-R3-1 — incumbent load/apply is unwired in the real runner

`load_verified_incumbent`, `apply_incumbent`, and `apply_incumbent_hazard` have
no production call site. Their only invocations are selftests. In `main()`, the
per-arm path remains:

```python
fr = fit_arm(arm, Xf, tgf)
ap = apply_arm(fr, Xe, tge)
rep = report_arm(ap, tge, EVAL[coin]["kept"])
rep["economics"] = q4_economics(ap, EVAL[coin]["kept"])
```

No incumbent is supplied. The executed dry-run therefore produced:

```text
12 AGGREGATION_UNDECLARED
 6 NO_INCUMBENT_COUNTERPART
 6 UNEVALUABLE
 0 OK
```

Every Q4 cell was `NO_INCUMBENT_COUNTERPART`; no Q4 p-value existed. Q1 only
carried its matched-random permutation. It never computed the frozen
candidate-minus-incumbent hazard comparison required by preregistration table
Q1 and section 5. The presence of a correct loader that the runner never calls
does not close rule 17.

### I11-R3-2 — fitting and metrics are still row-unit calculations relabelled
as action-unit

The amendment says a generation contributes once and every reported head uses
the action unit. The implementation computes `generation_weights(rows)` on the
unconditional row set and then subsets those weights for conditional heads.
An executed three-generation fixture produced Q2 weights `[0.5, 1.0, 1.0]`:
the generation with one preventable row and one non-preventable row contributes
half the mass of the other generations on Q2's own population.

Power checks in `fit_arm()` use `len(idx)`, not distinct action count. An
executed 200-row fixture containing only one action crossed the 100-observation
floor and fitted all five heads (`n_actions=1`).

Most decisively, `head_report()` calculates AUC and Brier over rows and merely
reports a deduplicated `n_actions`. Duplicating rows inside one of two actions
left `n_actions=2` but moved Q1 from:

```text
AUC 1.000000, Brier 0.010000
```

to:

```text
AUC 0.00990099, Brier 0.980588
```

The action population did not change. A metric that changes under row
duplication within the same action has not implemented the frozen action-unit
metric. The protocol needs a declared action-level label/prediction reduction,
then the fitting floor, weights, metric and reported `n` must all use it.

### I11-R3-3 — incomplete cells can survive the declared joint reading

`assemble_family()` applies Holm to every non-null `p_value` without requiring
that the cell's other governed legs passed. I constructed a complete 24-cell
family with one Q2 cell marked `NO_INCUMBENT_COUNTERPART` but carrying
matched-random `p=0.001`. It received `holm_p=0.024`, was marked
`survives_joint_reading_at_0_05=true`, appeared in `surviving_cells`, and
passed `assert_receipt_has_all_cells()`.

A named failure is honest reporting; it is not a discovery. The joint survivor
predicate must be the conjunction of the cell's frozen gates, not merely
`Holm(p) < .05` on whichever p-value happened to exist.

### I11-R3-4 — Q3's user-ruled gate is not computed

The user ruling is explicit: separate `m_harm` and `m_good` calibration-slope
intervals must each exclude 0; null-at-1 is diagnostic. The code computes point
slopes and a matched-random p-value. It computes no slope interval. It also
correctly emits `AGGREGATION_UNDECLARED` for Q3's two slopes and for the
two-coin collapse, but those open design choices cannot be deferred until
after numbers exist. The per-coin validation regime and a fixed 24-cell family
without a coin axis also remain inconsistent.

Therefore **ITERATION 011 HOLD MAINTAINED; do not fit or score**. Release needs
the incumbent wired through Q1 and Q4, a real action-unit reduction throughout,
status-aware joint adjudication, and a pre-result ruling/implementation for Q3
and the coin axis.

## 2. Phase 2B/A2 — corrected target, package not freeze-ready

### What is now credible

I reproduced the settlement result independently. On the pre-registered recent
split, `S60(T) >= S60(t0)` reports `n=8,022`, 99.8504% endpoint agreement, and
passes; the full-window form reports 85.2157% and fails. The separation is real
and large. The endpoint-staleness disclosure is also useful: 11 of the 12
disagreements lie in the 91 observations older than 2 seconds, while the
`<=2s` subset is 7,898/7,899.

Amendment A2's target, terminal-only part-realization, Chainlink
`crypto_prices_twap_sixty` reference, wrong-stream refusal, tie handling, and
full-window reduction identity are coherent. I accept **A2 as the working
settlement target**.

### 2B-R3-1 — the “one frozen snapshot” equivalence claim is false

`snapshot_inputs()` copies `markets.jsonl` and `resolutions.jsonl`, but creates
`prices` as a symlink to the growing live directory. The audit side also calls
the imported `load_streams()` against its module-global live `PM`; the original
subprocess reads the snapshot symlink. Both therefore read the same live price
files sequentially, not immutable price bytes.

The committed receipt has `n_windows_all=18,047`; my exact-tip rerun later
observed 19,148 while the recent split stayed 8,022. Growth is visible. The
receipt contains a temporary `snapshot_dir`, but no source manifest or source
hashes with which a future reader could reproduce the input.

The comparison is not exact either. `assert_equivalent()` compares only `n`
and one-decimal formatted event-time `agree`/`agree_big`. It omits hit counts,
exact floats, knowledge-time results, exclusions and source identities. An
executed known-bad with 9,976 versus 9,984 hits out of 10,000 was accepted
because both format as `99.8`.

The recent endpoint conclusion remains persuasive; the new harness does not
yet justify its stronger “frozen, enforced-equivalence” provenance claim.

### 2B-R3-2 — frozen estimator inputs are mutable under one estimator name

The fair-price selftest is green at 110 checks and the named round-3 closures
work. Direct executed calls still accepted all of these:

```text
FairPrice(estimator="NOT_DECLARED", otherwise valid)       -> accepted
bn_bookticker_s60_probability(window_s=30)                 -> OK
bn_bookticker_s60_probability(window_s=-60)                -> OK, p=1
sigma_lookback_s=1 or NaN                                  -> OK
correct lo/hi PartialTwap, integral=1e9, covered=span=0,
status=OK, n_used=0                                        -> OK, p=1
```

Thus the A2 60-second target and the declared 30-minute shifted volatility
lookback are caller-controlled metadata, not enforced estimator identity.
`PartialTwap` has no validating record boundary; matching interval endpoints
alone does not prove coverage. The structural challenger also accepts `spot`
as an un-attributed scalar: no source, receive timestamp or hf_ws_v2 era
identity reaches the signature. The reference similarly lacks a verifiable
source interval/coverage record even though A2 makes complete
`[t0-60,t0]` coverage an admissibility precondition.

### 2B-R3-3 — the scoring protocol remains underdeclared

The operative draft still says an honest Brier forecast “maximises” expected
score; Brier **loss is minimized**. More importantly:

- it declares “two challengers × budgets” as the Holm family even though the
  defined Brier forecast score is not budget-specific;
- it gives no exact UTC-day inferential null/test for “positive,
  Holm-corrected” skill;
- “every decision instant” has no frozen instant generator or common eligible
  grid, so update cadence can change the weighting and population.

These choices can move a pass/fail result and must be frozen before data, not
filled in by the scorer. Therefore **the corrected 2B package is NOT FIT FOR
USER FREEZE**. The minimum closure is an immutable and exactly compared audit
snapshot; enforced estimator constants/source identities; a validating
`PartialTwap`/reference boundary; and an amended score section with the exact
instant grid, day-unit null and derivable multiplicity family.

## 3. O1 boundary — no adverse finding

I re-ran the exact-tip real seam and verifier:

```text
da_o1_daybar_seam_test.py                 7/7 green
da_forward_day_verify.py --selftest       80/80 green
```

The producer-supplied end, synthesized open-at-exit end, structural refusal,
dual-report and freeze-epoch paths remain intact. The Iteration 011, Phase 2B,
state-tape and trajectory findings in this filing do not touch the O1 collector
or its day-bar consumer. **No O1-relevant adverse finding is filed; do not
postpone the 08-30 arming on account of this review.**

## 4. Real inert trajectory — authentic narrow milestone, broken global clock

I re-ran `be_inert_arm_run.py --selftest` (16/16 green), checked the committed
receipt, and recomputed both local artifact hashes and sizes:

| arm | bytes | recomputed SHA-256 |
|---|---:|---|
| `QR_CANCEL_HOLD_X_SKEW` | 64,259,195 | `af878dff6d64ec5d538ca430fe68fc0fe5c15e5e5334d77c2e2cbab5e15932d7` |
| `QR_SKEW_ONLY` | 64,259,173 | `818c71f6eb54287dd3d594b2087c02c32827d2f6fb068732cc45bea83ac9d19a` |

They match the receipt. Both contain 457,268 `PLACE` events, their canonical
inert digests match, and the receipt's no-economics/easy-input limitations are
honest.

However, `opportunities()` discards `t0`, sets event `t=float(t_start)`, and
sorts globally on that relative offset. A two-window falsifier with governing
instants 2,000 then 1,200 emitted `later-window@t=0` before
`earlier-window@t=200`; no opportunity retained `t0`. The real artifact begins
by interleaving unrelated slugs around `t=-60`, confirming this is not merely
a synthetic edge.

This does not invalidate inert bit parity: both inert producers consume the
same damaged opportunity ordering and only emit `PLACE`. It does invalidate a
broader “real trajectory” reading. Active rate-limit windows, cancel lags,
repost/hold sequencing, and cross-window order cannot be evaluated until event
time is the absolute governing instant (`t0 + t_start`, with its clock basis
declared and checked).

## 5. Truncation/per-row sweep — closure not reached

### T1 — the fixed layout helper is bypassed after the first 400 rows

`locate_features()` now validates every row **passed to it**, but production
`verify()` passes only its 400-row buffer, chooses `under`, then flattens the
rest of the stream without per-row conformance. An executed 401-row tape with
valid nested state in rows 1-400 and an empty `state={}` in row 401 returned:

```text
n_rows=401, verify_refused=false, all_pass=true
```

That is the exact union-counterexample class the batch says is closed; the
helper closure did not reach its streaming consumer.

### T2 — EOF before `]` is treated as normal completion in three consumers

I wrote a valid declared header plus one complete `OK` row, then truncated the
file before its closing `]}`. All of these returned the completed prefix with
no refusal:

```text
harmful_rows_loader.stream_ok_rows      -> 1 row
be_inert_arm_run.stream_rows            -> 1 row
phase2_arms._stream_tape_rows           -> complete prefix rows
```

`da_state_tape_verify._stream_array` has the same structural fault: a file cut
as `{"rows":[{"x":1},{"x":2}` yielded both rows and no refusal. These loaders
feed Iteration 011, the inert/active trajectory line, and the Phase 2 fit/score
index. A future truncated artifact can therefore shrink its tail and still be
reported as a complete population.

The state verifier selftest is also non-hermetic at the reviewed ref: in the
clean worktree it crashes trying to read the ignored live
`data/pm_5min/collector_gaps.jsonl`; it passes only in a workspace where that
external file happens to exist. This is reproducibility debt, separate from the
executed false-green above.

## 6. Additional execution record

- settlement-audit checker selftest: 5/5 green;
- exact settlement audit rerun: completed, `n_recent=8,022`; maximum RSS about
  2.62 GB; run output redirected so the committed receipt was not overwritten;
- replay-parity battery: 108/108 green;
- production seam suite: 1 deliberate red at 47j. The live fit manifest pins
  gate hash `1da60b56e1fb2801` while the reviewed checker hashes
  `a07edebb3cb7383d`; this is the explicitly recorded R-275/R-277 state awaiting
  a fit-time restamp, not a new O1 issue and not silently converted to green;
- state verifier: non-hermetic failure in the clean worktree as described;
- executed known-bads: incumbent production call graph and dry receipt,
  row-duplication/action power, invalid-status Holm survivor, mutable/rounded
  settlement equivalence, fair-price frozen-input bypasses, relative trajectory
  clock, 401st-row layout loss, and four truncated-array readers.

## 7. Release order

1. Keep 011 dark. Close I11-R3-1 through I11-R3-4 and request another non-fit
   review. Do not use development numbers to choose the missing aggregation
   rules.
2. Keep A2 as the working settlement target, but amend and re-review the whole
   2B freeze package before asking the user to freeze it.
3. Proceed with O1's existing 08-30 runbook boundary; this review adds no O1
   hold.
4. Preserve the inert receipt as a narrow parity milestone. Repair and falsify
   the absolute trajectory clock before any active cancel/rate-limit run.
5. Make EOF-without-array-close a refusal in every streaming consumer and run
   per-row state-layout conformance throughout the production stream.
