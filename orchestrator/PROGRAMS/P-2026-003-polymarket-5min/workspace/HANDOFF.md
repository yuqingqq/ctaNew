# HANDOFF — P-2026-003 Polymarket Crypto 5-min

Updated: 2026-08-20, session 3. All work is on branch **`mm-research`** (pushed);
nothing is on `main`. Sigma is at **Revision 5** / contracts **v16**;
Route A has now been measured under frozen protocol `route_a_v1`. The result is
**DESCRIPTIVE / PRICING HOLD**: one OOS test day cannot authorize a law.

## Read this first

The programme now has a **verified settlement target**, a **modular
architecture with machine-readable contracts**, and a reviewed sigma plan.
Revision 5 makes the reduced-form/structural split hold at the executable
boundary. A real, strictly-forward Route-A candidate is now fitted across all
42 symbol/horizon cells, but the current tape has only two collected days and
one OOS test day. All 84 gates are therefore `INSUFFICIENT_EVIDENCE`, so
probability-level use remains on **HOLD**. The Route-A input lane may keep
accumulating through frozen filters. The historical CLOB lane is **DEGRADED**;
a second receive-path repair is now live and monitoring, but has not passed its
full busy-day acceptance gate. Two headline results from
session 1 were withdrawn or downgraded after discovering bad underlying data;
do not cite either without reading §"What was withdrawn".

Reading order:
1. `live/pm_research/PM_ARCHITECTURE.md` (v12) — the entry point; structure.
2. `live/pm_research/contracts/contracts.yaml` (**v16**) — machine-readable
   source of truth for types (**v16**). The prose defers to this file, not the
   other way round.
3. `live/pm_research/SIGMA_ROUTE_A_RESULTS_2026-08-20.md` — the measured,
   strictly-forward Route-A result and current verdict.
4. `live/pm_research/SIGMA_ROUTE_A_PROTOCOL.md` — protocol frozen before fit;
   includes the non-analytic post-run embargo-wording erratum.
5. `live/pm_research/EXP_RESULTS_2026-08-20.md` — earlier model results.
6. `live/pm_research/SIGMA_PLAN.md` — **REVISION 5, canonical.** One consumer
   matrix, one PRICING law (route A) and one DIAGNOSTIC decomposition (route B),
   never summed, now enforced as a TYPE boundary. **Read §2.3 then §1a** — the
   route decision scopes everything, and §1a says where each consumer's number
   actually comes from. v1/v2 text is in git history.
7. `live/pm_research/SIGMA_PLAN_REVIEW.md` — first implementation-readiness review.
8. `live/pm_research/SIGMA_PLAN_REVIEW_ITER2.md` — review of Revision 2.
9. `live/pm_research/SIGMA_PLAN_REVIEW_ITER3.md` — review of Revision 3 and v14;
   historical input to Revision 4.
10. `live/pm_research/SIGMA_PLAN_REVIEW_ITER4.md` — review of Revision 4/v15; its
   six items are applied in Revision 5/v16.
11. `live/pm_research/SIGMA_PLAN_REVIEW_ITER5.md` — pre-measurement verdict:
   MEASUREMENT GO / PRICING HOLD**, plus the frozen fit sequence.
12. `live/pm_research/sigma_kernels.py` — executable model **fixture**, not a
   frozen spec. `--selftest` checks exact arithmetic under a **declared and
   still UNVERIFIED** sampling convention; it does not establish that convention
   against the Chainlink streams.
13. `live/pm_research/plans/` — BE_BELIEF, BE_FLOWANDFILLS, MEASUREMENT,
   PRELIMINARY.

Before any book/trade/queue analysis, read
`live/pm_research/DATA_COLLECTOR_AUDIT_2026-08-20.md` — current collector
verdict, live evidence and acceptance boundary. **Read its v3 section: the v2
addendum's "repair successful" is withdrawn, and the root cause is now measured
rather than hypothesised.**

### Session close-out 2026-08-21 01:34 — what the monitoring loop established

**The binding constraint is now the calendar, not the code.** `route_a_v1`'s
frozen gate needs **10 OOS test days**; 2026-08-19 is training-only, so the count
is **1 of 10** (08-20 complete, 08-21 in progress), tracking to ~2026-08-29.
Nothing about the collectors or the spec blocks that; it has to elapse.

**The Route-A prices lane is self-healing, which is not the same as clean.**
I had been reporting it healthy on the strength of `open_gaps=[]` at each check.
Over 11 hours it actually logged **58 gaps and 26+ reconnections** — roughly one
11–13 s gap every 20 minutes — under four causes: `GLOBAL_SOCKET_SILENCE` 38,
`TOPIC_STALE` 7+, `PEER_TOPIC_RECONNECT` 3, `CONNECTIONCLOSEDERROR` 2. Per-hour
counts `15:4 16:2 17:4 18:8 19:4 20:6 21:10 22:10 23:4 00:4 01:2` — a level in
the 2–10/hr band, no trend. Every gap closed; none ever left open.

**Why that matters:** an 11–13 s gap landing on a decision time breaks the
protocol's ≤5 s predictor-staleness rule for that horizon, and a long one breaks
the 90 % coverage rule for the whole window. **This is a candidate mechanism for
`route_a_v1`'s 374 `s30_window_coverage` exclusions (19 % of windows)** — the
MNAR risk that is still unaudited. One outlier so far: a **44.8 s `TOPIC_STALE`
pair at 22:28**, 3.5× any other, which at 14.5 % of a 310 s span fails the 90 %
rule outright for every coin and horizon.

**CLOB lane, closed out.** 46 disconnects across all eras; `ws_ever_paused` has
read True **0 times**, deepest backlog 254 of 65,536 (0.4 %). Cumulative loss
149.6 s across 36 windows, **zero unclosed gaps**. 1013s are bursty around
2–3/hr with whole hours of quiet. Gap-touched BTC windows sit at ~22 %.

**A discipline note worth keeping.** Twice I called a direction on one interval —
a "rising 1013 frequency" and a "30 % gap-touched share" — and both dissolved
within the hour as the denominator grew (30 → 25 → 23 → 21 → 22 %). Report burst
minutes and per-hour counts; do not describe a direction until several hours
separate the observations. Neither claim reached these documents, which is what
saved them.

**Correction, so nobody re-derives the wrong urgency.** I claimed twice in
session that the counts-only exclusion ledger meant selection-audit information
was "not reconstructible later". That is **wrong**: the protocol reads immutable
rotated `.csv.gz` archives plus jsonl byte snapshots, so re-running it reproduces
the identical exclusion set and window identity can be recorded then. Building
the exclusion ledger is a **scheduling choice, not a race**.

### Residuals — open, in priority order

1. **`route_a_v2` variance spec — pre-register BEFORE the refutation arrives.**
   `route_a_v1`'s variance gate is on track to read `MODEL_REFUTED` for a
   predictable reason: OOS residual variance is **U-shaped in `|S30−S60|`**
   (mid tercile ~0.70–0.80, both tails 1.1–1.7, i.e. 1.4–2.2× across terciles)
   against a frozen tolerance of 0.25. That is volatility clustering — the same
   ρ = 0.19–0.40 persistence already in the plan — and it refutes **pooled
   residual variance**, not route A. The disciplined move is a conditional
   variance spec frozen *now*; fitting one after seeing the refutation is
   exactly what pre-registration exists to prevent.
2. **Exclusion ledger** in `exp_sigma_route_a.py` — exclusions are a bare
   `Counter` (13 increment sites), so the accepted-versus-excluded activity and
   volatility comparison that both the protocol and the collector audit require
   cannot be run. Not urgent (see correction above), but it gates the day-10
   verdict.
3. **`PING_TIMEOUT` classification, unresolved at n=8.** It is **8/8 BTC**, which
   behaves like the MNAR `SLOW_CONSUMER_1013` class, not like the across-coins
   MAR cycling class where `3b9bddc` grouped it. If it stays BTC-only as n grows,
   the cause-aware rule must move it. Accumulate before amending.
4. **Sigma spec** remains at Revision 5 / contracts v16 with Phase 0A 5–6 unrun
   (S30/S60 semantics; the route-A fit is descriptive until 10 OOS days).

### Collector: the 1013 is VENUE-SIDE — measured, not inferred

**Resolved 17:46:41 UTC.** `clob_v3_1` samples the `websockets` Assembler, which
pauses reading from the transport once its inbound backlog passes a
65,536-frame high-water mark — and a paused transport is exactly what fills a
server's send buffer. On the first 1013 under v3_1:

```
ws_ever_paused      False        <- never stopped reading
ws_queue_depth_max  133          <- 0.2% of the pause threshold
lag_ms_max_interval 1.8 ms
```

**We were draining at 0.2% of capacity while the venue said its send buffer was
full.** Every client-side cause is now excluded by measurement: loop stall
(1.8 ms), gzip (off-loop, none in flight), write backpressure (`writer_wait=0`,
`q_hi=1`), memory (RSS 260 MB stable), and network throughput (11.7 Mbps
sustained; one BTC socket is 0.24 MB/s).

**Two successive repairs failed because neither addressed the cause.** The v2
write-queue decoupling and the v3 gzip offload were both real defects worth
fixing — the gzip stall was 1.8–1.9 s of measured loop block — and neither was
the answer. I asserted the gzip finding as the root cause; that was an inference
written as a measurement, and it is corrected in `2d5503f`.

**What this changes.** The acceptance boundary as written — one full busy UTC
day with zero `SLOW_CONSUMER_1013` — tests something we do not control and is
probably unachievable. The pre-registered *alternative*, a cause-aware exclusion
rule with enough complete independent days, is now the operative path. That
branch existed before any of this was known, which is what makes the finding
actionable instead of a dead end.

**The exclusion rule already has its input**, and there are **two loss
mechanisms**, which is what makes the pre-registered *cause-aware* framing
load-bearing rather than stylistic:

| mechanism | cause codes | pattern | missingness |
|---|---|---|---|
| venue send-buffer / slow-consumer label | `SLOW_CONSUMER_1013` | 12 of 14, **all BTC**, bursty | **MNAR** — activity-correlated |
| venue server cycling | `CONNECTIONCLOSEDOK` (1001), plausibly `PING_TIMEOUT` / `NO_CLOSE_FRAME` | hits whichever sockets a restarting server held, across coins | plausibly **MAR** |

They need different handling. A 1001 going-away gap can be excluded and the rest
stays representative; a 1013 gap cannot, because it lands preferentially on the
busiest windows, so *excluding it is itself a selection* and the excluded set
must be reported next to the retained one — the lesson of the original MNAR
incident. A rule keyed only on seconds-lost would treat the two as
interchangeable.

Loss to date: ~40 s across 10 windows, worst `btc-updown-5m-1787247000` at 21.5 s
(5.5% of its 390 s). Tally by coin: **btc 12, sol 1, eth 1** — an earlier note
saying every disconnect was BTC was true when written and is now superseded.

Posture: stop fixing this client-side; keep `clob_capture_clean: false`; leave
`DISK_WORKERS` and `ping_timeout` alone — there is no client-side hypothesis
left to test.

### Historical — `clob_v3` deployed 16:31:26 UTC

**The v2 repair did not close the failure.** `clob_v2_1` logged a
`SLOW_CONSUMER_1013` on BTC **5.8 minutes after deployment** and finished its
80-minute run at `retries=5 slow=5`, all BTC, over 4.82M messages — **one drop
per ~16 minutes**. The v2 addendum called the repair successful at 15:12; the
ledger contradicted it at 15:16.

**Root cause, measured:** `gzip_atomic` ran **synchronously on the event loop** —
**1,818–1,915 ms** to compress a ~180 MB BTC shard at level 6, every five
minutes per coin, during which **no socket is drained** and the venue's send
buffer to us fills. The v2 write-queue repair was real but was never the binding
constraint: `writer_wait=0`, `q_hi=1` across 4.8M messages.

`clob_v3`: gzip off-loop on a dedicated disk pool; **disk and HTTP executors
split** (they shared the default 20-worker pool, where a stalled `urlopen` could
starve a shard write and reproduce the same 1013 by a second path); an
**event-loop lag probe** reported per heartbeat and **stamped into every
disconnect**; `gap_open_at_exit` so a gap running to window end is no longer
indistinguishable from a lost close record; `markets_force_cancelled` replacing
the misleading `active_markets_drained` (2 reported, 14 actually drained); and a
narrow chunk-loss window on cancellation closed.

**Why the lag probe matters more than the fix.** A 1013 has two candidate causes
with *opposite* remedies — the loop stalled (offload work) or the socket rate
genuinely exceeded capacity (shard connections across processes). Nothing
previously distinguished them, so every diagnosis was an argument. Now it is a
number in the disconnect record.

Selftest is 12 checks including a **control**: the same gzip inline must stall
the loop ≥100 ms and ≥20× the off-loop figure, or the off-loop test proves
nothing. Measured **211 ms on-loop vs 0.5 ms off-loop, 393×**.

**Acceptance is unchanged and not yet met:** one full busy day with zero `1013`,
or a cause-aware exclusion rule with enough complete independent days. Compare
against the v2_1 baseline of one drop per ~16 minutes. A few clean minutes prove
nothing — that was exactly v2's error. Never pool v2 and v3 rows without the
`collector_version` field the ledger records.

## Done this session

**Route-A sigma candidate measured — DESCRIPTIVE / PRICING HOLD.** The
preregistered `route_a_v1` run produced **9,332 admissible rows**, **5,796
strictly-forward OOS rows**, and all **42** independent fits (7 symbols x 6
horizons). Settlement direction agreed **1,560/1,560** after admissibility
filters. An independent post-run audit found zero timing, formula, uniqueness,
fold, coefficient or source-hash violations. Only 2026-08-20 is an OOS test
day, so every one of the 84 gates is `INSUFFICIENT_EVIDENCE`. The point
diagnostics are not reassuring—42/42 conditional-mean effects and 40/42
conditional-variance effects exceed their frozen tolerances—but one regime-day
cannot refute the law. Full result: `SIGMA_ROUTE_A_RESULTS_2026-08-20.md`.

**E-M6 settlement truth — PASSED, the foundation gate is cleared.**
`S60(T) vs S60(t0)` reproduces the winners Polymarket actually paid on
**1,465 windows at 99.8%** (99.9% restricted to `|margin| > 0.5 bp`). This
settles the open `w = 60 s` vs `300 s` ambiguity: the full-range reading scores
86.9% and is **refuted**. Reading the same grid at knowledge time gives 99.3%,
and that 0.5 pp gap is the size of the look-ahead a careless backtest banks.

**E0 data integrity audit** (`exp_e0_data_audit.py`, 7 checks) quantifies every
known incident rather than assuming it benign: the duplicate-collector overlap,
the ~16 min market-side outage, the 8 malformed resolution rows, restart shards,
TWAP gaps, knowledge-time lag per coin, and the up-rate drift confound.

**Architecture v12 + machine-readable contracts.** 6 planes, 85 contract types,
a structural diff checker with 13 selftests, and version-bound migration
records. Twelve external review iterations. **Two of my own artefacts were
proven unsound during that loop and replaced** — worth knowing because both
failures were of the same kind, a checker that reported success without
checking:
- the first contract checker: an invalid ref exited 0, generics were invisible,
  and narrowing a type produced an identical inventory. All three reproduced.
- the path-keyed allowlist (M11-1): entries authorised **any** change at that
  path. Reproduced `CouplingSource -> CompetitionState` passing. Replaced with
  migration records bound to (operation, key, old, new, version).

**Collector MNAR bug found; second repair deployed, acceptance pending.** The
hot loop
allocated an `asyncio` timer per message; at BTC's rate that dominated and
backed the server's send buffer up into `1013 slow consumer` disconnects. **27
of 47 disconnects were our
own doing, and 32 of 47 were BTC** — i.e. the loss was concentrated in exactly
the busiest intervals, which is missing-not-at-random. The initial short probe
read 0 post-fix drops, but extended observation of the repaired 10:55 process
found **13 further 1013 closes across 11/41 recent completed BTC windows**, plus
seven other BTC retries. `50dd889` replaces that path with a minimal receiver
and bounded ordered writer queue; `2deb8e8` adds active/rate/age telemetry.
The first versioned high-load run lasted **19m23s**; its last heartbeat reported
**908,843 messages** (**552,166 BTC**) with zero retries, slow closes or writer
waits. That is a successful smoke test, not the required full busy day. Never
pool an unpaired statistic across repair eras, and do not use this tape for
flow/fill/queue inference yet.

**Collector lane audit.** Discovery grids are complete, resolutions are current
(1,963 final, zero give-ups), TWAP parsing has zero malformed rows or negative
knowledge lags, and capacity is ample. The price socket nevertheless has
unreplayed global gaps: recent full-horizon admissibility is 224/273 (82.1%).
That is sufficient row flow for filtered Route-A accumulation, not evidence
that excluded regimes are ignorable. `prices_v2` now detects global and
per-topic silence at 8 s and persists exact topic gap boundaries; its first
real outage recovered both topics after about 11.5 s. It reduces future loss
but does not repair historical missingness. See
`DATA_COLLECTOR_AUDIT_2026-08-20.md`.

**SIGMA_PLAN Revision 5 reviewed and measured.** The route split and fit
specification remain frozen; the next action is more OOS days, not Revision 6.

## What was withdrawn — do not cite these

**1. "The book beats our model at every horizon" — WITHDRAWN, not held.**
The 2026-08-20 run showed the book winning by a stable 2.5–3.2 Brier points at
all six horizons, which read as a uniform information deficit and prompted the
conclusion *"no alpha, therefore pure market making"*. That model was
**mis-anchored**: `E_t[X_T]` used the trailing S60, which lags spot by
`w/2 ≈ 30 s`, while being paired with a *conditional* variance law. The
resulting `σ_eff` was ~2.6× too small at `r = 30`. The candidate
`P̂ = 2·S30 − S60` gained **−0.0101 Brier pooled, at every horizon** on one test
day, but Revision 3 correctly shows that coefficient is a biased trend
extrapolator under the Brownian fixture. It establishes that the lagging-S60
direction was wrong, not that `alpha=2` is the final anchor. The residual verdict must be re-read
on the corrected specification before it means anything.

**2. The FLB edge — downgraded from an edge to a rounding error.**
"+3.6 c/share at `p ∈ [0.15, 0.35)`, stable" was the measurement that
recommended the Option-B re-scope. It was computed on `book` snapshots, which
are **p90 6.2 s stale**. Rebuilt from `price_change.best_bid/ask` (the
executable quotes): `b̂ = 1.145` where the stale read inflated it to 1.182, the
walk-forward gain is **0.0004 Brier**, and the effect is **one-sided** — a drift
signature rather than a genuine bias.

**3. Everything book-derived in `PM_DEEP_REVIEW.md` inherits defect 2**,
including the "+95 bps maker gross / +136 with rebate". Treat as unverified
until re-measured on `price_change` quotes.

**4. Five premises I had briefed into the flow/fills work were wrong**, all
corrected in `plans/BE_FLOWANDFILLS_PLAN.md`: trades are NOT double-reported
(zero dupes); the modal spread is **1 tick**, not 2–4 (ATM runs 6–8 c); the
1.7 s lag is the **signal clock only** — the book itself arrives in **47 ms**,
which materially softens the session-1 FATAL-1 latency finding; `side` is the
**taker's** (90.8%); and `price_change` carries **post**-change quotes.

## ⛔ Decision point A/B is still open, but B lost its prior

Session 1 closed on Option A (fix the mechanism program) vs Option B (re-scope
to FLB harvest), with the reviewer *and the data* recommending B. **The data
that recommended B was measured on stale books** (withdrawal 2 above). Re-frame
the choice before making it; do not read `PM_MM_PLAN §17`'s recommendation as
current.

## Immediate next step — accrue OOS days and rerun route_a_v1 unchanged

Phase 0A step 6 now executes end to end. Preserve `route_a_v1` and rerun it as
immutable days accrue. A formal verdict needs at least **10 OOS test-day
clusters** and 30 OOS rows in every frozen conditioning cell; because the first
day is training-only, that normally means at least 11 collected days. Do not
respond to the one-day point effects by changing the cells, tolerances or
functional form.

Phase 0A step 5 may proceed in parallel and gates Route B only. Do not add
structural `k/v/Omega` terms to the Route-A residual. Probability-level use and
the estimator integration remain on HOLD until all per-fit gates pass.

In parallel, monitor `clob_v2_1`; the receive-path and slug/token/time gap
ledger repair is deployed. No model-vs-book, queue, flow or fill result may use
this lane until a full busy day has zero 1013 losses or a pre-registered
cause-aware exclusion leaves enough complete day clusters. Derive collector
version from the versioned start/stop ledger, preserve it as a repair-era
field, and never pool pre-v2 observations unpaired.

Iteration 5 found three cleanup items to land alongside that implementation:
the duplicate YAML `ReducedFormFit` plus stale `ReducedFormLaw`, malformed-input
refusal totality, and validation of the `GateEvidence.effect_size` payload.
They are not new model choices and do not block the regression. Full evidence:
`SIGMA_PLAN_REVIEW_ITER5.md`.

### Historical — iteration-4 boundary review and application

Revision 4 makes the right central decision: **Route A's reduced-form residual
is the whole pricing variance; Route B is a structural diagnostic; never add
them.** Iteration 4 retains that decision and narrows the HOLD to six integration
problems:

1. Route A is documented as independent of internal S30/S60 kernel semantics,
   but `ReducedFormLaw` and `pricing_var` still require its convention to be
   `VERIFIED`. Remove that Route-B gate from Route A.
2. `pricing_var` and `conditional_mean` bypass `check_request`; the helper also
   accepts future-issued laws and reversed target intervals. Build one atomic
   request-to-distribution API with unavoidable temporal/link checks.
3. Negative residual variance and NaN evidence can price, while infinite rates
   throw. Validate every domain and return a typed refusal.
4. High p-values are treated as proof of conditional mean/homoskedasticity.
   Pre-register per-symbol/per-horizon equivalence/calibration gates with effect
   sizes and confidence bounds.
5. Route B's `anchor_var` changes with empirical alpha because it absorbs
   squared model gap, and it exposes `model_total_var`. Separate bias/MSE and
   make the diagnostic a type that cannot satisfy a pricing protocol.
6. `PathLaw` is not a discriminated route union; YAML schedules lack offsets;
   kernel coefficients lose their seconds unit; `CalibrationCurve` is stale;
   and the source of operational dynamics shape is unresolved.

Shipped checks passed at the time of that review (kernel **41** pre-repair,
checker 13, v14→v15 migration), but focused
adversarial probes reproduce every issue. Full evidence and acceptance tests are
in `SIGMA_PLAN_REVIEW_ITER4.md`.

### ITER4 APPLIED — Revision 5 / contracts v16 / boundary rewritten

Every ITER4 probe was **reproduced before acting**: `resid_var = -1` priced; NaN
cluster counts and NaN p-values passed (every ordered comparison against NaN is
false); a **future-issued law with a reversed target interval** returned `True`;
`anchor_var` moved with the empirical alpha; `model_total_var` was exposed;
pricing took no request; an infinite rate raised `OverflowError`. All correct.

**M4-1 — the one that mattered, and it was a self-contradiction.** The plan said
route A regresses published streams and needs no internal kernel; the code
refused route A unless the convention read `VERIFIED`. That gate removed the very
advantage that selected route A. Route A's precondition is now
**`StreamProvenance`** — stream identities, point-in-time reads, units, alignment
at *published* timestamps. `SamplingConvention` gates the structural arm and
nothing else, and there is a test that a well-formed route-A law prices **while
every convention is still UNVERIFIED**.

**M4-2 — one atomic query.** `pricing_distribution(law, request, observables)` is
the only pricing entry. It validates every temporal and identity invariant
*before* computing either moment and returns mean and variance from **one**
validated fit. v15 exposed `pricing_var` and `conditional_mean` separately and
neither called the checker, so correctness rested on every future caller
remembering a pre-call. Now also refused: a law issued after the request, a
reversed target interval, and observables newer than the knowledge cutoff. Every
boundary refusal carries `since` and a machine-actionable `cause`.

**M4-3/M4-4 — the gate was numerically and statistically unsafe.** Positive,
finite, integer validation throughout; NaN and ∞ refuse instead of passing or
raising. And **failure to reject is not equivalence**: `GateEvidence` carries a
verdict (`PASS` / `INSUFFICIENT_EVIDENCE` / `MODEL_REFUTED`), an effect size and
a tolerance, and `PASS` requires the |effect| confidence bound *inside* the
tolerance. A p-value gate treats "not enough data" as "verified", which is
exactly backwards at a ten-cluster minimum.

**M4-5 — bias had crept back into the variance, on route B.**
`cond_var_at_model` now takes **no alpha**, so an empirical anchor can no longer
change the alleged conditional variance. Route B returns a **distinct type** with
no pricing protocol; v15's "no total is reachable" tested two key *names*.

**M4-6 + §1a.** `PathLaw` is a real discriminated union; `WeightAtOffset` gives
schedules their support; `KernelCoefficient` carries SECONDS; `c(r)` is the
route-agreement ratio. And **"diagnostic" means "not a probability input", not
"not operational"**: §1a maps every consumer to its source, and the four *shape*
consumers read **route A's own horizon profile**, so route B feeds no control.

**Verify:** `python3 live/pm_research/sigma_kernels.py --selftest` (45 checks) ·
`contract_check.py --selftest` · `contract_check.py HEAD WORKTREE`.

### Historical — the pre-measurement directive (now executed)

This directive produced `route_a_v1` and is retained as provenance. Four review
rounds had each found a real defect, and the pattern is worth
naming: **each error was the previous error one level of abstraction higher.**
v1 used a lagging anchor; v2 replaced it with a trend extrapolator and buried the
bias in the variance; v3 named the bias but kept two incompatible estimators; v4
chose one route in prose and contradicted it in code. Every one was caught by an
adversarial probe rather than by a test I had written. Nothing further is a
specification task.

1. **Phase 0A 6 — FIT THE ROUTE-A LAW.** Regress observed `x_T` on observed
   `(S30, S60)` per horizon and per symbol, cross-fitted, day-blocked, embargoed;
   emit `GateEvidence` with an effect size and tolerance, not a bare p-value. Do
   **not** estimate `Ω` on this route — it is inside the residual.
2. **Phase 0A 5 — S30/S60 semantics** against the 1 s Binance tape. Gates route B
   only, and may run in parallel; it must not block a valid route-A fit.

Estimator implementation and any probability-level use remain on **HOLD** until
the repeated step-1 run has enough OOS days and produces a law whose gates all
read `PASS`.

### Historical — Revision 3 review and ITER3 application

Revision 3 gets the central mathematics right: under the declared Brownian
fixture, `alpha*=2700/1801`, the conditional anchor variance is `8.2590 sigma²`,
and the `2/-1` extrapolator's `9.5139 sigma²` is unconditional MSE containing a
known squared-bias term. The false ordered bracket, hidden nugget and several v13
carrier defects are also repaired. Keep those changes.

The iter-3 review found six remaining integration blockers:

1. Direct regression on `(S30,S60)` estimates a reduced-form mean and total
   residual, while the plan separately adds structural `k`, `v` and `Omega`.
   Choose one route; combining them double-counts forecast error.
2. The fixture compares every supplied empirical `alpha` with the Brownian
   `alpha_star`, labels it biased, and pulls the corrected mean back to the
   Brownian fixture. It cannot express the empirical anchor the plan requires.
3. The plan/contract type `Omega` in bps², but the code treats it as a multiple
   of `sigma²`. With `sigma²=4`, an identity covariance contributes 9.9867 bps²
   instead of 2.4967. Non-PSD inputs can produce negative total variance.
4. The default convention is `UNVERIFIED`, yet `settlement_var` returns a number
   and discards the status. The “weight schedule” cannot represent arbitrary
   temporal weights/support, and negative rates are accepted.
5. v14 carries request timestamps but does not enforce request/law instrument,
   target, horizon, knowledge or link equality. Checker green is structural.
6. Canonical guidance still conflicts on whether semantics gates the anchor,
   whether alpha is estimated or assumed, and whether regression residuals are
   the whole law. Tracking also retained the overclaim that `alpha=2` “fixes” it.

Shipped tests pass (kernel 24, checker 13, v13→v14 migration clean), but the
adversarial cases above fail semantically. Full evidence and acceptance tests are
in `SIGMA_PLAN_REVIEW_ITER3.md`.

### ITER3 APPLIED — Revision 4 / contracts v15 / fixture rewritten

Every ITER3 probe was **reproduced before acting**: pricing under `UNVERIFIED`,
`bias_coeff 0.200833` against an empirical `α`, the 4× `Ω` unit error
(9.9867 vs 2.4967), negative rate → −4.691, non-PSD → −120.905, the one-slot
`Unavailable`, the `KeyError` on an unknown convention, and the
`1799/1200`-vs-`2700/1801` contradiction inside my own file header. All correct.

**M3-1 — THE ROUTE DECISION, and it scopes everything else.**

|  | **Route A — reduced form** | **Route B — structural** |
|---|---|---|
| object | fitted law of `x_T` on `(S30, S60)` | `σ²k_law + σ²v(r) + uᵀΩu` |
| needs sampling semantics? | **no** | **yes** |
| identifies `Ω`? | **no** — it is *inside* the residual | **yes** — the lag-0 nugget |
| delivers | a **pricing** law | the **decomposition** |
| status | **PRICES** | **DIAGNOSTIC ONLY** |

**Route A prices; Route B diagnoses; they are never summed** (`R-ROUTE`,
`PathLaw.estimand_route`). The consumer matrix decides: the only LEVEL consumer
is the BE-Belief fallback, which needs `Σ(r)`, not its parts. `c(r)` is
redefined as the *agreement* between routes, `Σ̂_A/model_total_B ≈ 1` — a
model-adequacy diagnostic, not a term in either.

**And OLS is not a free lunch.** It gives the best *linear projection* and a
*pooled* residual — the conditional mean only if that mean is linear, the
conditional variance only under homoskedasticity. Otherwise it is an
unconditional forecast MSE, which is the same category error we removed from the
Brownian variance line one revision earlier, one level up. So route A ships with
**gates**: cross-fitting, ≥10 day clusters, a residual conditional-mean test and
a heteroskedasticity test. `pricing_var()` refuses if any fails.

**`Ω`'s identification has an answer** (§9-2a): contemporaneous moments give **3
numbers for 4 unknowns**, so it is *not* identified from them. Under route B it
is the lag-0 discontinuity of the bivariate cross-variogram — the **nugget**,
already in the per-symbol table — which needs a VERIFIED convention and is
entangled with `ŵ = 47 s`. **`Ω`, the nugget and `ŵ` are one problem, not three.**

Also applied: **M3-2** `AnchorSpec.selected` (MODEL|ESTIMATED), horizon-indexed,
bias measured against the *selected* estimand so a fitted `α` is unbiased with
respect to itself, `model_gap` kept as a diagnostic, `conditional_mean`
implemented. **M3-3** `Ω` is bps² once, PSD-validated, `RateQuantity` separates
bps²/s from terminal bps². **M3-4** fail-closed on status, rates, PSD, unknown
conventions; `Unavailable{reason, since, cause}`; conventions are `(offset,
weight)` schedules. **M3-5** `check_request` **evaluates** the comparisons with 8
negative fixtures — a typed timestamp that is never compared is documentation.
**M3-6** the scan now covers plan, code, contracts, STATUS and HANDOFF.

**Verify:** `python3 live/pm_research/sigma_kernels.py --selftest` (41 checks) ·
`contract_check.py --selftest` · `contract_check.py HEAD WORKTREE`.

**Revision 4's proposed next steps (superseded by the iteration-4 boundary
review above):**
1. **Phase 0A 5 — verify S30/S60 semantics** against the 1 s Binance tape. Gates
   route B entirely; does **not** gate route A.
2. **Phase 0A 6 — fit the route-A law**: regress `x_T` on `(S30, S60)` per
   horizon and symbol, cross-fitted and day-blocked, and report both residual
   diagnostics. Do **not** estimate `Ω` on this route.

Estimator implementation remains on **HOLD**.

## Then

- **G-FF4 queue bracket** (`plans/BE_FLOWANDFILLS_PLAN.md`) — potentially
  **programme-ending**. If the bracket on `Q_ahead` is wide enough, MM
  profitability is not knowable from data we can collect. Effort saved by D3
  goes here.
- Structure review loop is at 12 iterations and **not converged**
  (LOCAL 11 / SPREADING 1 / STRUCTURAL 1). Iteration 12 produced *zero*
  documentation change — it was a checker patch labelled as an architecture
  version. Semantic rules/producers/ports, scenario type consistency,
  incentive-to-outcome wiring, decision-snapshot equality and ModuleManifest
  coverage all remain unresolved.
- `pm_backfill.py` — historical windows. Gamma canNOT resolve these; enumerate
  via paginated CLOB `GET /markets`. Note calibration may **never** cross the
  2026-08-07 rule change (snapshot → 60 s TWAP, after the Stanford/SMU
  manipulation study).

## Standing rules (each one paid for)

- No design decision that a measurement on existing data could settle may be
  recorded as settled until that measurement is run.
- Read book state from `price_change.best_bid/ask`, **never** `book` snapshots.
- Read everything at knowledge time (`recv_ns`), never payload timestamps.
- Dedup prices by `(timestamp, symbol)`, raw by message identity — `recv_ns`
  differs per process so exact-line dedup does **not** catch a duplicate
  collector. Check with `ps -eo pid,etimes,cmd | grep live/pm_research`;
  pgrep patterns must include the `pm_research/` path segment.
- `fee_rate_bps` must be read from trades, not assumed. Currently 0 on every
  trade observed, which conflicts with both the docs (7%) and CLOB `base_fee`.

## Watch out for

- `resolutions.jsonl` holds 8 garbage rows from the first hour — filter through
  `is_final`, dedupe by slug keeping the final row.
- Thin windows exist (a BTC window with `volumeNum = 2`). Stratify by volume;
  never average across dead and live windows.
- The tick regime changes away from the money (0.01 → 0.001), which makes the
  tick a first-order economic parameter far from ATM.
- `PM_THEORY_CHECK_ORCHESTRATOR.md` §2's claim that "MM-under-obligation theory
  doesn't exist" was a **search failure** — the body is principal–agent MM
  contracts (El Euch et al.; Baldacci et al.). Marked superseded; don't cite it.
- Trading access (US status, CLOB auth) is a deployment question, out of scope
  until the gates pass. Don't let it leak into research design.
