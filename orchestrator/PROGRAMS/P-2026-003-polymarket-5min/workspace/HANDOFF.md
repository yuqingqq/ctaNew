# HANDOFF — P-2026-003 Polymarket Crypto 5-min

Updated: 2026-08-21, flow-and-fills Revision 4 development lane. All work is on branch
**`mm-research`**; nothing is on `main`. Sigma remains **Revision 5 / PRICING
HOLD**, while the offline measurement stack is complete through contracts
**v22**. `route_a_v1` has one OOS test day; per-symbol `route_a_v2` is
pre-registered and begins primary evaluation on 2026-08-22. Neither is
authorized for probability-level use.

## Read this first

The sigma **engineering pipeline is finished and frozen**. Collectors are
supervised; the all-coin Tier-1 batch, knowledge-time leak canary, model-free
Tier-2 terminal markout and fixed-grid calibration scaffold are immutable,
resumable and commit-last. Hourly user timers run the measurement lane at minute
20 and Tier-2 at minute 40. At this checkpoint both correctly returned `IDLE`:
2026-08-20 is not eligible until the adjacent 2026-08-21 UTC day closes. No
partial smoke is a research result.

The empirical state has not changed: `route_a_v1` has one of ten required OOS
days and all 84 gates are `INSUFFICIENT_EVIDENCE`. `route_a_v2` is per symbol
and horizon, uses signed-x conditional variances with no cross-instrument
pooling, and needs all 126 gates to pass on primary days from 2026-08-22.
Pre-freeze rows are design/training data, not fresh validation evidence. Leave
sigma code untouched while days accrue and continue independent mechanism work.

Reading order:
1. `live/pm_research/PM_ARCHITECTURE.md` — the entry point; explanatory structure.
2. `live/pm_research/contracts/contracts.yaml` (**v22**) — machine-readable
   source of truth for types. The prose defers to this file, not the
   other way round.
3. `live/pm_research/MEASUREMENT_PIPELINE.md` and `EVALUATION_PIPELINE.md` —
   current Tier-1/Tier-2 runbooks and claim boundaries.
4. `live/pm_research/SIGMA_ROUTE_A_RESULTS_2026-08-20.md` — the measured,
   strictly-forward Route-A result and current verdict.
5. `live/pm_research/SIGMA_ROUTE_A_PROTOCOL.md` — protocol frozen before fit;
   includes the non-analytic post-run embargo-wording erratum.
6. `live/pm_research/SIGMA_ROUTE_A_V2_PROTOCOL.md` — pre-registered
   conditional-variance successor; evaluation begins 2026-08-22 and no v2 fit
   exists yet.
7. `live/pm_research/GFF1_RESULTS.md` — frozen v3 side-convention PASS evidence.
8. `live/pm_research/EXP_RESULTS_2026-08-20.md` — earlier model results.
9. `live/pm_research/SIGMA_PLAN.md` — **REVISION 5, canonical.** One consumer
   matrix, one PRICING law (route A) and one DIAGNOSTIC decomposition (route B),
   never summed, now enforced as a TYPE boundary. **Read §2.3 then §1a** — the
   route decision scopes everything, and §1a says where each consumer's number
   actually comes from. v1/v2 text is in git history.
10. `live/pm_research/SIGMA_PLAN_REVIEW.md` — first implementation-readiness review.
11. `live/pm_research/SIGMA_PLAN_REVIEW_ITER2.md` — review of Revision 2.
12. `live/pm_research/SIGMA_PLAN_REVIEW_ITER3.md` — review of Revision 3 and v14;
   historical input to Revision 4.
13. `live/pm_research/SIGMA_PLAN_REVIEW_ITER4.md` — review of Revision 4/v15; its
   six items are applied in Revision 5/v16.
14. `live/pm_research/SIGMA_PLAN_REVIEW_ITER5.md` — pre-measurement verdict:
   MEASUREMENT GO / PRICING HOLD**, plus the frozen fit sequence.
15. `live/pm_research/sigma_kernels.py` — executable model **fixture**, not a
   frozen spec. `--selftest` checks exact arithmetic under a **declared and
   still UNVERIFIED** sampling convention; it does not establish that convention
   against the Chainlink streams.
16. `live/pm_research/plans/BE_FLOWANDFILLS_MODEL_PLAN.md` — **flow-and-fills
   Revision 4, canonical and frozen**; per-coin marked flow, execution reach,
   observable queue bounds, and separate development/validation states.
17. `live/pm_research/FLOW_MODEL_PROTOCOL_V4.yaml` — machine-readable freeze;
   development fitting is allowed now while promotion still requires 10
   complete forward UTC days per coin.
18. `live/pm_research/FLOW_FILL_DEVELOPMENT_RESULTS.md` and
   `flow_fill_development.py` — two-hour B0–B3/mark/Hawkes/fill development run;
   explicitly not decision eligible.
19. `live/pm_research/FLOW_INTENSITY_RESULTS.md` and `flow_intensity.py` —
   corrected same-state descriptive `f_r`/`f_p` evidence and executable guards.
20. `live/pm_research/plans/` — BE_BELIEF, MEASUREMENT, PRELIMINARY, and
   historical plan inputs.

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

### Post-close-out implementation 2026-08-21 — completed through Tier-2

**`route_a_v2` is now frozen before its evaluation data.**
`SIGMA_ROUTE_A_V2_PROTOCOL.md` (SHA-256
`c75fd12e74e8400f3761111028a14f75ddc6ae6e2629dd7fc13d1cf5e116456a`)
keeps the v1 mean and replaces only pooled residual variance with three signed-x
tercile variances estimated from historical strictly-forward residuals, with a
fixed 30-row shrinkage weight. Primary evaluation begins 2026-08-22. It needs
10 future evaluation days, 30 rows per cell and 126 passes: conditional mean,
conditional-variance calibration and paired Gaussian quasi-score for every
symbol/horizon. It is **PRE-REGISTERED / NOT FITTED / PRICING HOLD**.

**The Route-A selection ledger is now executable.** Commit `d8a8481` first
recorded excluded-window identity and an S60 range proxy. The current extension
normalises the audit unit to one `(slug,horizon)` candidate, expands window-wide
failures across all six horizons, records both accepted and excluded rows, and
joins separately hashed price-gap cause/version intervals without allowing them
to affect eligibility. A temporary full run over 2,943 final resolutions emitted
17,658 unique candidate keys (exactly six per resolution), 14,644 accepted and
3,014 excluded, with zero duplicates. Those counts include the incomplete
2026-08-21 day and are verification figures, **not a new v1 result**.

**The post-sigma measurement foundation is complete.** `da_state.py`,
`tier1_pipeline.py`, `coverage_ledger.py`, `replay_canary.py`,
`daily_pipeline.py` and `measurement_batch.py` implement the closed-day Tier-1
DAG. All requested coins are preflighted before writes; partitions are
code/schema/input-addressed, immutable and merge-never-overwrite; coverage facts
remain separate from the frozen admissibility rule; the cross-coin receipt is
published only after exact validation. Interrupted valid staging is reusable
but never mistaken for completion.

`evaluation_pipeline.py` implements Tier-2. It requires a complete `full`
Tier-1 receipt and the exact frozen G-FF1 v3 PASS artifact. It emits one
model-free gross terminal maker observation per parent trade and exactly one
calibration row per `(slug,r_s)` on the nine-point frozen grid. Knowledge-time
quote selection matches `StateView`; invalid and unavailable states remain
named rows instead of disappearing. Every markout summary carries both per-fill
and share-weighted estimates per coin and phase. With too few day clusters,
manifests explicitly say `DESCRIPTIVE_POINT_ESTIMATE` and CI unavailable.

Contracts **v20** add R-BATCH, R-DERIVE, R-GROSS, R-DUAL and R-ONEROW plus the
batch/evaluation carrier types and orchestration modules. The two user timers
are installed and active. Their first 2026-08-21 invocations returned `IDLE`
because the adjacent day was still open. That is the expected readiness gate,
not a failure. Partial smoke counts remain wiring evidence only; the first real
all-coin receipts will be created automatically after the boundary closes.

### G-FF1 v3 passed — `side` is the taker's

The frozen v3 run is **PASS**: agreement **600/600 = 1.0000**, Wilson95
**[0.9936, 1.0000]**, with every one of seven coins and five moneyness buckets
perfect. There are zero excluded rows. The side-evidence artifact and SHA are
mandatory Tier-2 inputs, so terminal markout cannot run on an assumed sign.

**The exchange ABI in the older docs does not apply to this tape.**
`0xe111180000d2663c0091e4f400237545b87b996b` replaces `(makerAssetId,
takerAssetId)` with one asset id plus an explicit `uint8` side. Signatures were
verified by keccak-256, not taken from a lookup:
`OrderFilled(bytes32,address,address,uint8,uint256,uint256,uint256,uint256,bytes32,bytes32)`
and `OrdersMatched(bytes32,address,uint8,uint256,uint256,uint256)`.

**`fee_rate_bps = 0` is a websocket artefact, and the fee schedule is now
MEASURED.** `taker fee = 0.07·p·(1−p)` $/share, matching to four decimals on
**600** transactions across the full moneyness range — 1.75 ¢/share at `p=0.5`,
0.63 ¢/share at `p=0.1`. **The taker pays and the maker does not**: 600/600
taker legs carry a fee, **744/754 maker legs carry zero**. `BE_FLOWANDFILLS_PLAN`
§12.1 had already derived the formula from a single transaction; this confirms
it at scale and adds the incidence, which was not previously established. The
Q5 reading `0.07·min(p,1−p)` (3.5 ¢/share) is **REFUTED at 2×**.

Consequence for the model, and it is not the comfortable one: makers pay no fee,
but a taker crossing at ATM pays ~0.50 ¢ half-spread + 1.75 ¢ fee = **~2.25
¢/share, about 225 bps on a $1 binary**. Nobody pays that casually, so every
maker fill is against a counterparty who expected more than 2.25 ¢ of move.
**The fee does not kill market making on cost; it loads the entire question onto
adverse selection.**

Still open: the maker **rebate** (zero fee is NOT evidence of a rebate paid, and
`OrderFilled.fee` is unsigned so a rebate could not appear in it), and **10 of
754 maker legs that DO carry a fee** — verified under an unambiguous test
(`taker == exchange address`), so this is a real residual class, not a
classification artefact. Mechanism unexplained.

(A tempting corollary — that fee presence explains the residual
mismatches — was tested and **refuted**: fee is present on validated and
mismatched rows alike, so it discriminates nothing.)

**Why `gff1_v1` is on the record as superseded.** v1 hard-coded the BUY reading
of the amount pair and returned `226/226 = 1.0000` — on a sample that was
**100 % BUY, zero SELL validated**, with a 0.548 mismatch rate. The frozen
mismatch ceiling is the only reason that one-sided result did not read as clean.
An order's `makerAmount` is what its creator *gives*, so the pair is ordered by
direction; under the BUY-only reading a SELL decodes to a price above 1, which
is impossible in a prediction market. 235 of the 274 mismatches reconciled
exactly under the inverted reading, which is what confirmed the diagnosis.
**Keep the guard discipline: it caught a defect that pooled agreement hid.**

**Open, characterised, unexplained:** 27 residual legs (5.4 %) where size
matches *exactly* and only price differs — the websocket price sits on the tick
grid (0.12, 0.65, 0.05) while the chain effective price does not (0.115862,
0.649168, 0.048008). Direction resolved for all 27 regardless. Not a direction
failure; a price-comparison artefact of unknown mechanism.

The v1/v2 failures remain useful provenance: v1 validated only BUY rows; v2 had
473 validated clusters and missed its frozen sample/mismatch guards. Neither is
retroactively called a pass. Full evidence is in `GFF1_RESULTS.md`.

### Flow-and-fills Revision 4 — development runs now; validation still waits

`plans/BE_FLOWANDFILLS_MODEL_PLAN.md` is now the only authoritative flow spec;
`FLOW_MODEL_SPEC_REV2.md` is explicitly historical. The machine-readable freeze
is `FLOW_MODEL_PROTOCOL_V4.yaml`, made before the primary period beginning
2026-08-22. Revision 4 separates `DEVELOPMENT` from `VALIDATED`: existing hours
may test the estimator and queue mapping now, while promotion still requires at
least 10 complete forward UTC days per coin.

Revision 4 retains the Revision 3 state corrections and closes the missing fill
seam:

- `lambda` is per-coin **event count intensity** in events/s. Side is a
  conditional mark, never the realized-next-side covariate in total intensity.
- Actual monetary mark is `size * native_execution_price`; USDC/s is derived as
  count intensity times the conditional monetary mark. It has no point-process
  compensator and never substitutes for an underpowered count model.
- `MICRO_002` and `MARKET` are labelled subprocesses. Their independence is a
  tested null after cause-specific baseline time changes, not a prerequisite
  for estimating ex-micro flow. If dependence exists, use a two-type model.
- Hawkes is an optional residual in baseline operational time. It is admitted
  only after residual clustering survives Holm correction on a complete B0–B3
  baseline and at least 10 forward days. Retention then requires forward NLL
  improvement, stable branching (`n<1` or spectral radius `<1`), and improved
  residual calibration.
- Execution price and size are marks. For an exact frozen shadow action they
  determine cumulative shares that reach the action level; unconditional
  arrival rate never determines notional or fills by itself.
- Public level-total L2 cannot identify exact queue position. Every action
  therefore returns the optimistic front and conservative trades-only
  back-displayed fill quantities. The midpoint is forbidden and collector-gap
  paths remain explicit unavailable rows.

The original `f_p` profile is **WITHDRAWN**. Its numerator used folded execution
price while its denominator used midpoint dwell, so it did not estimate one
conditional rate. `flow_intensity.py` schema v2 now uses the exact same
250 ms-lagged Up-midpoint intervals for arrivals and exposure; collector gaps
kill state until a new quote matures. All 31 semantic/control selftests pass.
On the corrected six-window design sample, execution price would have selected a
different bin for **6.9% of BTC** arrivals and **38.1% of HYPE** arrivals, which
shows the defect was material. The replacement shape is descriptive only and
is conditioned on `r` in the forward model.

`flow_fill_development.py` now runs the first executable lane on 24 consecutive
five-minute windows per coin (2026-08-20 17:45–19:45 UTC): 80,714 admitted
arrivals. Within-design held-window NLL says B1 beats B0 on all seven coins; B3
beats B2 on six, while B2 is mostly unsupported/neutral. The exploratory
operational-time Hawkes grid selects branching 0.40–0.55, but it resets each
market with no warm-up and is stamped `DEVELOPMENT`. At 15 seconds the 5-share
join-touch any-fill bracket is very wide: HYPE 71.3% front versus 2.4% back;
BTC 94.6% versus 76.9%. This is quantity evidence only—no fill-conditional
markout or P&L verdict. Full results are in `FLOW_FILL_DEVELOPMENT_RESULTS.md`.

Contracts **v22** retain R-FLOW and add R-FILL, `FlowAction`,
`QueuePositionBound`, `FillQuantityBound`, `FlowActionFillFit`, and a separate
non-decision `HawkesDevelopmentDiagnostic`. `BE-FlowAndFills` now requires a
`VALIDATED` action-fill artifact and remains unavailable.

**Next:** freeze and implement the conditional M1–M4 mark-law families, then
freeze the candidate code before scoring post-cutoff days. Continue accumulating
primary days in parallel. Do not promote the provisional Hawkes or fill
parameters; the 10-day minimum and forward gates still apply.

### Flow-model evidence audit trail — pre-Revision 3

Charter `live/pm_research/FLOW_UNCERTAINTY_LOOP.md`; plan
at that time (now superseded by Revision 4); probe `flow_uncertainty.py`. Coordinator
writes the decision rules, research agent runs the measurements — the split is
deliberate and has already stopped two rules being re-cut after their answers
were visible.

**The plan is a first-principles rebuild.** The old G-FF1..G-FF4 chain is
replaced. From the identity `net = half_spread + rebate − maker_fee − AS`, the
**sign is independent of queue position** — queue enters only via `E[N]` and the
conditioning inside `AS` — so the old chain put an unidentifiable quantity
(`Q_ahead`) ahead of a question that does not need it. New order: cost schedule →
sign → marginality → scale. `E[outcome − ℓ | fill]` needs **no fair-value model**,
which decouples this module from the 10-day sigma clock entirely.

**Closed so far:**

- **U1a `CLEARED`** — `size` is shares at 6 dp, exact against chain 600/600.
  **The volume layer is UNBLOCKED.**
- **U1b `CLEARED` / SINGLE-ACTOR — but the pooled share is a BTC ARTEFACT.**
  **One address**, present in all seven coins, at exactly 0.02 shares and
  **99.98% SELL**, carrying **0.0145% of notional**. 300-transaction unstratified
  draw: top-1 = 100%, distinct = 1, HHI = 1.0000.
  **CORRECTED 2026-08-21 by the intensity fit — the "16.3% of events" figure I
  recorded here is POOLED and hides a 45x range**, because btc is 64% of the
  pooled denominator:

  | coin | arrivals | micro share |
  |---|---:|---:|
  | btc | 270,404 | **2.0%** |
  | eth | 56,265 | 22.4% |
  | xrp | 24,107 | 59.9% |
  | bnb | 16,925 | 78.2% |
  | hype | 16,160 | **90.0%** |
  | pooled | 423,134 | 18.3% |

  **On btc the count layer is barely touched; on hype it is not contaminated by
  that actor, it largely IS that actor.** So R-DUAL is not a uniform reporting
  convention -- for thin coins the count layer is close to unusable and the
  notional weighting is the only meaningful one. This is the SIXTH instance of
  the denominator/population defect, and it was inside a number this file
  already carried as established.
  **THE INVERSION still holds:** raw-count intensity is contaminated by an
  economically empty class; notional-weighted intensity is not. Rule **R-DUAL**:
  every intensity AND every **signed** flow quantity (imbalance, side mix,
  signed volume) is reported both ways, exclusion published beside the retained
  set. Signed quantities are the fragile ones -- the contamination is ~100%
  one-signed and does not average out.
  **What the address is doing is NOT established and must not be narrated.**
- **U2 `CLEARED`** — tick composition: 0.001 exists only in the tails (6.75% at
  `p<0.15`, 6.73% at `p>=0.85`), **absent from the middle three buckets**. Where
  0.001 is available the spread is 1 tick in **99.9%** of quotes, so the 1-cent
  spread is a **CONSTRAINT, not a convention** — makers step inside the moment
  the venue allows. `γ_tick` is collinear with extreme moneyness and must be an
  interaction inside the tail buckets, never a main effect.
- **U3 + U3a `CLEARED` via the bound branch** — gap exposure concentrates at
  window **open** (31.7% of lost seconds in the first 30 s, 3.2x mean). KS
  occurrence was refused as **insufficient power**, not uniformity
  (`D=0.132, p=0.312`, min detectable `D=0.190` at n=51). Bound: **0.155% worst
  decile, 0.0488% overall**, `clob_v3_1` only. **That bounds EXPOSURE, not
  FLOW.** The earlier “long gaps are quiet” reading used a window-mean baseline
  against first-decile gaps and is withdrawn; phase-matched U9 does not support
  it.
- **U4 `CLEARED / REPLACED`** — the stale-book `+0.45 c/share` maker markout was
  ~2.6x too high. The model-free terminal identity gives in-window per-fill
  `+0.165 c` and share-weighted `+0.173 c`; after excluding the single-actor
  0.02 class, per-fill is `-0.211 c` while share weighting stays `+0.172 c`.
  **CORRECTED 2026-08-21 (U10/U10b) — THE SIGN IS UNDETERMINED, NOT NEGATIVE.**
  Window-clustered bootstrap, 931 windows, 10,000 resamples:

  | figure | estimate | 95% CI | |
  |---|---:|---|---|
  | per-fill, all flow | +0.165 | [-0.377, +0.734] | spans 0 |
  | per-fill, ex-0.02 | **-0.211** | **[-0.849, +0.457]** | **spans 0** |
  | 0.02 class alone | +1.987 | [+1.529, +2.440] | **excludes 0** |
  | share-wtd pooled | +0.173 | [-0.251, +0.596] | spans 0 |

  All seven per-coin CIs span zero on both weightings; the permutation test on
  coin labels is p=0.0482 but **names no coin**, so the surviving set is empty.
  **"On real flow, makers lose per fill" is NOT SUPPORTED and is withdrawn** --
  it was published in commit `6a0e593` and relayed as a finding. The only
  interval in the whole analysis that excludes zero is the **+1.987 against the
  single-actor class**, tight precisely because it is one counterparty behaving
  consistently, and carrying **~$91 of capacity over two days**: the sole
  statistically distinguishable maker edge here is the un-harvestable one.
  **What survives is the ESTIMATOR finding, which needs no interval:** the two
  weightings diverge in sign on the same fills, so a single-weighting spec would
  have reported "+0.165, makers profitable" and never revealed the dependence on
  one counterparty. Keep that apart from the economic claim -- conflating them is
  how `+0.45` survived two sessions.
  Window clustering misses day-level common factors, so these intervals
  **understate** uncertainty, and nothing on real flow excludes zero even so.
  **PROGRAMME STATE: cost settled bar rho; SIGN MEASURED AND UNDETERMINED;
  marginality and scale sit behind a sign that only more days resolve** -- so
  answering the queue question perfectly would still not say whether there is
  anything to harvest.
  Per-coin signs are mixed and two days permit no clustered CI.
- **U5/U7** — the rare maker fee is a thin per-address tier: 0/~10/50 bps.
  No in-transaction rebate appears in 600 receipts, but periodic/off-chain
  rebate remains `Unavailable`.
- **U6 `UNRESOLVED`** — cross-price chain-leg order is non-random but misses its
  frozen clearance bar: 49/63 = 0.778, Wilson95 [0.661, 0.863]. Same-price time
  priority, 59% of adjacent pairs, remains invisible; this cannot identify
  counterfactual `Q_ahead`.
- **U8 `CLEARED`** — spread is one tick on BTC/ETH but 3–7 ticks on the thinner
  coins. The pooled one-tick headline was BTC denominator dominance. Spread
  width is **not** an edge predictor: equal-width per-coin markout signs flip,
  consistent with wider spreads pricing adverse-selection risk.
  **WITHDRAWN (U10): that mechanism is NOT supported.** All seven per-coin CIs
  span zero, so scattered signs across spread widths is exactly what
  all-coins-near-zero plus sampling noise produces. Calling CIs-spanning-zero
  "signs flip" asserts structure the intervals deny. What survives is only the
  NEGATIVE result: **spread width does not predict edge.**

**Fee schedule — see `fee_structure_known`.** Taker pays `0.07·p(1−p)` $/share
(n=600, four decimals); maker pays zero on 744/754 legs. Crossing at ATM costs
~2.25 c/share (~225 bps), so the fee does not kill MM on cost — **it loads the
question onto adverse selection.**

**Two cross-cutting defects, both recorded with binding consequences:**

1. **The quote guard `0.0 < bid < ask < 1.0`** appeared **independently in both
   the coordinator's and the agent's code**, against the same tape, and excludes
   exactly the deep-tail quotes where the 0.001 tick lives. Caught once, only
   because 84 observed transitions contradicted a reported 0.00% share. Cost:
   124,772 quotes (5.2%), all from the tails. **Any quote filter must print its
   exclusion count beside its result.**
2. **`coin_msg_rate_hint` is a cumulative counter, not a rate**, despite the
   name (`collect_pm.py:489`); using it produced a spurious 3.26x. **Confirm any
   collector telemetry field against its definition in the collector source
   before use — the name is not the definition.** A real rate exists in the
   heartbeat `rate_msg_s`.

**Open:** U9 remains `UNRESOLVED` at seven phase-matched `PING_TIMEOUT` gaps;
five more in one collector era are required. The checked-in uncertainty ledger
and reproducible U1–U9 probes are at commits `6a0e593` and `6e125dc`.

### Queue and type tests — 2026-08-21. C1 CLOSES A LEAD STRUCTURALLY.

Protocol `QUEUE_AND_TYPE_PROTOCOL.md` (frozen before measurement), probe
`queue_and_type.py` (34 checks), results `QUEUE_AND_TYPE_RESULTS.md`.

**C1 — cancellations and the fill bracket: `UNIDENTIFIABLE`.** The coordinator
proposed crediting cancellations as an independent source that could narrow the
bracket. **It is not, and the reason is structural rather than empirical.**
Cancellation *volume* is abundant — saturation p50 2.0-13.2, with **86-99% of
actions saturated** — so crediting it collapses the pessimistic bound onto the
optimistic one (btc and eth agree to three decimals, 0.946/0.946 and
0.848/0.848). What displayed L2 withholds is cancellation **position**. Credit
all and you get FRONT; credit none and you get BACK_DISPLAYED; the interior
needs an ASSUMPTION, not a bound.
**THE BRACKET WIDTH IS THE QUEUE-POSITION AMBIGUITY RESTATED, and cancellation
data cannot reduce it because the missing quantity is the same one.**
Consequence is close to "fill is not determinable from data we can collect" —
but NOT for the expected reason: displayed depth ahead does not trade through,
it **churns**, and we cannot tell whose.

**Two defects in the coordinator's own rules, raised and upheld:**
- **R1 — the reconciliation gate was a TAUTOLOGY.** `cancelled` is definitionally
  the residual, so the identity balanced by construction, and a 1% threshold
  against gross churn running **60-1000x trade volume** could never fire. A gate
  that cannot fire is not a gate. Re-anchored to the independent
  `last_trade_price` stream, the residual against **traded volume** is
  **2.3-12.1%** (SOL one share in eight) — trade volume with no matching
  displayed decrease. Consistent with hidden liquidity or sequencing; not
  separable here and NOT narrated. Sixth instance of the denominator/population
  defect, inside a rule written to guard against exactly that class.
- **R2 — `MATERIAL` could not distinguish tightening from DEGENERATION.** The
  rule as written granted `MATERIAL` on a 97-100% width reduction; the agent
  **declined the win it was entitled to** and reported `UNIDENTIFIABLE`, because
  the bound had not tightened. Taking it would have published "cancellations
  narrow the bracket by 97%", which is false. A saturation guard is now required:
  the credited bound is a bound only where `cancelled_at_level < queue_ahead`,
  which holds in **1-14%** of actions.

**C2 — bivariate Hawkes on {MICRO_002, MARKET}: `RETAIN`. The coordinator's
motivating hypothesis is REFUTED.** The 2x2 branching diagonal dominates the
off-diagonal on every coin — `market<-market` 0.18-0.45 against cross terms
0.02-0.18 — so market self-excitation SURVIVES being modelled alongside the
micro actor, and the scalar 0.40-0.55 was not cross-excitation wearing a
self-excitation label. The Hawkes layer stays. A1 is not contradicted: cross
terms are non-zero, just smaller. Separately the micro actor is strongly
**self**-exciting (0.18-0.35 on five coins).
**Corrects a published number:** the scalar figure OVERSTATES market
self-excitation for most coins — only eth reaches 0.45, four sit at 0.18-0.35.
**Scope:** intervals are grid-quantised and conditioned on the selected
half-life, so they show fit STABILITY, not sampling uncertainty (btc's
degenerate [0.180, 0.180] proves it). `RETAIN` means not-deletable-on-this-
evidence, NOT a validated branching estimate.

**C2b — the instrument floor, and it blocks the obvious next step.**
Websocket-frame batching is **REFUTED**: 17.6% of btc market-market pairs fall
under 5 ms and 12.0% under 1 ms, but **not one shares a frame and not one has a
zero gap**, on any coin. However the sub-millisecond gaps pile up at **0-50 us
with a 26 us median**, which is **16.2x** the Poisson expectation. `recv_ns` is
stamped at PARSE time, so several messages arriving in one TCP segment are
stamped microseconds apart by processing cadence — distinct frames, distinct
timestamps, **no market information in the spacing**. The test rules out
batching at the websocket-message level and NOT at the transport level, and
26 us is more consistent with the latter.
**So neither branch is established for btc**: not a frame artefact, but its
grid-floor selection cannot be read as clustering either.
**DO NOT EXTEND THE HAWKES GRID LOWER — it would make this worse**, letting the
fit chase into the region where timestamps carry processing cadence rather than
arrival time. Prerequisite: establish a **timestamp-resolution floor** (the
shortest interval at which `recv_ns` differences reflect venue timing) and
truncate the grid there. Until then btc branching stays **CENSORED**.

### Residuals — open, in priority order

1. **Accumulate OOS days.** Tier-1/Tier-2 infrastructure is complete and the
   timers own catch-up. Do not turn partial partitions or design days into a
   result; rerun frozen v1 only at its formal boundary and evaluate v2 only on
   primary days from 2026-08-22.
2. **Finish the flow candidate while days accrue.** The B0–B3, mark-census,
   exploratory-Hawkes and queue-bound development path already runs under
   `FLOW_MODEL_PROTOCOL_V4.yaml`. Freeze/implement the conditional M1–M4 law
   families next, without treating the two-hour receipt as forward evidence.
   Hawkes and action fills remain unvalidated until the completed forward
   time-change and ten-day gates say otherwise.
3. **`PING_TIMEOUT` classification.** Phase-matched U9 is unresolved at n=7;
   retain `MNAR-suspect` and wait for five more same-era gaps before amendment.
4. **Phase 0A 5 — S30/S60 internal sampling semantics.** This still gates Route
   B only; the route-A fit remains descriptive until 10 OOS days.
5. **Downstream model work waits on data.** Tier-2 deliberately leaves sigma
   forecasts, walk-forward isotonic calibration and inferential intervals
   unavailable. Attach them only when the frozen day/fold requirements exist.

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

**Architecture v12 + machine-readable contracts v22.** Six planes, a structural
diff checker, version-bound migration records, and executable DA/EV pipeline
contracts through the commit-last Tier-2 boundary. Twelve external review
iterations. **Two of my own artefacts were
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

No manual normalization step remains. The supervised collectors and hourly
measurement/evaluation timers accumulate and materialize each eligible closed
day. Preserve `route_a_v1` and rerun it unchanged as immutable days accrue. A
formal verdict needs at least **10 OOS test-day clusters** and 30 OOS rows in
every frozen conditioning cell; because the first day is training-only, that
normally means at least 11 collected days. Do not respond to one-day point
effects by changing the cells, tolerances or functional form.

Keep `route_a_v2` separate: it is per symbol/horizon, begins primary evaluation
on 2026-08-22, and requires all 126 frozen gates. Rows through 2026-08-21 may
train a fold but may not contribute to its headline score or interval.

Phase 0A step 5 may proceed in parallel and gates Route B only. Do not add
structural `k/v/Omega` terms to the Route-A residual. Probability-level use and
the estimator integration remain on HOLD until all per-fit gates pass.

In parallel, continue the mechanism/flow work from the reproducible uncertainty
ledger. Keep cause/version gap facts, per-coin primary reporting and both
per-fill/notional weightings. The 1013 is measured as venue-side; the operative
path is the pre-registered cause-aware exclusion, not another client tuning
cycle. No two-day point estimate is an inferential profitability result.

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

- **Check the data before using it, every time — no lane is trusted by default.**
  Before any analysis reads a lane, verify *for the exact rows it will consume*:
  coverage and the gap ledger over each window, predictor staleness at each
  decision time, collector version / repair-era boundaries, and whether the lane
  is cleared for that class of inference (`clob_capture_clean`). Report the
  excluded set beside the retained one, and characterise it on the statistic the
  model actually estimates. Paid for four times: the FLB edge measured on p90
  6.2 s stale books; the v2 "repair successful" the collector's own ledger
  contradicted four minutes later; the prices lane called "clean" on
  `open_gaps=[]` while logging 58 gaps in 11 h; and the exclusion-MNAR reading
  that reversed sign once a variance statistic replaced a displacement one.
- No design decision that a measurement on existing data could settle may be
  recorded as settled until that measurement is run.
- Read book state from `price_change.best_bid/ask`, **never** `book` snapshots.
- Read everything at knowledge time (`recv_ns`), never payload timestamps.
- Dedup prices by `(timestamp, symbol)`, raw by message identity — `recv_ns`
  differs per process so exact-line dedup does **not** catch a duplicate
  collector. Check with `ps -eo pid,etimes,cmd | grep live/pm_research`;
  pgrep patterns must include the `pm_research/` path segment.
- **Read fees from the CHAIN, never from `fee_rate_bps`.** The websocket field
  is `"0"` on all 446,412 trade events and that is an **artefact**, not a zero
  fee. There is no three-way conflict any more: the docs' 7 % is correct, and
  `taker fee = 0.07·p·(1−p)` $/share is confirmed to four decimals on 600
  sampled transactions across the whole moneyness range. **Incidence: the taker
  pays and the maker does not** — 600/600 taker legs carry a fee, 744/754 maker
  legs carry zero. The `0.07·min(p,1−p)` reading (= 3.5 ¢/share at p=0.5) is
  **REFUTED**; it is 2× too large. Still unknown: the maker **rebate** (a zero
  fee is not evidence of a rebate, and `OrderFilled.fee` is unsigned so a rebate
  could not appear there), and the 10/754 maker legs that do carry a fee.

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
