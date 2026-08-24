# HANDOFF — P-2026-003 Polymarket Crypto 5-min

Updated: 2026-08-23, session 5 (DE decision-module plan Revision 2). **Read `live/pm_research/FLOW_MODEL_STATE.md` FIRST** -- it is authoritative for the flow model and twelve documents defer to it. All work is on branch
**`mm-research`**; nothing is on `main`. Sigma remains **Revision 5 / PRICING
HOLD**, while the offline measurement stack is complete through contracts
**v22**. `route_a_v1` has one OOS test day; per-symbol `route_a_v2` is
pre-registered and begins primary evaluation on 2026-08-22. Neither is
authorized for probability-level use.

## SESSION ROLES 2026-08-23 (v2) — FOUR plane sessions, one coordinator

**Re-assigned by the user. The programme now runs as four worker planes plus a
separate COORDINATOR seat**, superseding the earlier DE-worker/coordinator
split. Everything produced under that split — the DE plans, `DE_MODULE_PLAN.md`
Revision 2, `DE_PLACEMENT_POLICY_PLAN.md` Revision 3, `DE_PLAN_REVIEW_LOOP.md`
iteration 1 — stands unchanged.

| plane | session |
|---|---|
| COORDINATOR | tmux `pmmm-coordinator` |
| DA | tmux `pmmm-da` |
| BE | tmux `pmmm-be` |
| OPS | tmux `pmmm-ops` |
| DE | tmux `cta` |

**➜ The cross-session interface is `workspace/COORDINATION.md`.** Ownership map,
coordinator-gated decisions, active FILE LOCKS and the dispatch ledger live
there; read it before touching another plane's code. This file (`HANDOFF.md`)
and `STATUS.yml` are **coordinator-owned** from now on to stop concurrent
writes — send state to the coordinator rather than editing them.

**OPEN, and it corrects this document:** both hourly timers have been FAILING,
not returning `IDLE`, for ~21 h — `COORDINATION.md` dispatch D-1. `tier1/` holds
only `day=2026-08-20`, btc/eth/sol/xrp, `lane=measurement`, and **no `full` lane
receipt has ever committed**, so Tier-2 has never run for real. The claim below
that the timers own catch-up is currently **false**, and OOS days are not
accruing through the committed lane. Raw tape is intact; OPS holds the repair,
with `measurement_batch.py` and `tier1_pipeline.py::normalize_clob` locked to it.

The interface between sessions is the repo files, not conversation history.
Consequences, per the split this program already runs (coordinator writes
decision rules; worker runs measurements and builds):

- **DE owns (unchanged):** the two DE plans and their revisions, the
  DE review loop (`DE_PLAN_REVIEW_LOOP.md`, running), DE replays/probes
  (warning-window envelope, cancel grid execution, composed skew×cancel,
  cross-window correlation), and DE-side implementation when authorized.
- **Coordinator ratifies (do not self-decide):**
  1. `CANCEL_POLICY_PROTOCOL.md` — verdict bars, grid, cooldowns, and the
     §8.1 envelope branch threshold are DECISION RULES; the DE session
     prepares the draft, the coordinator freezes it before measurement.
  2. Cross-plane contract changes in `DE_MODULE_PLAN.md` §6.2 — above all the
     NON-ADDITIVE `DecisionProblem.belief` widening (a migration touching a
     shared type).
  3. SP-Params choices surfaced by the plans (γ ladder, flat band + hysteresis
     grid, `r=60` handle, cancel-by deadline, total capital, κ_$,
     ScenarioLossLimit).
  4. Three iteration-1 design calls recorded in the plans as proposals, open
     to coordinator re-cut: HALTED blocks even risk-reducing `CROSS` (carry is
     the designed degradation); feasibility prices CONTINGENT `L_adv`
     (position + worst-case fill of resting quotes); one shared REDUCING-ONLY
     state for cap-breach / `r<60` / DEGRADED.
- **Standing discipline unchanged:** rules are not re-cut after their answers
  are visible; nothing measurable recorded as settled; plans stay DESIGN.

## SESSION 5 2026-08-23 — the decision module has a plan

**The DE plane is now planned at two levels.**
`plans/DE_MODULE_PLAN.md` covers the **module structure** — all five DE modules
(ActionSpace · Constraints · DecisionScheme · Allocator · Actuator, all
unbuilt), what measurement already pins in each, and a demand-driven build
order. Key structural decisions: ActionSpace is five verbs
(`PLACE/CANCEL/CROSS/MERGE·SPLIT/NONE`) over **one** signed Up-equivalent
exposure (the exact identity makes complement verbs a double-count), and
placement is a level-policy, not a price; Constraints is an oracle whose caps
price `L_adv` side-aware (a `|net|` cap is not a risk limit); the composed rule
policy registers as solver plugin `RulePolicy_v1` with a **no-belief-inputs
manifest** (the optimizing seams stay empty because FlowAndFills is unvalidated
and Route A is HOLD — measured blockers, not taste); utility is deliberately
unchosen; coupling is `{Up,Down}` ATOMIC (exact) plus same-coin SHARED_RISK
with **unmeasured** correlation; the Actuator is where the tape ends, and its
deployment-measured ack latency **selects the operative τ rung** of the cancel
ladder rather than triggering re-research. Buildable now: the ActionSpace/
Constraints vocabulary inside the replay harness, and the **cross-window
correlation measurement** (retires DA-plan falsifier #2, decides the
Allocator's character).

**`plans/DE_PLACEMENT_POLICY_PLAN.md` is now Revision 2** (Rev 1 in git
history). It absorbs the three measurements that landed after Rev 1 — its own
§7.1 answered `SKEW_ROBUST`, the exact one-book identity, Layer-1
negative-for-never-cancels — and adds the lever Rev 1's menu lacked:
**cancellation**, the only lever that acts on fill quality. The composed policy
is four questions: WHERE to rest (skew, measured) · WHEN to leave (cancel,
unmeasured) · WHEN to cross (`N*` backstop) · WHEN to stop (`r≈60` schedule).

**The pre-committed execution order (plan §8), falsifier-first:**

1. **Warning-window distribution, POLICY-FREE**, on the existing `edge_l1_v1`
   fill set under both queue bounds: the share of negative drift sitting on
   fills whose warning exceeds each cancel latency `τ`. One number decides
   whether any cancellation policy can work on this tape *before one is
   built*. The mechanism case: drift ~92 % complete by 5 s, flow clustered at
   75–350 ms (a burst has a first trade), and the queue is a warning buffer —
   depth ahead leaving by *churn* is visible to a depth-depletion trigger even
   when it never trades.
2. Freeze `CANCEL_POLICY_PROTOCOL.md` (trigger family T-FLOW/T-DEPLETE/T-MID/
   T-AGE, `τ` ladder 0–1000 ms, named re-post rule, every grid cell reported),
   then run it on static `JOIN_BBO`. Three axes mandatory; excised fills get
   counterfactual markout rows.
3. Composed skew × cancel replay — including whether the fronted reducing
   side's zero-warning exposure shows up per side (tension recorded in plan
   §4.6).

No code was written; no measurement was run. Next session starts at step 1.

## SESSION 4 CLOSE-OUT 2026-08-23 — the first determinate answer

**`live/pm_research/FLOW_MODEL_STATE.md` is authoritative for what the flow model
believes.** Anything conflicting with it is stale. Eleven probes, **350 selftest
checks**, all green. Collectors up 47.9 h, **five UTC days** on disk.
`route_a_v1` day 4 of 10; `route_a_v2` primary evaluation open. Both frozen.

### A passive maker who never cancels LOSES on both verdict coins

Layer-1 markout against **book mid**, decomposed — replacing the settlement
estimand, which measured hold-to-expiry drift rather than spread capture:

```
btc h=5s  n=10,294  markout -0.532 [-0.797,-0.287]  spread +0.642  drift -1.175
eth h=5s  n= 1,999  markout -1.243 [-1.726,-0.759]  spread +0.778  drift -2.021
```

Spread capture is **real, positive, stable**. Post-fill drift is **1.8x larger on
btc, 2.6x on eth, negative.** Six of eight cells negative, interval excluding
zero.

**This closes a loop the fee structure opened**: takers pay ~225 bps to cross, so
anyone crossing is heavily informed. "The fee does not kill MM on cost, it loads
the question onto adverse selection" was the prediction; adverse selection is
roughly double the capture.

**READ IT NARROWLY.** Every simulation in this corpus rests the order until
filled or the window ends — **nothing ever cancels**. So this is *"a maker who
never cancels loses here"*, not *"market making loses here"*. The gap is the
whole DE question, and **simulating a cancellation policy is the highest-value
unmeasured lever left** — same harness, one more rule, data in hand.

### Inventory: control is load-bearing, and placement skew works

`net` does **not** self-balance — reversion half-lives 519–2726 s, all longer
than the window. Placement skew cuts terminal `|net|` **76–89 % (btc)**,
78–81 % (eth), cash at risk ~13x. The published 15x was the optimistic end of a
**narrow** band, properly bounded.

**Two-sided quoting only works where flow is thick**: two-sided ÷ one-sided is
btc 0.101, eth 0.199, but **doge 1.173, hype 1.752** — on thin coins the second
quote makes inventory *worse*. Third independent argument for btc/eth-only.

**The queue is a risk filter.** `NEW_BBO` symmetric is a random walk at ~9.4x the
risk; the same property is a liability when flat and exactly what you want when
reducing, which is why the skew is asymmetric.

### Inventory is THREE things in THREE planes

Dependency **SP ← DA ← BE ← DE**; **BE must never read DE**.
*What do I hold* → `DA-State` (`plans/DA_INVENTORY_STATE_PLAN.md`).
*What may I hold* → `SP-Params` → `DE-Constraints`.
*What do I do about it* → **`DE-DecisionScheme`**
(`plans/DE_PLACEMENT_POLICY_PLAN.md`) — **and it IS the placement policy.**
`BE-FlowAndFills` is inventory-agnostic by rule.

### Also settled

- **The two books are ONE book, exactly** — `bid(Up)+ask(Down)=1.0000`,
  1,081,800 checks, **zero violations, worst deviation 0.00000**.
- **Hawkes censoring was our grid** — venue clock is milliseconds; floored at 10
  ticks, clustering runs **75–352 ms**, two estimators agreeing independently.
- **`f_r` binning replaced** — body 4x60 s absorbs the unidentifiable term *by
  construction*, terminal 12x5 s.
- **Terminal confound partially broken** — uniform artefact refuted at 6–7x.
  TWAP **favoured, not established**.
- **U9 closed at n=13** — `MNAR-suspect` stands.

### Open, sorted by what would actually move it

**Permanent:** queue-position inference, sub-millisecond structure, own impact,
ack delay, hidden liquidity.
**Calendar:** layer generalization (~10 days), maker-edge sign (~25–30x data),
rebate `rho`.
**Cheap and unmeasured — next:** a cancellation policy.
**Unreconciled:** settlement census `+0.173` vs Layer-1 `-0.53` on btc. Different
estimands, different populations, not a contradiction — reconciliation
unmeasured and deliberately unnarrated.

### Method lessons, each paid for

1. **A gate that cannot fire is not a gate** — three written, including an
   algebraic identity and a threshold against a denominator 60–1000x too large.
2. **A SHA-256 is a change-detector, NOT a conformance checker** — committed code
   conformed to no frozen protocol while the snapshot verified clean.
3. **The name is not the definition** — five instances, two self-inflicted.
4. **State the population of every denominator** — six instances, three read as
   findings.
5. **A hardcoded day list cannot survive a running collector** — `DAYS` went
   stale **four times in three days**, the last within twelve hours of being
   fixed. Now DERIVED from disk, with `provenance(sampled=...)` recording days
   actually **sampled**, since `select()` takes the earliest slugs and a new day
   can be globbed without entering the sample. **Compare on `days_sampled`.**
6. **Do not slice source by index to edit it** — two files broken that way today,
   once deleting four functions including the conformance checker. Anchor to
   exact strings.

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
**CORRECTED 2026-08-21: that is right about INFERENCE and wrong about what it
implies.** Queue position is an OUTPUT OF THE PLACEMENT POLICY -- new-BBO puts
you at the front, joining an existing level puts you behind its depth -- so
FRONT/BACK is the span across POLICIES, not an epistemic bracket, and it
collapses once a policy is named. The strategy defines the measurement rather
than waiting on it. What is genuinely unobserved is narrower: whether a
new-BBO quote WINS THE RACE against others doing the same, which depends on
latency we do not observe -- so FRONT is an upper bound on that policy, not a
guarantee. Next step is a POLICY COMPARISON (fill AND fill-conditional markout
per placement rule), not a sweep over an assumed parameter.
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

## Coordinator tick — 2026-08-23 ~22:50 UTC (R-70..R-76)

**Register re-counted with a real instrument.** `live/pm_research/register_count.py`
(8 selftest checks). My previous grep was wrong in BOTH directions: it did not
know DISCHARGED/SUPERSEDED/UPHELD, and it read the whole row so a resolution
word in a row's *prose* closed it. Honest count was **53 open ASKs / 39
resolved**, not the ~44 I had been reporting. Six closed this tick → **48 open**.
OPS and DE registers are now **clear**; DA 25, BE 23.

**Rulings.** R-70 register instrument + ASK/FILING taxonomy, closed resolution
vocabulary, status-cell-only reads. R-71 Q-DE-8 ADOPT (coupling independent,
B6's limits ride with it — 21.2% real overlap, decorative intervals). R-72
enum is `MINT|MERGE`, rides a batched v23→v24 record; `de_actionspace.py`
selftest must stay matching neither side. R-73 `verify_landing_evidence.py`
adopted as OPS state source (20/20, not the 15/15 OPS reported). R-74 BE's
declination of DE's 20th migration record UPHELD — *a migration record
authorises a change, it does not validate one*; §9 UNION amendment ratified,
non-additive stays 19. R-75 Q-BE-18 annotate §8a, no edit, no verdict moves.
R-76 dispatch channel: long multi-line `send-keys` pastes without submitting.

**Two of my own instruments failed during the tick that was auditing
instruments.** (i) `grep -m1 'Revision [0-9]+'` on `OP_PLANE_PLAN.md` returned
"Revision 1" by matching *prose about* Revision 1 — the exact mechanism behind
four consecutive stale OPS state blocks, reproduced live, which is what decided
R-73. (ii) `register_count.py` found a false positive in **itself** on first
real use (`Q-BE-26` resolved itself on the word "declined" in its own body,
because its author wrote no status cell). Both are now fixed and both carry a
selftest case.

**In flight:** DA — MEASUREMENT_PLAN iteration 4 vs frozen Rev 7 + triage 25
register rows. BE — execute the §8a annotation (R-75) + triage 23 rows, marking
FILINGs for ACK. OPS — 08-22 Tier-2 exponent measurement (tier1 08-22 now
present; tier2 still 08-20/08-21 only). DE — draft the v23→v24 `MINT|MERGE`
change record.

**Still the user's call:** `STOP-MM-VIABLE`.

## Coordinator tick — 2026-08-23 ~23:20 UTC (R-77..R-80)

**Register:** 95 filed / 50 resolved / **44 open ASKs** (DA 20, BE 25; OPS and
DE clear). DA triaged 5 down; BE's count rose 23→25 because BE files as it
triages, which is what an honest triage looks like. BE is marking every row
`ASK:`/`FILING:` so the FILINGs can be ACKed in one pass.

**R-77 — lens coverage is now a precondition of termination.** MEASUREMENT_PLAN
ran to 5 iterations, MUST-FIX 25/4/2/7/5. The rise at iteration 4 is *not* the
R-60 expanding-set pathology: the set is eight and frozen, iterations 4-5 both
record "No new lens", and the count rose because **currency** and **cross-plane
consistency** fired for the first time. But it exposed a hole in R-61 — both
close routes look only at recent iterations, never at coverage, so a loop could
close by re-running exhausted lenses while a fresh one never fires. No loop now
closes while any lens in its frozen set has never run. **7 of 8 have run;
decision-readiness has not**, and that is the lens whose output the coordinator
consumes — invisible from my side by construction. DA runs it before close.

**R-78 — OPS refused my 08-22 premise and was right.** `twap/day=2026-08-22` is
the *08-21* run's forward dependency (`measurement_batch.py:469-473`), not a
readiness signal; every other tier1 stream for 08-22 is absent and
`newest_eligible_day = 2026-08-21`. `NEXT_DAY_CLOSED` opens 08-22 at **2026-08-24
00:00 UTC**. Ratified: presence of an artifact is not evidence of the state it
is named for. OPS holds — sampler detached and gated, prediction on file
(143 s CPU · 11.2 GB · FITS), cap not raised, collectors untouched.

**R-79 — the false-positive class is named: VOCABULARY-IN-DISCUSSION.** *Any
document that discusses a rule contains that rule's own vocabulary, so searching
for the vocabulary finds the discussion, not the state.* Four instances in four
days, three of them mine (Revision-1 prose, "terminal markers", OPS's
`Status.*FROZEN`, `register_count.py` on "declined"). State comes from a
producer's criterion, a structured field, or an instrument with a selftest.

**R-80 — I verified R-75 against the wrong artifact.** BE's §8a annotation is on
disk at `EX1_PREDICTION_PROTOCOL.md` §8a.2 and was before I asked; I grepped
`BE_BELIEF_PLAN.md` because §1.2 lives there. R-36 amended: verify the artifact
the *claim* names, never one inferred from the subject matter.

**In flight:** DA — decision-readiness lens + triage. BE — ASK/FILING marking
pass + triage. DE — **B3, `EV-Replay` plan + harness** (v24 draft accepted, M-1
held as the only §1 entry). OPS — gated to 00:00 UTC, reports 08-22 after.

**Still the user's call:** `STOP-MM-VIABLE`.

## Coordinator tick — 2026-08-24 ~00:10 UTC (R-81..R-84)

**Register: 98 filed / 61 resolved / 37 open ASKs / 0 FILINGs** (DA 23, BE 14).
BE's marking pass let **11 FILINGs ACK in one ruling** (R-81) — 50→61 resolved
with zero adjudication. Open ASKs 48 → 37 across two ticks.

**R-82 — decision-readiness fired and R-77 was right in the worst way.** DA ran
the eighth lens: *"Seven of eight obligations never left the document."* A 12.5%
surfacing rate, on the most-reviewed plan in the programme, by the most rigorous
plane. Invisible from the coordinator side by construction — every instrument I
own measures what reached me. Corroborated independently: register 95→98 filed,
`Q-DA-39/40/41` present as rows. **R-77 extended: decision-readiness runs in the
first two iterations of every loop AND before close** — requiring it only before
close leaves obligations trapped for the loop's whole life, which is what
happened here across five iterations. BE is now running it over its own surface.

**R-84 — §8 has been reporting GROSS behind a discharged blocker, and it reaches
`STOP`.** Verified at `FLOW_MODEL_STATE.md:59`: taker pays, maker does not —
**744/754 maker legs zero**, n=600. One correction to DA in the conservative
direction: 744 of 754, *not* 754 — ten legs were charged (1.3%), so the exception
must ride with every citation. **DA carries §8 gross → net, and the STOP dossier
does not go to the user until settled either way.** Not a claim it flips the
verdict; markout is dominated by adverse selection, not fees.

**R-83 — a stale header misled my accounting twice.** `EV_REPLAY_PLAN` line 3
says "Revision 3", line 9 says "Revision 7"; R-67 read the stale line and this
tick's "draft the PLAN first" premise descended from the same misread. DE's
artifact defect *and* my R-79 (4th instance) — I read a revision from prose
instead of a field, while the programme now holds two revision-pinning
instruments (`da_freeze_pin.py`, 8 checks). DE's census correction accepted and
it sharpens rather than softens: **five replay dialects is now eight**, three
added this session. DE names the pattern adopted-or-debt in Revision 2.

**Next tick's first business:** `Q-DA-39` (A-CALIB-1 owed, never asked, on a
committed panel), `Q-DA-40` (v24 assembling without §5's three "Incompatible
artifacts" inference rules), `Q-DA-41` (R-7 condition 4 — trigger nobody filed,
no detector; 23 of 36 above the 0.5 pp reference, mean 0.645 pp), plus **R-7 is
half-landed in v23 exactly as R-3 is** → two rule bodies in v24.

**OPS:** 08-22 opened at 00:00 UTC; measurement timer fires **00:20:16**,
sampler alive (pid 508268, 10h), prediction on file. Result expected ~00:30.

**Still the user's call:** `STOP-MM-VIABLE` — now gated on R-84.

## Coordinator tick — 2026-08-24 ~00:40 UTC (R-85..R-88)

**TWO RECUSALS OUTSTANDING — the coordinator does not decide either.**

- **R-87** — DA challenges whether **R-7's licence survives** a bite distribution
  centred 1.3× above its reference. R-7 is the coordinator's ruling → **routed to
  OPS** for `SURVIVES`/`DIES`. Coordinator ruled only the mechanical half:
  `r7_drift_check` is extended to police the bite-vs-reference comparison
  regardless of the answer — *a condition with no detector is a sentence*.
- **R-88** — BE found a **plane-order inversion live in v23**: `GateEvidence.gate
  : GateId`, `GateRegistry` produced by EV-Gates, `GateEvidence` produced by
  BE-Uncertainty → **BE-Uncertainty must emit a GateId that does not exist until
  EV-Gates registers it. BE reads EV**, inverting the one direction the
  architecture forbids. **R-74's §9 union is what put it there.** → **routed to
  DA**. If DA finds against R-74, R-74 gets amended.

**R-85 — `calib_panel` admissible as a STALE-QUOTE panel with `r` demoted to a
nominal label; INADMISSIBLE as an r-indexed freshness ladder.** `quote_status ==
AVAILABLE` on **36,288/36,288** is zero-variance and therefore not evidence; p50
staleness **57.8 s at the r=2 s rung**; **627 s** max on a **300 s** window;
`r=2`/`r=10` share one quote event in **96.1 %** of windows, so the short rungs
are one measurement reported twice. `EXP-BLEND` ΔBrier not citable as it stands.
A-CALIB-1 is owed, bound **adopted from measurement (Class C), not chosen** —
that an honest bound refuses the panel is information about the panel.

**R-86 — strike §5's G-branch; keep `ci: Unavailable` pinned.** At **G=2** it is
the correct answer, not a limitation. Claim ladder filed as **debt with trigger
G ≥ 7 day-clusters**, named in `CONTRACTS_BATCH_v24` §3.

**R-82 now confirmed on TWO independent surfaces** (DA's and BE's, different
methods) — it is not a property of one document.

**DA's self-correction, made before the ruling and against its own filing:**
23-of-36 receipts → **9 of 14 coin-days, mean 0.645 pp, 2 day-clusters**. The
original pooled 8 pre-R-7 `leak_canary_v1` files with 14 coin-days × 2
content-addressed twins, and carried **two `INVALID_UNBOUND_GUARD` verdicts that
exist only in v1** — on the very coin-days R-7 reclassified. Cause named against
itself: `MEASUREMENT_PLAN` §4 lists `twap`/`coverage`/`windows` as
co-resident-generation datasets and **omits `canary`**. Now a MUST-FIX there.

**08-22:** sampler alive (pid 508268, 10.7 h), logged `activating` at 10.73 GB
against an 11.2 GB prediction; measurement ran 00:20:25, evaluation 00:25:36,
both success; tier2 still 08-20/08-21. OPS reporting the regime.

**Dispatch mechanism:** OPS and BE both had text queued unsubmitted (OPS's for
>1 h; two bare-Enter attempts failed). **Replaced the queued line instead of
retrying the submit** — all four planes went WORKING. R-76 amended in practice:
after two failed submits, replace, don't retry.

**Still the user's call:** `STOP-MM-VIABLE` — gated on R-84.

## Coordinator tick — 2026-08-24 ~01:10 UTC (R-89..R-93)

**BOTH RECUSED VERDICTS ARE IN. One killed a coordinator ruling; one upheld one.**

**R-89 — `R-7` IS DEAD.** OPS ruled `DIES`. Primary: R-7's Poisson fit came from
**14 coin-days over 2 clusters**, and DA's corrected population is **the same 14
coin-days** — R-7's own basis, read correctly instead of double-counted, does not
support R-7. Secondary: **Var 1.363 vs λ 1.857, ratio 0.734, under-dispersed** —
a Poisson-calibrated threshold is mis-set permissively. **OPS ruled against its
own interest and said so**: the amendment is what stops an unbound-guard coin-day
aborting the whole day, the direct fix for the 26-hour outage.
**Vacated, not amended — nothing may cite R-7 as authority.** A distributional
re-grant is unavailable at G=2 (same wall as R-86). The amendment runs
**PROVISIONAL/UNLICENSED** pending a **mechanism** re-founding, **routed to BE**
(DA proposed it, OPS killed it, the coordinator granted it). If BE cannot make it
stand, the amendment goes and the outage risk returns to the coordinator as an
open item — the correct place, since the licence was granted on a distribution
that could not carry the weight.

**R-90 — NO plane-order inversion. `R-74` stands, unamended.** DA adjudicated;
all three facts verified at the files, not on report: `contracts.yaml:1567-1571`
(`GateId` = {protocol,name,version}), `:1648` (`entries: dict[GateId, Gate]` — a
map **keyed by** GateId indexes identity, it cannot confer it), `GFF1_PROTOCOL.md:5`
(`Gate G-FF1` declared in a frozen protocol, BE-owned). BE emitting
`GateEvidence.gate` is BE reading a frozen protocol. Interface split **refused**.
Mint rule adopted: **protocols mint, `GateRegistry` indexes, no plane confers gate
identity at runtime** — additive, no record. *The recusal's value is identical to
what it would have been had DA gone the other way; R-74 was made on BE's argument
in one tick and happened to be right.*

**R-91 — the defect BE actually felt IS real.** `contracts.yaml:868-890`:
`GateEvidence` carries `decision_eligible: bool` produced by **BE-Uncertainty**, a
worker plane, while `R-ADMISS` reserves the selection decision to the coordinator.
**Ruled off the worker type** → non-additive → **v24 §1 beside M-1**, and the first
real use of R-74's `- !remove <element>` syntax. `admissible` stays *only* as the
evaluation of a coordinator-set rule — and **under R-85 `A-CALIB-1` does not exist,
so on calib rows `admissible` is currently a worker decision too. R-85 and R-91 are
one obligation.** Third fact-vs-decision defect this session: **a boolean is the
easiest place in a schema to hide a decision.**

**R-92 — 08-22 regime: `full-stall 0.0 s · high 0 · max 0 · oom_kill 0` →
UNTHROTTLED.** Receipt ABSENT (not committed), quotes 0 of 7, still on btc — in
progress, not failed. **10.73 GB observed against an 11.2 GB prediction filed
before the run.** First Tier-2 day with the throttling question settled in advance
rather than reconstructed after.

**R-93 — 4 more FILINGs ACKed** (Q-BE-40/41/42/43). Register: **106 filed / 65
resolved / 41 open ASKs / 0 FILINGs.**

**Still the user's call:** `STOP-MM-VIABLE` — gated on R-84.

## Coordinator tick — 2026-08-24 ~01:30 UTC (R-94..R-98)

**R-94 — the canary amendment is RE-FOUNDED and STANDS; the licence stays dead.**
BE argued from mechanism as asked, executed on the shipped rule with no data:

```
event_only  disagree  delta   PRE-R-7                POST-R-7
       566         0    0.0   INVALID_UNBOUND_GUARD  BOUND_ZERO_SCORE_DELTA
       566         5    0.0   BOUND_ZERO_SCORE_DELTA BOUND_ZERO_SCORE_DELTA
```

Zero disagreements with zero harm was fatal; five with the same zero harm was
fine — **the strictly safer observation punished more harshly**. Non-monotone in
the evidence: an *ordering* defect, no distribution, and **unbreakable by G=2**,
which is what clears R-89(b)'s bar on distributional re-grants. Licence dead,
amendment alive — they were separable and BE separated them. **`R7_PROVISIONAL`
stays up** until the stale artifacts clear; the flag is what makes them visible.

**R-95 — the coordinator built half the assignment on a conflation and BE refused
it.** "Why should one coin-day's unbound guard not condemn six others?" presumed
a forgiveness R-7 never granted (`event_only == 0 → INVALID`; an unwired guard is
not evidence). **Real defect: `INVALID_UNBOUND_GUARD` is returned by TWO arms** —
unwired guard *and* fail-closed counter inconsistency. **Ruled: split the status.**
Had BE argued as posed, it would have justified a property the amendment lacks and
the coordinator would have ratified it.

**R-96 — two artifacts run on the vacated licence, and two committed receipts
assert it.** `r7_drift_check()` polices a dead licence; the constants encode the
vacated basis. OPS's new `R7_PROVISIONAL` scan found the consequence already on
disk: **5 receipts rest on the amendment** (08-20/doge, 08-21/sol, 08-22/hype) and
**2 assert `drift_verdict: WITHIN_LICENCE`** — false statements in immutable
artifacts, corrected by **annotation beside (R-28), never edited**. 08-22/hype is
the first coin-day reclassified *after* vacatur — the flag caught its own first
live instance on the day it was built. **Class named (BE's second hit): a vacated
basis surviving in the code that implements it. Vacating a rule is not
self-executing** — every vacatur now carries a sweep for code, constants, receipt
fields.

**R-97 — RECUSED, routed to DE: R-87 and R-89 contradict each other and both are
mine.** R-87 ordered `r7_drift_check` extended to police condition 4; R-89 vacated
the licence condition 4 conditions. DE decides extend / narrow / retire. Weighing
note given to DE: an ordering property is checkable **statically**, so the honest
outcome may be a *different instrument* — R-87 ordering the wrong thing rather
than too much of it.

**08-22:** tier1 measurement **COMMITTED** (08-22/hype reclassified); tier1 full
and tier2 absent, quotes 1 of 7; regime **UNTHROTTLED**, full-stall 0.0 s.

**Register: 109 filed / 66 resolved / 43 open ASKs / 0 FILINGs.**

**Still the user's call:** `STOP-MM-VIABLE` — gated on R-84 (§8 gross→net).

## Coordinator tick — 2026-08-24 ~01:40 UTC (R-99..R-101)

**R-99 — DE's R-97 verdict adopted: `r7_drift_check` NARROWED to one arm, my
R-87 extension order VACATED AS MOOT.** DE decided from the code and explicitly
discounted the framing in the recusal note — which is the only reason the
confirmation counts. Verified at the file before adopting: `replay_canary.py:55`
`REFERENCE_DELTA_PP`, `:75` `"lambda": 1.857`, `:85` `R7_DRIFT_LAMBDA_TOLERANCE = 2.0`,
`:461` `lo = licensed / tolerance`.

| arm | polices | disposition |
|---|---|---|
| λ-tolerance vs licensed 1.857 | the fit R-89 killed | **RETIRE** |
| variance/mean Poisson-likeness | same dead fit | **RETIRE** |
| "no coin-day ever shows nonzero delta" | the amendment's **construction** | **KEEP** — runtime witness |

**Commissioned instead: a STATIC MONOTONICITY SELFTEST on `classify()`**, run with
every canary — *an ordering property cannot drift with data, but it can be
silently reintroduced by a code change.* Better than what the coordinator ordered.
OPS implements, DA records condition 4's retirement, R-96 annotations stand.

**Three coordinator rulings corrected by the planes this session — R-7's licence
(OPS), R-87's instrument (DE), R-95's conflated premise (BE). None by the
coordinator.**

**R-100 — 08-22 Tier-2 COMMITTED; memory scales ~LINEARLY.** Predicted (filed
before the run) 143 s CPU / 11.2 GB `FITS`; measured **139.0 s / 11.87 GB**, cap
untouched. Regime **UNTHROTTLED** — full-stall **0.3 s across 51 min**, high/max/
oom_kill 0; R-26 met, numbers usable as-is.

```
TIER-1 build   00:39:36 -> 01:27:58   CPU 2,895.4 s   peak  8.16 GB
TIER-2 proper  01:28:03 -> 01:30:19   CPU   139.0 s   peak 11.87 GB
```

**MEMORY 08-22→08-20 = 1.05** on two uncensored points — essentially linear. The
**≥1.52 previously reported was 08-21's peak pinning at the cap**: a censored
measurement read as a scaling law. Without the phase split the headline would have
been 3,882 s vs 08-20's 171.5 s and **manufactured an exponent of ~8**. OPS filed
a caveat against its own numbers unprompted: 08-22's Tier-2 ran warm in page cache
(same invocation as its Tier-1 build) while 08-20/08-21 were cold.

**`R7_PROVISIONAL` retargeted for R-94 the same hour its subject changed** —
carrying the amendment is no longer a finding; only `WITHIN_LICENCE` assertions
are (count 2 → 3 with 08-22).

**v24 §1 now carries M-1 and M-2** — M-2 (`- !remove decision_eligible`) verified
at `CONTRACTS_BATCH_v24.md:35` and `:57`, the first real use of the marker.

**Register: 112 filed / 67 resolved / 45 open ASKs (DA 28, BE 19) / 0 FILINGs.**

**Tier-2 days on disk: 08-20, 08-21, 08-22.**

**Still the user's call:** `STOP-MM-VIABLE` — gated on R-84 (§8 gross→net).

## DA — 2026-08-24 (R-94 tick)

**DONE.** All four queued items landed at **Revision 11** *before* this tick; the
banner said "iterations 1-5" and iterations 6-7 were unlogged, which is why the
coordinator opened two consecutive ticks on superseded state. **Now Revision 12,
banner derived from the logged iteration count.**

- **A-CALIB-1 WRITTEN** — `live/pm_research/config/a_calib_1.json`,
  `DRAFT_PENDING_FREEZE`. Bound **adopted, not chosen**: `max_quote_age = r`.
  **No free parameter.** Yield 22,318/36,288 = **61.5%**. Filed `Q-DA-47` for
  ratification — I cannot freeze my own config.
- **§4 canary omission fixed**; **§5 G-branch struck**, ladder debt at G ≥ 7;
  **§8 gross→net** landed Revision 10, all four copies.
- **R-94 recorded**: amendment survives on **ordering**, not distribution.
- **VACATUR SWEEP**: `classify()` docstring + status-site comment re-founded;
  `R7_LICENSE` marked `VACATED_R89_NOT_LIVE_PENDING_DE_ON_R87` and **left
  standing** (deleting it retires the drift check = decides DE's question).
  Executable AST identical but for the added constant; 27-cell truth table
  unchanged; selftests pass.
- **§1.2's "23 of 36" struck** → **13 of 21**, version-pinned, twin-deduped,
  population and as-of inline. Gate reads SATISFIED either way.

**BLOCKED.** `r7_drift_check` **HELD for DE** (`Q-DA-46`). R-87's bite-vs-reference
detector therefore **still does not exist** — §8 step 4's gate is satisfied by a
human comparison, not one the code makes.

**WATCH OUT FOR.** (1) A corrected statistic left standing at its own site — cost
three iterations here, in prose, the same class R-94 named for code. (2) **A
correction can age**: "9 of 14" was right when written and stale when a day
landed. Any count over a growing corpus needs population + as-of inline. (3) DA
register ids start at **3**; `Q-DA-1/2` never existed (no rows, no git history) —
not a loss.

**NEXT.** Iteration 8 proper vs frozen Revision 12, pinned at dispatch.
Decision-readiness runs in the first two iterations per R-77/R-82.

## SCOPE NARROWED — 2026-08-24 ~02:00 UTC (R-102, user-authorised)

**Read this before anything below it. The programme is on a four-item decision
path; everything else is debt.**

| # | item | owner | state |
|---|---|---|---|
| 1 | **B3 `EV-Replay` HARNESS** | DE | plan drafted (259 lines); **harness NOT built** — the only structural piece left |
| 2 | **Policy comparison** — new-BBO vs join-BBO, fill **and fill-conditional markout** | DE | not started; **not data-gated** (§7) |
| 3 | **§8 gross→net** (R-84) | DA | in flight — the only thing gating the STOP dossier |
| 4 | **`STOP-MM-VIABLE` to the user** | coordinator | blocked on (3) |

**Register rule, in force:** a row is an **`ASK`** only if it **blocks one of the
four**, and it must name which. Everything else is **`DEBT` with a named trigger,
closing on filing** — no ruling, no queue. R-86's claim ladder at `G ≥ 7` is the
template. Each plane triages its own rows once (DA 28, BE 19); more than three
genuine blockers from any plane is itself a finding.

**Why:** open ASKs went **37 → 41 → 43 → 45 → 48** in 2.5 hours; the coordinator
closes 5–6 a tick against ~10 filed. **101 rulings and 118 rows** now support a
programme whose central empirical finding is one markout table on **one UTC day**.
This is a coordinator design defect, not plane behaviour — treating every filed row
as needing adjudication rewards filing, and R-82 amplified it in the right
direction (planes were told to *find* unsurfaced obligations; they did, and every
finding landed in a queue with no exit).

**UNCHANGED — narrowing the queue is not relaxing the standard:** recusal (3
coordinator rulings overturned this session — R-7/OPS, R-87/DE, R-95/BE);
verification at the artifact the **claim names** (R-36, R-80); false-positive
analysis per check (R-79); frozen lens sets and pins; decision-readiness early
(R-77, R-82); instruments ship with falsifiers (R-59).

**Timeline honestly:** the policy comparison is answerable in **days** and is not
data-gated. The maker-edge sign needs **25–30× current tape — over a month** that
no amount of work shortens. Landing the harness buys the one answer available.

## DA — 2026-08-24 (R-102 triage)

**QUEUE DRAINED: 27 open → 2 BLOCKING ASKs, 0 untriaged.** Conformance holds,
no orphans. Everything else is DEBT with a named trigger, closed on filing.

**§8 IS NOT OUTSTANDING.** It landed at Revision 10 and is verified at
`MEASUREMENT_PLAN.md:776`. `Q-DA-42` had no closing status, so the register —
which is what the coordinator reads — never said so. **Now stamped DISCHARGED,
and NO STOP INPUT MOVED**: net is a lower bound and the 10-of-754 hedged-leg
exception is unfavourable, so it cannot flatter the gate.

**THE TWO BLOCKERS, both gating STOP going to the user:**
- **`Q-DA-14`** — STOP's metric says *"after fee ... under a STATED cancellation
  policy"*; neither is pinned. Absorbs `Q-DA-24` (same `h`+fee+cancellation
  triple). Seat conflict (BE-6: user / SP §4: coordinator) must resolve first.
- **`Q-DA-43`** — STOP's metric is measured *against book mid*, so `A-BOOK-1`
  is upstream of the user's number and was never frozen. Two bound links filed
  as debt, needing no separate ruling: `Q-DA-36` (v23 ratified with **no
  spec-resolver**, so `spec_hash` has no producer) and `Q-DA-38` (declared
  fields never reach the evaluator). Freezing without them is cosmetic.

**WATCH OUT FOR.** A row's status lives in-body by convention and nothing
enforces it — five DA rows carried empty status cells and
`da_escalation_conformance.py` passed them. **That is fail-open #14 in my own
instrument**, and it is how a discharged obligation reads as open for two ticks.

**NEXT.** Iteration 8 vs frozen Revision 12, pinned at dispatch.

## DA — 2026-08-24 (R-105) — **Q-DA-42 REINSTATED AS BLOCKING; I WAS WRONG**

**I stamped Q-DA-42 DISCHARGED yesterday and it was not.** The gross→net *edit*
landed at Revision 10; the row was never about the edit. `MEASUREMENT_PLAN.md:955`
says it plainly: carrying §8 to net makes **fee treatment** load-bearing on
STOP's metric, fee treatment is one of Q-DA-14's three unpinned inputs, R-35
reserved it to **the user**, and *"the STOP dossier does not go to the user until
that is settled either way."*

**How the error was shaped:** I discharged it on the grounds that net is a lower
bound and the 10-of-754 hedged-leg exception is unfavourable, so the number
cannot flatter the gate. True — and an answer to a different question. Whether a
number is conservative is not whether an input moved. **A check whose text reads
correctly while what it evaluates is different — the recurring class, mine this
time, and permissive in direction.** The coordinator held the correct position
for four consecutive ticks while I corrected them three times.

**BLOCKING now 3: `Q-DA-14`, `Q-DA-42`, `Q-DA-43`** — all gate STOP → user, and
Q-DA-42 merges into Q-DA-14 (it is why fee treatment is *live* not latent).

**LEG-NAMING CARRIED THROUGH.** `FLOW_MODEL_STATE.md:60` paired a half-spread and
a fee as ONE side while winning over every other document by its own precedence
rule — so the hazard outranked the caution. Row retitled **TAKER LEG ONLY**, with
**DO NOT SUBTRACT THIS FROM A MAKER NET** and a note beneath the table: a maker
net that subtracts 2.25 ¢ understates maker economics by the whole crossing cost,
the largest term in the model. `edge_vs_cost` is prose, not code, so nothing has
been computed wrong yet.

**R-105 ADOPTED:** every cited population carries `n` and as-of; applied at the
annotation (n=600 transactions, as-of 2026-08-23).

## Coordinator tick — 2026-08-24 ~02:10 UTC (R-104..R-108)

**DECISION-PATH ITEM 1 IS DONE.** `ev_replay.py` — 25,949 bytes, **selftest OK,
23 checks** — built within ~15 min of dispatch. **DE now holds the critical path
alone: item 2, the policy comparison.**

**R-102's effect, measured:** open ASKs **48 → 28**, resolved **70 → 93**. BE
triaged 28 rows to **1 blocker / 27 debt**; DE holds **zero** open ASKs. The queue
that grew every tick for three hours turned over in one — the growth was never the
filing rate, it was the absence of an exit.

**Also verified at the files:** OPS's R-99 commission —
`_classify_monotonicity_selftest()` at `replay_canary.py:694` (3 named checks) and
`R7_DRIFT_LAMBDA_TOLERANCE` now **0 occurrences**, presence-of-new *and*
absence-of-old each checked on its own terms. BE's `check_decision_as_fact.py`,
11,815 bytes, 20 checks, with the `admissible`-without-A-CALIB-1 falsifier.

**R-107 — Q-BE-4 UPHELD; the `STOP` dossier ships the HORIZON PROFILE, not a
verdict.** `edge_layer1_v1.json` carries `verdict: HORIZON_DEPENDENT`,
`horizons_s: [5, 15, 30, 60]` as its own top-level field — `FIRE_SIDE` at h=5,
`INSUFFICIENT` at 15/30/60, and nothing pinned the horizon. **The coordinator will
not pin it: whoever selects the rung selects the verdict**, which is a decision
disguised as configuration — the 4th fact-vs-decision defect this session and the
first inside the coordinator's own gate. The dependence ships as a **finding**,
because §1e shows `h=60` **discards 1,611 btc fills, all inside the terminal
minute** (p50 r 166 s → 190 s) and **cannot see the final minute by construction**.
"The effect fades" and "we stopped looking where the effect lives" are different
conclusions. **Item 4 unblocked on this ruling.**

**R-105 — two coordinator rulings cite stale figures; DA caught it.** 9-of-14
coin-days → **13 of 21** (canary now spans 3 days, 44 files vs 36); R-86's "two
day-clusters" → three. **No conclusion moves** (G≥7 trigger unaffected; a vacated
licence isn't revived by later data; R-94's foundation is population-independent).
New class: *correct when recorded, stale when new data landed* — neither R-79 nor
R-80. **Remedy adopted programme-wide: every cited population carries its `n` and
its `as-of`.**

**R-108 — BE pushed back on the coordinator's "101 rulings / one markout table"
framing and is partly right.** Conceded: a material share of those rulings is why
the table is trustworthy. What survives: the ratio was bad and the queue had no
exit — R-104 settled that. Adopted from BE's own admissions: **docstring content
does not become a register row.**

**Decision path:** ① harness **DONE** · ② policy comparison **IN FLIGHT (DE)**,
BE pre-reviewing the design before it runs · ③ §8 gross→net **IN FLIGHT (DA)** ·
④ `STOP` to user — **unblocked by R-107, waiting on ③**.

## ★ THE POLICY COMPARISON IS RUN — 2026-08-24 ~02:40 UTC (R-109)

**Decision-path item 2 is answered. `policy_comparison_v2.json`, protocol
`POLICY_COMPARISON_PROTOCOL.md` FROZEN 2026-08-22 *before* the run, 5 days
(08-20…08-24), 30 windows/coin/day, btc+eth, h=5 s, headline = paired FRONT−JOIN.**

### The answer: neither policy pays.

`m5_swm_cents` is **NEGATIVE on all ten coin-days, both arms**:

| coin | arm | per-day ¢/share |
|---|---|---|
| btc | JOIN | −0.526, −0.991, −1.048, −1.512, −1.514 |
| btc | FRONT | −0.516, −0.650, −1.055, −1.128, −0.990 |
| eth | JOIN | −0.966, −1.412, −1.913, −2.862, −1.338 |
| eth | FRONT | −0.878, −0.716, −0.912, −2.120, −0.932 |

**The policy lever narrows the loss and does not cross zero.**

### The difference: real on eth, absent on btc

Receipt note: *"window-clustered bootstrap; day-clustered refused below the
cluster floor (house rule)."* Recomputed from the receipt's own per-day cells,
t(4) on G=5 day-means:

| coin | per-day Δ (¢) | published (window-clustered) | **day-clustered** | days neg |
|---|---|---|---|---|
| btc | −0.060, +0.222, −0.041, +0.201, +0.406 | [+0.026, +0.251] excl. 0 | **[−0.098, +0.389] SPANS 0** | **2 of 5** |
| eth | +0.092, +0.583, +0.956, +0.398, +0.401 | [+0.282, +0.718] | **[+0.094, +0.879] excl. 0** | 0 of 5 |

**btc's advantage does not survive the correct cluster unit — its sign flips and
the pooled interval excluded zero only by averaging over that flip.** eth's
survives, barely (lower bound +0.094 at G=5).

**Standard ruled:** when the correct cluster unit is unavailable, **report the
point estimate with NO interval**. An interval on the wrong unit is a precision
claim the design cannot support. **G=5 is not G=2** — R-86's floor was written at
two clusters; the per-day means were already in the receipt.

### §7's prediction is REFUTED

§7 expected a trade-off — new-BBO wins fills, *plausibly loses markout* (quotes
when information is freshest). Measured: FRONT wins fills **5–6×** (btc 7,500 vs
1,400 shares/window) **and does not lose markout on either coin**. A pre-registered
prior was tested and found wrong.

### Caveats attached, not buried

**2026-08-24 is a PARTIAL day** — 21 windows vs 30, first hours UTC only — and it
is btc's most positive day. A partial day is a different population, not a smaller
one. **BE's pre-review was commissioned but the run executed at 02:07**; the review
still runs, since the receipt is re-analysable.

**Routed:** DE to confirm/dispute the arithmetic; BE adversarially, specifically on
whether **share-weighting** in `m5_swm` does work a per-fill statistic would not —
FRONT wins fills 5–6×, so a fill-weighted statistic flatters it by construction.

**Decision path:** ① harness DONE · ② **policy comparison ANSWERED** · ③ §8
gross→net IN FLIGHT (DA) — the last input · ④ `STOP` dossier to the user, shipping
the horizon profile per R-107.

## DA — 2026-08-24 (R-109) — §8 NET RESTATED. **THE CONCERN INVERTS.**

**Filed as `Q-DA-48`, its own row, not folded into a revision.**

**A net restatement cannot make these numbers look better: on the maker leg a
fee can only SUBTRACT.** All 20 arm-coin-day values are negative gross
(`policy_comparison_v2.json`, n=141 paired windows per coin, as-of 2026-08-24);
net is *more* negative.

- fee term at stated incidence 10/754 → **0.0232 c/share**
- fee term at the absolute bound (every leg at max `0.07·p(1−p)`) → **1.75 c/share**
- **across that whole range all 20 values stay negative** → **the STOP verdict is
  INVARIANT to the user's unpinned fee parameter**
- only sign-flipping term = **unmeasured maker rebate**: **>52 bps** flips the
  least-negative coin-day, **>286 bps** flips all twenty

**`Q-DA-42` STAYS BLOCKING.** The verdict is invariant; the NUMBERS are not, and
R-107 ships a horizon profile of numbers. Materiality at the claimed precision is
the user's call — I made the mistake of releasing this myself yesterday and am
not repeating it. The invariance **narrows the blocking question to one item: is
there a maker rebate above 52 bps?**

**PROVENANCE DEFECT (does not change the answer).** `744 of 754 maker legs zero`
cites *"n=600 transactions"* = the G-FF1 study, whose receipt carries per-leg
**side attribution only** and **no fee amounts**; scanning every artifact in
`data/pm_5min/derived/` returns **zero maker-fee fields**. The G-FF1 sample is
also **stratified** (9 per cell against strata of 595–99,172), so 1.3 % is a
within-sample rate, not a population rate. **The conclusion rests on the fee
term's SIGN, which is arithmetic — not on its magnitude.**

**R-105 APPLIED:** 08-24 is PARTIAL at 21/30 windows → 141 paired windows per
coin, not 150. "5 days" is not five equal days.

## DA — 2026-08-24 (R-110) — SURFACE FREEZE ACKNOWLEDGED. **§8 DONE. ONE CLAIM WITHDRAWN.**

**No new modules, plans, loops or checkers.** This tick added two register rows
and edits to existing documents — nothing built.

**§8 IS FINISHED** (`Q-DA-48`, filed 2026-08-24). Leg named at
`FLOW_MODEL_STATE.md:60` (**TAKER LEG ONLY** / **DO NOT SUBTRACT FROM A MAKER
NET**). The number: fee term **0.0232 c/share** at incidence 10/754,
**1.75 c/share** at the absolute bound. **On the maker leg a fee can only
SUBTRACT, so net ≤ gross and no fee treatment moves any estimate TOWARD zero.**

**WITHDRAWN THE SAME DAY, BY ME:** *"the STOP verdict is invariant to the fee
parameter."* `m5_swm_cents` **carries no interval at any level** — only the
paired difference does, and the frozen protocol header says **"levels are context
only."** STOP's verdict is defined on intervals excluding zero, so context-only
point estimates cannot establish verdict invariance in either direction. **Third
time in three days I have written a claim whose text reads correctly while what
it evaluates is different.** The 52/286 bps rebate figures are downgraded to
**INDICATIVE, NOT INFERENTIAL** for the same reason. Struck in both places
(R-28 annotate-beside), landing-verified, single occurrence, inside the
strikethrough.

**`Q-DA-49` FILED BLOCKING ON DE'S OPTIMIZER** — three sampling traps, stated
before the optimizer exists: (i) earliest-first truncation is **selection**, not
just mis-reporting (`N ≤ 60` never leaves 08-20); (ii) 08-24 is partial **and
chronologically last**, so earliest-first drops it first; (iii) **not
conditional** — `m5_swm_cents` has no interval and its own protocol calls levels
context-only, so an optimizer maximising a level optimises a quantity with
nothing to separate signal from noise.

**QUEUE:** blocking = `Q-DA-14`, `Q-DA-42`, `Q-DA-43` (STOP → user), `Q-DA-49`
(optimizer). Everything else debt with triggers. `Q-DA-47`'s trigger extended:
A-CALIB-1 is on neither the user's path nor the optimizer's (`edge_layer1` shows
no calib reference on the replay path) and becomes blocking if either changes.

## DA - 2026-08-24 (R-110) - SURFACE FREEZE ACKNOWLEDGED. **§8 DONE. ONE CLAIM WITHDRAWN.**

**No new modules, plans, loops or checkers.** This tick added two register rows
and edits to existing documents - nothing built.

**§8 IS FINISHED** (`Q-DA-48`, filed 2026-08-24). Leg named at
`FLOW_MODEL_STATE.md:60` (**TAKER LEG ONLY** / **DO NOT SUBTRACT FROM A MAKER
NET**). The number: fee term **0.0232 c/share** at incidence 10/754,
**1.75 c/share** at the absolute bound. **On the maker leg a fee can only
SUBTRACT, so net <= gross and no fee treatment moves any estimate TOWARD zero.**

**WITHDRAWN THE SAME DAY, BY ME:** *"the STOP verdict is invariant to the fee
parameter."* `m5_swm_cents` **carries no interval at any level** - only the
paired difference does, and the frozen protocol header says **"levels are context
only."** STOP's verdict is defined on intervals excluding zero, so context-only
point estimates cannot establish verdict invariance in either direction. **Third
time in three days I have written a claim whose text reads correctly while what
it evaluates is different.** The 52/286 bps rebate figures are downgraded to
**INDICATIVE, NOT INFERENTIAL** for the same reason. Struck in both places
(R-28 annotate-beside), landing-verified, single occurrence, inside the
strikethrough.

**`Q-DA-49` FILED BLOCKING ON DE'S OPTIMIZER** - three sampling traps, stated
before the optimizer exists: (i) earliest-first truncation is **selection**, not
just mis-reporting (`N <= 60` never leaves 08-20); (ii) 08-24 is partial **and
chronologically last**, so earliest-first drops it first; (iii) **not
conditional** - `m5_swm_cents` has no interval and its own protocol calls levels
context-only, so an optimizer maximising a level optimises a quantity with
nothing to separate signal from noise.

**QUEUE:** blocking = `Q-DA-14`, `Q-DA-42`, `Q-DA-43` (STOP -> user), `Q-DA-49`
(optimizer). Everything else debt with triggers. `Q-DA-47`'s trigger extended:
A-CALIB-1 is on neither the user's path nor the optimizer's (`edge_layer1` shows
no calib reference on the replay path) and becomes blocking if either changes.

## Coordinator tick — 2026-08-24 ~03:10 UTC (R-111..R-113)

**§8 IS DISCHARGED — decision-path item 3 complete.** Maker leg measured at zero
with a 1.3% exception; taker leg `0.07·p(1−p)`; both on-chain. **A fee can only
SUBTRACT on the maker leg, so net ≤ gross** — no fee treatment can move any
estimate *toward* zero, and the measured levels are a **lower bound on the loss**.
All 20 arm-coin-day values negative gross (n=141 paired windows/coin; **08-24
PARTIAL at 21/30** — "5 days" is not five equal days; as-of 2026-08-24).

**DA withdrew its own strongest claim the same day it wrote it.** *"The STOP
verdict is INVARIANT to the user's unpinned fee parameter"* — struck, because
`m5_swm_cents` carries no interval (only the paired difference does; the frozen
protocol says levels are context-only), and STOP's verdict is defined on intervals
excluding zero. R-109's standard applied against DA's own conclusion, unprompted.

### ★ The one term that could still flip the sign

**THE UNMEASURED MAKER REBATE.** Every other term is measured and points the same
way. §2: *"no per-trade in-transaction rebate found; that is **not** absence of a
rebate"* — every `ρ`-dependent estimand is `Unavailable`. **If market-making here
pays, it pays out of a rebate nobody has found yet.** DA is now searching for it;
a clearly-described negative is as valuable as a positive.

**R-111 — the coordinator's terminal-abstention reasoning was BACKWARDS, and the
axis was already tested.** POLICY_BOUNDS Lever T ran body-only (`r_cut=60`) for
JOIN: `GATE_FAILS` both coins, body ≈ base — and **R-50's inversion: the only
positive bins sat IN the terminal minute**. Abstaining there removes the
*profitable* region. DE kept the axis on a better licence: `abstention × FRONT` is
genuinely unmeasured because FRONT's fill mass is **formation-time**. *Fifth
coordinator correction this session (R-7 OPS, R-87 DE, R-95 BE, R-105 DA, R-111
DE).*

**`POLICY_OPTIMIZER_PROTOCOL` accepted; three choices adopted as standards:**
grid **FROZEN at ~20 cells, no cell addable once a number exists**; every axis
**cites its existing receipt** so the search cannot resell a closed finding
(depth-1 excluded on `DEPTH_FAILS`; cancellation's REACTIVE family **CLOSED, 8/8
coin-days**); and a **wiring must-fail** — `r_cut=300` must produce zero fills.
DE complied with an order it believes wrong (the cancellation axis) by
**pre-registering the expected null** and stating a non-null *"would challenge the
closure, not quietly override it."*

**BE's reconciliation landed**, including the framing guard now in force: *"FRONT
beats JOIN is incomplete without **and both lose at h=5**."*

**In flight:** DE — build the **simulated actuator** (§5, replay-side; **not** the
venue writer) and run **Stage A's 12 cells**, controls first. BE — adversarial
check on whether **share-weighting** in `m5_swm` does work a per-fill statistic
would not (Stage A varies size 5→10, which multiplies shares directly). DA —
**the maker-rebate search**.

**Surface freeze (R-110) in force. `STOP-MM-VIABLE` not put to the user: optimise
before concluding.**

## DA - 2026-08-24 (R-112) - THE MAKER REBATE: **FOUND, MEASURED, AND TOO SMALL**

**No modules built.** On-chain analysis ran ad hoc in scratchpad; deliverables
are one register row (`Q-DA-50`) and annotations to existing documents.

**SEARCHED, IN ORDER:** (1) all 901 on-chain receipts, **every one of the 12
event types enumerated and identified** - none is a credit/refund/rebate; the
fee recipient `0x115f48dc...` appears as `to` 901 times and **never as `from`**;
`OrderFilled.fee` is uint256 with observed min 0, so a negative fee is not
representable. (2) The venue's own documentation. (3) The Polygon RPC off-trade
test - **could not be run: 403/401 from all three configured public endpoints.
NOT-REACHABLE, not NOT-AFFORDABLE.**

**RESULT: A MAKER REBATE PROGRAM EXISTS** - which is why the in-trade search was
always going to be empty, exactly as U11 predicted. Crypto rebate share **20%**,
the lowest of any category; `fee_equivalent = C x 0.07 x p x (1-p)`; paid
**daily in pUSD**, min **$1** accrued; self-normalising to ~20% of the fee your
own fill generated.

**IT CANNOT FLIP THE SIGN, AND THIS IS A CEILING NOT AN ESTIMATE:**
`p(1-p) <= 0.25` by arithmetic, so the rebate **cannot exceed 0.35 c/share**.
Measured pro-rata on our corpus: **0.168 c/share**. Threshold to flip even the
least-negative coin-day: **0.5164 c/share**. **The maximum possible rebate falls
32% short.**

**BONUS: reproduced the `744 of 754` claim whose receipt I could not find**
(Q-DA-48's provenance defect). Decoding OrderFilled word 4 and splitting legs by
counterparty: **901 taker legs 100% charged; 1,056 maker legs, 10 charged,
1,046 at exactly zero.** The **10** matches U5 exactly - the denominator does
not. **10/1,056 = 0.95%, not 1.33%.**

**LIMITS, because the answer is negative.** (i) The thresholds rest on
context-only levels with no interval, so this is **INDICATIVE, NOT
INFERENTIAL** - the rebate is below the SCALE of the losses, which is not a
verdict. (ii) I found 0 of 218 addresses clearing the $1/day minimum **and then
withdrew it** - 901 receipts are a SAMPLE, so it bounds observed volume, not
their actual volume. (iii) Rebates pay in pUSD, not USDC.

**THE GAP I AM NOT CLOSING.** The venue runs **TWO** programmes. I measured
Maker Rebates. The separate **Liquidity Rewards Program** pays for orders
**resting near the midpoint with no fill required** - not a share of taker fees,
so **the p(1-p) ceiling does not bound it**. It is now the only remaining term
that could move the sign, and unlike the rebate I cannot bound it by arithmetic.
**ASK filed: is measuring it authorised under the surface freeze?**

## DA - 2026-08-24 (R-116) - LIQUIDITY REWARDS: **REAL, LARGE, AND NOT CLOSEABLE BY ME**

Filed `Q-DA-51` **BLOCKING on DE's optimizer**. Unlike the maker rebate this one
is **not bounded by arithmetic** - it pays the QUOTE, not the TRADE.

**WHERE I LOOKED:** venue docs; `rewards_registry.jsonl` (confirmed a **size
heartbeat** - 552 records, keys `{recv_ns,n}` only); `tier1/quotes` (**top-of-book
only**, score not computable there); **`raw/` - FULL BOOK DEPTH, ~50 levels/side,
25 GB**, which is what made this measurable.

**POOLS: $1M across AUGUST**, $550k to 5-minute markets, BTC $300k. **I counted
the markets rather than assuming: 288 btc 5-min markets/day** -> **$33.60 per btc
window**, $5.60 per eth window.

**RESULT 1 - IT SPLITS BY ARM.** 100% of the pool would cover the loss on **9 of
10 JOIN coin-days and 0 of 10 FRONT coin-days**. JOIN fills ~1,400 sh/window vs
FRONT's ~7,500: a fixed pool is a large fraction of a small loss.

**RESULT 2 - AT THE TESTED CONFIG IT DOES NOT CLOSE THE GAP.** The replay rests
`quote_size_shares = 5.0`. Against real book depth (698-1,382 shares within 3c of
mid) our score share is **median 0.69%** -> **$0.23/window vs a $7.36-$17.94 JOIN
loss = 3.1% coverage**.

**RESULT 3 - THE ONE THAT MATTERS.** Reward is **strongly concave in RESTING
size** while loss is roughly linear in FILLS. Score share by resting size (v=3c):
5 sh -> 0.69%; 50 -> 6.5%; 500 -> 40.9%; 1,400 -> 66.0%. Robust across
v=2/3/5c; `b` cancels in the ratio.

**THESE ARE NOT P&L FIGURES.** The loss is fixed at the 5-share config's measured
loss; resting more WOULD fill more. **The fill-vs-resting-size response is the
missing term and I have not measured it.**

**WHY IT IS DE'S:** the reward/loss ratio is a function of the resting-size
policy - the optimizer's own free parameter. **An optimizer maximising markout
alone will systematically under-quote, because markout prices the cost of being
filled and never the revenue of resting.**

**LIMITS:** `max_spread` not published per-market (2/3/5c sensitivity used, not a
known value); scored **one-sided** while the real rule takes `min(Q_one,Q_two)`
and needs two-sided quoting - not modelled; pool assumed uniform over 31 days;
**book sample is 48 snapshots from 8 markets on ONE day**; and at 40-66% of the
reward zone **other makers would react - the book is not static under our own
size**, which nothing here models.

## DA - 2026-08-24 (R-125) - REWARDS STOPPED; **FORWARD POPULATION IS BROKEN THREE WAYS**

**Rewards out of scope.** `Q-DA-51` -> DEBT, trigger "the user reopens the
rewards question". **Stopping point is written down in `Q-DA-52`**, so a
successor restarts from measurement: fills scale **sub-linearly** (elasticity
**0.50-0.83**, never above 1); the reward/loss ratio improves with size by
construction but **converges to 31-48%, not 100%**; and **even 100% pool capture
leaves -$19.97 to -$48.09 per window**. That last one is an arithmetic ceiling
like the rebate's. Also recorded there: my own Q-DA-51 Result 1 was **not
commensurable** - it set a reward at one resting size against a loss at another.

**`Q-DA-53` FILED BLOCKING ON BE'S FORWARD EVALUATION.** All counts as-of
2026-08-24.

- **(i) The admissible holdout is 2.2 hours.** Freeze 2026-08-24T07:30:44Z; PM
  tape ends 09:40Z, mm_hf 09:48Z. **btc n=26, eth n=26 windows, 07:35-09:40Z,
  08-24 only** - out of 1,384/coin across 6 days.
- **(ii) It is a single day-cluster, on the partial day.** Day-clustered
  inference is not computable on n=1; `DAY_BLOCK_UNAVAILABLE` is the correct
  answer. 26 windows is **below the 30/coin/day** the policy comparison used.
- **(iii) THE SILENT ONE: there is NO Tier-1 or Tier-2 data for 08-24.**
  quotes/trades/coverage and all of tier2 **stop at day=2026-08-22, two days
  BEFORE the freeze**; twap stops at 08-23. A forward run against Tier-1 returns
  **zero admissible rows**; one against `raw/` **bypasses knowledge-time
  truncation, the distiller and the coverage receipts**. Quiet either way.
- **(iv) No coverage receipts for 08-24**, so the blind-period accounting
  (30/112 btc hours, 15/112 eth) **cannot be computed for the forward span**.

**NOT broken, and verified rather than assumed:** the earliest-first truncation
defect is **fixed in `ev_replay`** - provenance carries `days_sampled` distinct
from `days_read` with `sampled_is_known: true`. **That fix is in that harness;
whatever BE uses must be checked separately.**

**NEXT:** answer on (a) Tier-1 vs raw, (b) whether a 2.2-hour single-cluster
partial-day holdout is accepted and under what inference. If not, the remedy is
**collecting more forward tape - a wait, not a computation.**

## DA - 2026-08-24 (R-126) - REGISTER MARKING PASS: **51/51, 5 ASK / 46 FILING**

Was 0/51 marked, so the counter read all 51 as open ASKs. Now every row carries
`**ASK:**` or `**FILING:**` in BE's format. `conforms: true`, `register_rows: 51`,
no orphans, no malformed keys.

**THE FIVE ASKs - all BLOCKING, each naming its gate:**
`Q-DA-14`, `Q-DA-42`, `Q-DA-43` (STOP -> user) - `Q-DA-49` (DE's optimizer) -
`Q-DA-53` (BE's forward evaluation).

**`Q-DA-48` marked FILING, and the demotion is recorded IN the row** rather than
done quietly: its materiality question is the same one `Q-DA-42` blocks on, so
tracking it separately was double-counting my own row.

**FOUND WHILE MARKING: THREE OF MY ROWS WERE STRUCTURALLY MALFORMED** and did not
match `_REG_ROW` at all - `Q-DA-24` and `Q-DA-41` had no status cell, and
**`Q-DA-42` had lost its closing pipe in my own R-105 edit**. That row is a
BLOCKING ASK, so **my single most important open row was unparseable by the
counter**. Repaired; all 51 now parse. This was my contribution to the miscount
BE audited, and it was invisible because the row still LOOKED right.

**Forward-population guard (R-125) stands unchanged:** freeze 07:30:44Z, partial
days never counted as clusters, chronological truncation, blind-period coverage,
n and as-of on every count. `Q-DA-53` carries the open findings.

## DA - 2026-08-24T10:37Z (R-128) - **CONTAMINATION FOUND BEFORE FORWARD EVAL STARTED**

**Marking pass CONFIRMED COMPLETE:** DA 51/51 Form-A marked, **5 ASK / 46
FILING**, `register_rows: 51`, `conforms: true`, no orphans, no malformed keys.
DA is the only plane fully marked and fully parseable.

**`Q-DA-54` (FILING): the register uses TWO marker conventions** - `**ASK:**`
(63 rows: DA 51, BE 12, OPS 1) and `**ASK: <text>**` (29 rows: BE 19, DE 10);
57 rows carry neither. A counter keyed to one form miscounts the other by 29.
**I do not give a whole-register ASK total** - my own ad-hoc parser gave two
different answers on the same file, so I report only DA's slice.

**`Q-DA-55` (BLOCKING): a positional selector cannot express a mid-day freeze.**
`select_by_day` correctly fixed CROSS-day truncation (R-9), but **earliest-first
survives WITHIN each day**. BE's freeze is mid-day (07:30:44Z), which makes that
load-bearing. btc 08-24 has 127 windows; positions 1-91 are pre-freeze,
**92-127 (n=36) are the admissible tape**. At the shipped `per_coin=30` the
sample ends **02:25Z, 5.1h before the freeze, ZERO admissible windows** - and
**the day still counts as `holdout_complete`, not partial**. `per_coin=90` still
misses it by 5 minutes; you need **>= 92**. Raising the number only shrinks the
contamination ratio - **the forward population must be selected by a TIME
PREDICATE, not by rank.** Not edited: DE owns the optimizer, BE is user-held.

**THE HOLDOUT IS GROWING WHILE WE COUNT IT.** Collection is LIVE (12 btc files
in the hour to 10:36:30Z). **Admissible went 26 -> 36 during this session**,
~12 windows/hour/coin. `Q-DA-53`'s "2.2h / n=26" is **superseded to ~3.0h / n=36
as-of 2026-08-24T10:37Z**. Every forward count must carry its as-of or it is
wrong within the hour - and the "wait, not a computation" remedy is already
working on its own.

**STANDING WATCH continues:** post-freeze days into training sets, partial days
as clusters, selector truncation, blind-period coverage, n + as-of on every count.

## DA - 2026-08-24T12:57Z (R-129) - **VERIFICATION PRE-REGISTERED; DE HAS NOT LANDED**

**DE has not landed** as-of 12:56Z (`warning_window.py` 04:38Z,
`policy_optimizer.py` 07:29Z - both predate the finding; no time predicate).

**MY TEST IS FIXED BEFORE THEIR RESULT EXISTS** - freeze-pin discipline applied
to my own verification, so I cannot move it after seeing theirs.
Script: `scratchpad/verify_admissible.py`. Computes from `raw/` filenames only,
**never calls DE's selector**, so independence is structural not promised.

**PREDICATE:** admissible iff `window_start_epoch >= 1787556644`.

**BASELINE (as-of 2026-08-24T12:56:56Z):** btc **64 admissible of 1,422**;
eth **64**; all on 08-24; span 07:35Z onward; **usable day-clusters = 0**;
correct inference verdict **`DAY_BLOCK_UNAVAILABLE`**.
Growth ~12/hour/coin - it went **63 -> 64 in the 53 seconds between two runs**,
so DE and I must compare **at a stated common instant** or disagree for nothing.

**MIDNIGHT HAZARD, FILED BEFORE IMPLEMENTATION (`Q-DA-56`):**
**2026-08-24 STRADDLES the freeze, so it is PERMANENTLY admissibility-partial** -
its first 91 windows can never become admissible. After midnight it will hold
~199 admissible windows/coin, **clearing any `per_coin` threshold** and getting
labelled `holdout_complete` while being a **truncated day**. That is Q-DA-55's
defect in different clothes: **a cardinality test standing in for a boundary
test.** Correct rule: **a day is admissibility-complete iff EVERY window of that
calendar day is admissible.** 08-24 never qualifies and **must never count as a
cluster**.

**SCHEDULING CONSEQUENCE:** the first complete admissible day-cluster is
**08-25, and it does not exist until 2026-08-26T00:00Z** - not tonight's
midnight, which only starts 08-25 accruing. Day-clustered inference on forward
tape stays `DAY_BLOCK_UNAVAILABLE` **until at least 08-26**; a multi-cluster
interval is later still. Anything presented before then is window-clustered at
best and must say so.

**ON VERIFICATION I REPORT:** the counts recomputed at a stated instant, whether
`holdout_complete` is derived from the filter or from a count, and whether 08-24
is excluded as a cluster.

## DA - 2026-08-24T14:40Z (R-132/R-133) - **FREEZE VOID FILED; UNIFORM GATE LANDED**

**`Q-DA-57` FILED BLOCKING** - the freeze is VOID because it was never
anchored: pin `c83d5132...` at 07:30:44Z, but both builders were **added** in
commit `3454f60` (09:57) and **did not exist in git before it**. Pre-edit HEAD
and post-edit working tree both hash to `dd9fe9b1...` - **the user's edits are
bit-neutral**. The difference is not the `exact_receipt_events` flag, not a
cooldown value, not a name-block truncation, so it is a **real feature-set
difference**. Remedy is BE's: re-freeze against committed code, new instant,
**`frozen_at` references a COMMIT HASH** from here. **Cost is hours, not
results** - forward was `DAY_BLOCK_UNAVAILABLE` regardless.

**UNIFORM GATE LANDED (R-133, cited in-file), maintenance under R-110.**
`da_hf_pm_alignment.py`: a window straddling a collector-restart boundary now
**fails joint coverage exactly as a data gap does**, admissible only by an
explicit `stamp_waiver` naming the window. New: `stamp_boundaries_ns`,
`window_stamp_uniform`, `hf_collector_run_defects`. Result now carries
`stamp_covered`, `stamp_covered_pct`, `stamp_straddling_windows`,
`stamp_waived_windows`, `hf_collector_ledger_defects`.

**FAIL-CLOSED CHOICE WORTH KNOWING:** `hf_collector_runs` drops malformed lines
so bad text cannot relabel raw data - right for reading, wrong for CERTIFYING,
since a dropped line may have carried a boundary. **While any ledger line is
unreadable, NO window is certified uniform.** An ABSENT ledger is different and
is treated as uniform-legacy, per the reader's documented semantics.

**GATE BITES ON REAL DATA:** boundaries 13:48:08Z and 13:48:54Z; of 175 btc
windows on 08-24 **exactly one (13:45Z) is refused**, 174 admitted - it fires,
and it does not over-refuse. **40 selftests pass**, including an R-42 mirror
test (edge vs interior must answer differently) and the absent/malformed/clean
ledger cases.

**VERIFIER PARAMETERISED:** `scratchpad/verify_admissible.py` no longer
hard-codes the instant - it reads `frozen_at_utc` from the candidate (and
reports `frozen_at_commit` as **ABSENT - Q-DA-57**), or takes `--freeze`. A
hard-coded instant would have verified the void freeze forever.

**Baseline as-of 2026-08-24T14:39:45Z:** btc **84 of 1,442** admissible, eth
same, 0 usable day-clusters, `DAY_BLOCK_UNAVAILABLE`.

**PENDING:** DE's timestamp-predicate selector - re-issued, confirmed never
landed (`select_by_day` byte-identical, `git diff` empty).

## DA - 2026-08-24T15:03Z (R-134) - **ASKs RE-TRIAGED 7 -> 5; STANDING READY**

**NEITHER HAS LANDED as-of 15:03Z:** only `be_adverse_move_candidate_v1.json`
exists (no v2 re-freeze receipt), and `git diff` on `warning_window.py` is still
empty with `select_by_day` byte-identical (no timestamp predicate).

**RE-TRIAGE — 2 closed, 2 narrowed, 3 unchanged. Register: 55 rows, 5 ASK /
50 FILING, `conforms: true`.**

- **`Q-DA-57` CLOSED** - ruled and adopted in full (R-132). The remedy is BE's
  and in flight; I hold the verification, not the remedy.
- **`Q-DA-42` FOLDED into `Q-DA-14`** - its substance is *why* Q-DA-14 is
  urgent (net made fee treatment live, not latent), not a second question.
  Same dedup I applied to `Q-DA-24`; tracking it twice inflated my own count.
- **`Q-DA-49` NARROWED** - limbs (i)/(ii) superseded by `Q-DA-55`/`Q-DA-56`.
  **Live limb (iii): `m5_swm_cents` has NO INTERVAL and its protocol calls
  levels "context only"** - an optimizer maximising a level optimises a
  quantity with nothing separating signal from noise. Untouched.
- **`Q-DA-53` NARROWED AND RE-VERIFIED THIS TICK** - limbs (i)/(ii) superseded,
  but **(iii)/(iv) re-checked and STILL TRUE: tier1 quotes/trades/coverage and
  tier2 calib_panel/markout_events ALL still stop at `day=2026-08-22`**, two
  days before the freeze. Forward eval on Tier-1 returns **zero admissible
  rows**; on `raw/` it bypasses knowledge-time truncation, the distiller and
  the coverage receipts. **Least-attended open row, and it does not self-heal -
  the distiller has to run.**
- **Unchanged and live:** `Q-DA-14` (STOP inputs unpinned, now carrying
  Q-DA-42's urgency), `Q-DA-43` (A-BOOK-1 never frozen - and R-132's lesson now
  applies to its eventual freeze too: reference a COMMIT), `Q-DA-55` (positional
  selector; upheld, re-issued, awaiting DE).

**VERIFIER NOW FOLLOWS THE ARTIFACT:** `verify_admissible.py` resolves the
**highest-numbered** candidate rather than a remembered path, so when
`candidate_v2` lands it verifies against the NEW instant automatically. A
verifier pinned to v1 would have gone on certifying the void freeze - the same
shape as the defect it exists to catch. Absence of any candidate is loud.

**Baseline as-of 2026-08-24T15:03:04Z:** btc **89 of 1,447** admissible,
0 usable day-clusters, `DAY_BLOCK_UNAVAILABLE`. (26 -> 36 -> 77 -> 84 -> 89.)

**STANDING READY:** on the v2 receipt, verify DE's selector against the new
instant read from the receipt. The split freeze waits on that and nothing else.

## DA - 2026-08-24T15:56Z (R-136) - **FALLBACK FIXED; MY OWN BOUND WAS 2x TOO SMALL**

**Fixed before the recompute, as instructed.** `scratchpad/m5_concentration.py`
scores a fill **only if the horizon is observable**: inside the window AND a
quote exists **at or after** it. Unobservable fills are **EXCLUDED and
COUNTED**, never approximated.

**Actual exclusion 4.98% btc / 4.50% eth** against the **2.46%/2.01%** I filed
as the bound. My bound used a narrower test (`ts[i] < tm`) and so measured a
smaller failure than the one I had just described. Dominant cause is
**quote availability (81,474 btc), not window end (2,394)** - the book goes
quiet, the same venue behaviour A-CALIB-1's staleness ladder found.

**CORRECTED (tape population):** btc worst-10% **80.7%**, bar -> **1.60% fills /
12.60% volume**; per-share **7.10% / 6.71%**. eth worst-10% **81.0%**, bar ->
**4.38% / 17.93%**; per-share **11.74% / 12.10%**. n = 1,599,690 / 341,828.

**Every figure moved slightly AGAINST the favourable reading** - truncated fills
had short horizons and understated drift, so dropping them raises what a gate
must sacrifice. **Conclusion unchanged: btc needs 1.60% of fills where a diffuse
tape needs 45% - 28x concentration.**

**PENDING BE:** which population the conditional-markout curve is on. On the
answer I recompute concentration on the matching population - the scorer is
built and parameterised, so it is a rerun.

**ALSO PENDING:** DE's timestamp-predicate selector (`Q-DA-55`), verification
standing ready against the latest candidate.

## DA - 2026-08-24T16:30Z (R-137) - **REPLAY-ARM CONCENTRATION: MY TAPE NUMBER FLATTERED IT**

**`Q-DA-61` FILED.** The commensurable number is in, and it is **~4.7x (btc) /
3.1x (eth) weaker** than the real-fill tape figure I filed first.

| population | btc worst-10% | btc bar -> fills/vol | eth worst-10% | eth bar -> fills/vol |
|---|---|---|---|---|
| real-fill tape | 80.7% | 1.60% / 12.60% | 81.0% | 4.38% / 17.93% |
| **replay arm (BE's)** | **53.2%** | **7.47% / 8.84%** | **52.8%** | **13.46% / 15.93%** |

n = 31,645 btc / 5,705 eth fills over 90 windows each, as-of 16:25:57Z.
Convention NOT reimplemented - `edge_layer1.decompose` called directly, so the
drift term is the one BE's curve conditions on.

**VALIDATION PASSES:** mean markout **-0.8654 c/share** vs `policy_comparison_v2`
three-day JOIN mean **-0.855** - 1.2% apart, so the replay I ran IS the policy
arm. (BE's -0.5325 differs because `edge_layer1.run` uses `iw.select`
cross-day-earliest-first; I used `select_by_day` on three fixed days. Different
SAMPLE, not different method.)

**Route not dead** - 7.47% against a diffuse 45% is still **6x concentration** -
**but the predictor's job is materially harder than my tape number implied.**
R-137 barring the cross-citation is what stopped that error reaching a decision.

**STRUCTURAL POINT FOR GATE DESIGN:** on the replay, cash-ranked and per-fill
ranked concentration nearly coincide, because the arm quotes a **FIXED 5 shares**
- no size variation. On the real tape a meaningful part of concentration was
**size** (worst 2.04% of fills were ~6.7x average). **The replay has no size
lever, so its concentration is PURE TOXICITY.** A gate designed against replay
numbers is asked to do by prediction alone what the real book could partly do by
sizing.

**ALSO THIS TICK:** `Q-DA-60` filed BLOCKING on the v2 re-freeze verification -
builder `sha256 e8a82b66` MATCHES (content-anchored, verified), numbers
recomputed, but `feature_schema_hash` is **vestigial** (nothing produces it) and
`frozen_at_commit` predates the freeze with `committed_at_freeze: false`.
Corrected my own Q-DA-57 mechanism: v1's builder **was never in the repo at all**.
