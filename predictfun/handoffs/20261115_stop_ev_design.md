# STOP-EV design: keep a resting order when its expected spread capture beats its expected sweep loss (2026-11-15, session aea5bfb9)

USER (2026-11-15): "Generally, if our resting orders earn more than adverse selections, we can keep there right? ... we need to be front in
the queue to catch the spreads" / "You mean we mainly measure the expected capture spread vs get swept loss" / "Does this model make more
sense or is this target easier to formulate, it does not have so many downstream effects that are hard to measure" / "Does this also more
similar to what top makers do? If so, we can design the model first, record in handoffs and status".

Status: ACCOUNTING CONTRACT REVISED after the user's request to resolve the accounting issues.
The implementation and exact integration contract are in
[20261115_stop_ev_accounting_fix.md](20261115_stop_ev_accounting_fix.md), with
versioned test evidence under `../reviews/stop_ev_accounting_20261115/`.
The other session's data builder is in progress; this does not claim a validated
historical dataset or fitted STOP-EV model. Sections 2–6 below retain the original
proposal for provenance; their aggregate formulas, sign sanity gates and
harm/good ratio are **superseded**, not implementation instructions. Sections
7–8 and the linked accounting contract define the corrected target.

## 1. Why (evidence)

- Harm-only is not selective (STOP-MODEL harm-capture check, 5 test days, 3.06M keep-all order states, 1 s value w/o young fills): pulling
  the riskiest 35% of states by v6's P(sweep) avoids 72% of the harm but gives up 75% of the good fill value (guard 61% / 54%; oracle pulls
  9% and avoids 93% / gives up 41%). At every pull share harm avoided ~ good given up: the risk score finds activity at our level, not
  sweeps against benign fills.
- Our surviving orders earn nothing: 1 s value of fills on orders not run over +0.007 c per order share vs -1.29 c when run over. Even
  perfect sweep foresight takes the always-resting ladder only from -$5.9k to +$0.8k a day (M 50).
- The top makers' behaviour matches an expected-value keep rule, not a sweep rule (FINDINGS, real fills / public tape): they pull 27.8% of
  intervals vs our guard 59.8% (LEARN-PULL); they are gone before 47% of trades that go THROUGH their price vs 27% that stop, and among
  STOPPING trades 1.77x as often before a fill that loses at 1 s as before one that wins - kept fills +0.56 c, skipped -0.12 c
  (LEARN-PULL-TARGET: their pulls track a ~1 s fill-value target, gains and losses); their at-best fills are 82% stopping trades vs our 37%,
  worth +0.378 vs our +0.004 c/sh at the best (TOP-COMPARE); a front fill on a stopping trade is worth +0.50 c/sh, +0.07 even on a
  through trade (LEARN-PULL); their pulls follow the level's own state (thin / depleting queue), hardly Binance volatility, which drives ours;
  they are at the next price before a move on the protection side (SIDE-MOVES). We observe fills and inferred pulls, not their model:
  "consistent with", not proof.

## 2. The decision (the USER's design, made explicit)

Every 50 ms, for each of our resting orders (side s, price p = level j of the known best, remaining size Z, estimated queue ahead q):

    EV(keep) = E[ min(Z, max(0, V_stop - q)) ] x v_stop  -  P(sweep) x Z x v_sweep  [ - queue-option term, variant ]
    keep iff EV(keep) > margin (hysteresis: pull at EV < -m1, re-enter only at EV(post) > +m2)

- V_stop = shares traded at price p within (t + 22 ms, t + H] by taker trades that STOP at p (their last fill is at p; the next price is
  not taken) before p is swept. A front order (q = 0) catches the first V_stop shares; a back order only what exceeds the queue ahead.
- P(sweep) = P(the best moves >= j ticks against the side within (t + 22, t + H]) - the existing STOP-MODEL v6 heads. A sweep fills our
  remaining size wherever we stand in the queue.
- v_stop, v_sweep = 1 s markouts per share (cents) of a stopping fill at p and of a sweep fill (deeper sweeps lose more); measured from
  public trades, by level and state.
- The regime behaviour the USER described follows from the formula: calm -> P(sweep) ~ 0, the levels near the best have V_stop > q
  -> kept (deep levels ~0 either way; kept for next-price positioning); one-sided move -> the adverse side's inner levels get P(sweep) high
  -> pulled, the protection side's next-price orders get V_stop high -> kept; two-sided swing -> deep levels kept iff the state-dependent
  v_sweep (reversion) and V_stop make EV > 0 (to be measured: the top makers' big-revert cases pay the side that was run over, MOVE-CASES).
- Posting / re-entry stays with the other session's post-vs-wait model (HANDOFF direction 2026-11-14). The same components give EV(post) at
  the back of the queue (q = the visible size) and are offered to it; until it lands, STOP-EV re-enters a level when EV(post) > +m2.

## 3. Components (each its own label, checked on its own)

| Head | Label (all from public data, known-time inputs = v6's 71) | Simulator? |
|---|---|---|
| SWEEP (have) | y = best moves >= j ticks against within (t + 22, t + H], H = 250 ms / 1 s, j = 1..6 | no |
| STOP-FLOW (new) | for hypothetical queue positions q in {0, visible size x 0.25 / 0.5 / 1}: F = min(Z, max(0, V_stop - q)) in (t + 22, t + H] - computable for ANY q from the public trades at p, so no own-order simulation; q enters as an input | no |
| VALUES (new) | v_stop(j, state), v_sweep(j, state): the 1 s (and 5 s) markout per share of stopping / sweep fills at level j | no |
| QUEUE (estimate) | q of OUR order: the visible size at p when we posted (we join behind), then minus trades at p and the minimum of the visible size since (the lane's rule M estimate, client-known version) | yes - the only simulator-dependent piece |

Trade timing for STOP-FLOW labels: WS-timed legs are exact; E2-estimated anchors are +-250 ms (C12) - build the label from WS-timed legs
with E2 legs as a status (counted; a sensitivity variant includes them). Trade legs reach us 2.5-13 s late, so V_stop is a LABEL only,
never an input (inputs stay the book / Binance / fair-price features known at t).

## 4. Readings to declare before any run (draft)

- Components: SWEEP as v6 (calibration, AUC / Brier skill by level, 5 test days). STOP-FLOW: calibration of predicted vs realized fill
  shares at each q, and skill against a base rate by level. VALUES: per-level markouts with day-clustered spread.
- Selectivity (local, before any replay): on the keep-all order states, the EV rule at the harm-only rule's pull share must avoid at least
  as much harm while giving up less good value (harm / good ratio above harm-only's ~1.0; the oracle's is 2.25).
- Portfolio (STOP-PLAY lane, whole portfolio, C15): STOP-EV vs harm-only STOP vs keep-all vs the oracle vs LDG at M 0 / 50 / 250, with and
  without fills on orders < 100 ms old, WS / E2 split, queue zero / inf, 30 and 100 sh; per day and pooled; pulls and quote time per rung.
- Against the top makers: our at-best fill STOP share and value vs theirs on the same trades (TOP-COMPARE metrics).

## 5. Risks and limits

- Queue position is a simulator estimate (rule M, unvalidated on live orders, B1a) - it now sits in ONE explicit term; validate with the
  live-queue probe (plan step 4) before trusting the gain side.
- Cancels ahead of us are invisible (size changes do not separate trades from cancels): the queue estimate is an upper bound of the
  queue ahead (conservative for the gain side).
- 1 s values ignore inventory beyond 1 s at decision time; the whole-portfolio replay checks the total.
- Unknown queue beyond the 5th visible level (L1): report zero / inf bounds.
- Consumed days only (development); a frozen candidate needs >= 5 untouched days.

## 6. Build order (each step on the USER's go; code + synthetic checks with planted errors + commit before any run)

1. STOP-FLOW labels from public trades (with the q grid), falsifier and brute-force checks; VALUES table.
2. STOP-FLOW head (v6 inputs + q), chronological folds as v6; component readings.
3. EV policy in the STOP-PLAY lane (queue estimate from the lane), selectivity check, then the portfolio replay against the harm-only baseline.

## 7. Decisions agreed with the USER (2026-11-15, before the build)

- LOCAL EV (USER: "it looks still have downstream complications"): EV(keep) values only this order's own fills within the next horizon
  H = 1 s under the fixed neutral continuation (subject to declared engine safety/end rules), each tranche at its own fill time + 1 s.
  Policy decisions may repeat every 50 ms; the label does not assume a learned next-step cancel. Downstream effects are handled outside the EV: pre-positioning value -> KEEP BY DEFAULT (an EV ~ 0 order stays); the value of
  the queue place -> ONE margin (pull only if EV < -m, chosen on the calibration day); inventory from multi-level sweeps -> the joint lane's
  exposure cap (when it binds, keep the highest EV per share); re-entry -> the posting rule (the other session's model; meanwhile EV(post at
  the back) > +m + h); P&L beyond 1 s -> the whole-portfolio replay.
- QUEUE (USER: "how do we keep our position front in the queue"): no churn (every pull resets us to the back), pre-position deeper levels
  before they become the best, be early at new levels (posting rule); q = the visible size at our price when we posted, then the minimum
  visible size there over KNOWN frames since (client-known; the one simulator-dependent piece).
- MULTIPLE LEVELS (USER: "how do we manage multiple levels"): each level its own order, own EV and own q, decided independently (gaps
  allowed); sweeps go inner -> outer, so the predicted stop emerges; values v_stop / v_sweep per side and level (deep fills in reverting
  moves can win); v1 = 30 sh per level; v2 = size per level by EV per share.
- SIDES (USER: "Does bid/ask have separate models"): one pooled model with side-aligned inputs and fside (the sweep model is symmetric:
  AUC bid / ask 0.928 / 0.923 at 72 ms, 0.770 / 0.767 at 1 s); values per side; per-side calibration and decision checks; split a side if it
  differs materially.
- v1 LABEL: one outcome-independent probe per neutral order generation; actual chronological fill tranches from the corrected joint
  ladder, limited by shared public-leg quantity and the order's remaining size. Fills before cancellation arrival reduce remaining size
  but contribute no avoidable value. Fills at or after arrival and within the one-second exposure horizon contribute quantity times
  side-signed markout at their own fill time + 1 s. No aggregate average, automatic full-size sweep, or replacement-order value.
  Per-share normalization uses client-known remaining size. Invalid/missing timing, book or mark data produce explicit statuses.
  Queue-grid hypothetical tables can remain diagnostics, but are not independent economic actions or the primary order-level labels.

## 8. Build stages (USER 2026-11-15: "we should have a few preprocessing steps, first, data processing, check its coverage, data quality, etc, then model, then the pipeline")

STAGE 1 - DATA (no model): build neutral actions with `pf_stop_ev_accounting.replay_neutral`, `Ledger`, and `build_actions`.
Reconcile all raw legs plus exclusions, then engine per-leg/per-order/flat fills, actual remainders and position. Use the corrected
activation-time queue engine. Report action/status coverage by day, side, rung and timing assumption; include SHARED and ENDED actions
through the exact focal-order ledger. Verify known-time feature joins and per-tranche marks. Run strict timing and explicitly qualified
E2 sensitivities separately. Economic sign and similarity to top makers are descriptive comparisons, never data pass gates.
STAGE 2 - MODEL: local order value from the existing market inputs plus causal order-level inputs (age, depth, submitted and known
remaining size, known queue and visibility). Keep the agreed pooled side-aligned model with side indicator; check each side separately.
Assess calibration and incremental skill over edge/queue/depth/size baselines. Economic evaluation uses unique neutral actions, local
kept value and a count/exposure-matched null with >=200 draws; carry all exclusion counts. Repeated steps and queue variants do not
multiply the economic population. Freeze the full pipeline and multiplicity before historical execution; consumed dates remain development.
STAGE 3 - PIPELINE: evaluate the learned keep rule in the corrected joint lane, with declared calibrated margins and separate posting
logic, at the frozen clocks/queue assumptions/sizes/caps. Report whole-portfolio value and churn; local action-value sums are not strategy
P&L. Retain historical baseline receipts and produce new versioned artifacts. Any validation claim requires >=5 later untouched days.
