# STOP-EV design: keep a resting order when its expected spread capture beats its expected sweep loss (2026-11-15, session aea5bfb9)

USER (2026-11-15): "Generally, if our resting orders earn more than adverse selections, we can keep there right? ... we need to be front in
the queue to catch the spreads" / "You mean we mainly measure the expected capture spread vs get swept loss" / "Does this model make more
sense or is this target easier to formulate, it does not have so many downstream effects that are hard to measure" / "Does this also more
similar to what top makers do? If so, we can design the model first, record in handoffs and status".

Status: DESIGN ONLY - nothing built or run. The harm-only STOP-PLAY replay running now (v6, 30 sh, queue zero / inf, then 100 sh) is the
baseline this design must beat.

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
