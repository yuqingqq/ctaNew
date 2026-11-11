# B13b / B13c: do the pressure features explain the top makers, and why we never hold the front of the queue (2026-11-11)

As of 2026-11-11 07:15Z (session ce68ebe0). USER "Then check if these top makers pull their quotes because of these
moves, do these features somehow explain their behaviors. After this, check why top makers still have front position
in the queue, it looks our model guards on some orders that should be there?". Scripts `research/pf_b13b_top_behaviour.py`
and `research/pf_b13c_queue_pulls.py`, committed and pushed before their runs (54dc5df9; outputs 07:10Z and 07:12Z).
Output in `live_vs_replay/b13_20261111/`, copies in `reviews/b13bc_20261111/`. One consumed partial day (the 220 windows
of 2026-11-08, first minute): point estimates; 11-08 is the day the pressure idea came from, so B13b is descriptive.

## B13b: do the pressure features explain the top makers' behaviour?

Pressure composite: B13a's fold models (trained on other days, none on 11-08) predicting the next-second Predict move,
read 50 ms before each event and signed against the maker's side (adverse) or for the taker (favourable). Controls
passed; the event counts and re-hit opportunities equal B12e's and B12d's.

| Behaviour | Result |
|---|---|
| Taking: AUC of SMART takes against random times | EMPH 0.771, BASE (current features) 0.725; raw: Predict L1 imbalance 0.669, Binance trades 0.606, Binance flow 0.590, Binance imbalance 0.568, Predict 250 ms move 0.538 |
| Staying after an absorbed fill, by adverse-pressure tercile (low / mid / high) | top 5 57.9% / 56.5% / 64.9%; ranks 6+ 36.7% / 33.7% / 44.4%; AUC for leaving 0.461 (top 5), 0.457 (6+) |
| Presence: the top 5's share of at-best fills by tercile | 29.4% / 28.1% / 27.8% (high / low 0.94) |
| Net at 1 s by tercile | top 5 +0.68 / +0.15 / -0.50; ranks 6-20 +0.55 / +0.04 / -0.63; ranks 21+ +0.64 / +0.12 / -1.24 |

Readings: R100 NO-PRESSURE-STAYING (+7.0 pp; they are re-hit more, not less, under adverse pressure); R101 NO-PULL-SIGNAL
(0.94); R102 FEATURES-EXPLAIN-TAKES (0.771); R103 NOT-PRESSURE-LEAVING (0.461); R104 EMPH-EXPLAINS-BETTER (+0.046).

1. Their taking is what the features explain: the pressure model picks out the moments they take (AUC 0.77, against
   0.73 for the current features). Predict's own L1 imbalance is the strongest single raw feature.
2. Their quoting does not show pulling on these moves: their share of fills is about the same when the model predicts a
   move against them, and after a hit they are re-hit as often or more (for everyone, the next taker order comes sooner
   under pressure, which leaves less time to leave). Under adverse pressure everyone loses; the top makers lose less
   (-0.50 against -1.24 for ranks 21+). What makes their fills in those moments less bad is still not visible.
3. For us the useful part is the first: the informed takers' moments can be predicted from the books. A maker who
   steps back when the pressure model turns against its side stands where those takers will not hit it.

## B13c: why we never hold the front of the queue

Live order paths of tokyo-6 + basis-tokyo-1 (849.867 sh filled, as B1a); pulls = our orders cancelled while their price
was the best of their side, first minute. Controls passed (FIFO carry, would-be fill rule, queue raise, re-place match).

| Measure | Result |
|---|---|
| Pulls at the best (first minute, 220 windows) | 12,240 |
| Order life, ack to cancel request | p10 1 ms, p50 23 ms, p90 231 ms; 99% under 1 s |
| Re-placed at the same price within 1 s | 82.5% |
| Queue ahead when our order goes live | p50 381 sh |
| Queue lost per re-place | p25 -2, p50 0, p75 +20 sh |
| Pulls while the research guard still said quote (ON) | 10,606 (87%); flow at our price in the next 2 s +0.45 c/sh on 428,980 sh; would-be fills had we stayed 1,631 sh at +0.04 |
| Pulls with the research guard OFF | 1,627; flow +0.31 c/sh on 67,221 sh; would-be fills 267 sh at -0.98 |

Raw-log check (all minutes, every cancelled order): life p50 26 ms (tokyo-6) and 38 ms (basis); 52% of orders live under
50 ms; more than 10% are cancelled before their ack arrives; a new decision on a lane every ~250 ms (p50); 58-69% of new
orders are at the same price as the lane's previous one.

Readings: R110 CHURN-HEAVY (82.5%); R111 QUEUE-KEPT (0 sh per re-place); R112 PULLS-COST (+0.45 c/sh after ON pulls); R113
STAYING-PAYS (+0.04 c/sh on 1,631 sh); R114 GUARD-PULLS-GOOD-ORDERS.

1. Our engine almost never lets an order rest: it cancels at-best orders a median 23-38 ms after they go live and puts a
   new one at the same price, about four times a second. 87% of these pulls happen while the research guard still says
   quote, so they are not guard decisions on the signal; they are churn.
2. Each re-place costs little queue by itself (the old order never lived long enough to move up), but the result is that
   we are always at the back: about 380 sh ahead of us every time an order goes live. The top makers keep orders
   resting (B12d: re-hit 59.5% after a hit), so they gain time priority; we never do.
3. The flow we step away from is good: makers at our price earned +0.45 c/sh in the 2 s after our ON pulls. Had we stayed
   at the back, we would have been filled on 1,631 sh at only +0.04 c/sh: from the back, only the big trades reach us. The
   value is in resting long enough to move up, which the current loop never does.

## What this changes

- Strategy: stop the churn. An order that is still at the best and still has the guard ON should stay; cancel only on
  a real change (guard OFF, price no longer the best, size change). Then measure time at the best and queue position
  with the A4b capture (B1).
- Guard: use the pressure model (B13a) to step back from the informed takers' moments, which it predicts (AUC 0.77).
- Live session (Tokyo / Dublin): the engine's cancel-replace loop (life p50 26-38 ms; >10% cancelled before their ack)
  is an execution issue to check alongside the slow cancels of B12b; churn also multiplies exposure to slow cancels.

## Limits

One consumed Sunday. The would-be fills use the shown queue with FIFO and no cancels ahead (B1a: this rule over-credits
live by 1.5-2x); pro rata flow is not our fill. Re-hits under pressure mix staying with the next order's arrival time.
The live engine's guard is not the research guard; "ON" here is the research guard.
