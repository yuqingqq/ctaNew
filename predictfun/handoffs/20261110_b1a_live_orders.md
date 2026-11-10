# B1a: live's own orders through the research fill model (2026-11-10)

As of 2026-11-10 13:19Z (session ce68ebe0). Item B1a of `handoffs/20261109_TODO.md` (USER "Yes" 2026-11-10;
specification in HANDOFF 2026-11-10 04:56Z). Script `research/pf_b1a_live_orders.py`: v1 committed before its first run
(7ef3de6e, output 13:15Z), v2 committed before its run (1a253a45, output 13:19Z). v2 supersedes v1 (one change, below).
Output in the cohort directory `live_vs_replay/b1a_20261110/` (b1a.{json,txt}; v1 as b1a_v1.*), copies in
`reviews/b1a_20261110/`.

A historical reference, not a calibration: nothing is fitted or selected. One consumed partial day (the 220 windows of
2026-11-08, cut 23:00Z, 0.764 window-days): point estimates only.

## Question

Does the research fill model reproduce, order by order, what live was filled? Every replay result in the strategy review
changes sign with the assumed trade time. Live's own orders are the only orders whose outcome is known.

## Design

- Population: 30,221 accepted live orders (tokyo-6 16,312; basis-tokyo-1 13,909); 849.867 sh filled, equal to the REST
  total. tokyo-5 (11-07) has no research tape in this cohort and was not read.
- Each order is replayed on its own path through the review's fill rule (M: queue ahead = shown size at our price,
  capped by later frames; L: no cap; U: no queue ahead), with trade times shifted 0 / 50 / 250 ms.
  - P1 (primary): from receipt of the ack to receipt of the cancel ack (or the final fill notice, or the window end).
  - P0 (bound): from the decision to the cancel request.
- Identity: live's 99 REST maker legs from the fill-clock audit (hashes checked against its receipt). 92 legs (796.9 sh)
  are eligible; 6 have an ambiguous clock (28 sh) and 1 is untimed in the replay (25 sh), both statuses. A replay fill is
  REAL if it comes from the taker order that filled live's order.
- Readings R24-R29 were declared in the script header before the first run.

v1 to v2: v1 matched on the maker leg row. Within one taker order the replay serves the legs at one price in arbitrary
row order. A post hoc check of v1 found that 33 of the 59 uncredited own legs had a replay fill from the same taker
order. v2 matches on the taker order. Readings, thresholds, controls and configs are unchanged. REAL shares at M 0 / M 50
were 47.0 / 5.1 in v1 and are 316.3 / 281.4 in v2. Volume, value, lag and cancel-pending numbers are the same in both.

Controls, all passed before any table:

- 2,000 random synthetic window-sides through the committed replay (`pf_review_queue_reach.sim`), 16,226 fills: its
  orders replayed here give 0 differences. Four planted errors are each caught: frame cap off (54 differences),
  activation +10 ms (1,486), through-trades ignored (841), service order reversed (synthetic case S1).
- Tape: the 0-60 s trade statuses equal the review's committed ones (753 untimed legs / 9,131.4 sh; 94,783 timed legs /
  1,074,957.4 sh).
- Identity: every own leg is found on the nine content keys; side and price equal the order's (99 of 99); the replay's
  trade time equals the audit's engine clock exactly on all 92 eligible legs.
- A planted credit case (recall 0.5, precision 0.5) and the refusals (unsorted inputs, an order at the cutoff) pass.

Statuses: 2,722 untimed legs (36,397.6 sh) in 0-300 s excluded; 7 orders with no time to rest under P0; no window
without frames. Measures (2) and (3) of the specification are the audit's and A4's and were not recomputed (causal width
p50 279 ms; fill notice p50 435 ms after the public pending increase).

## Results

**Live's own fills.** 92 legs, 796.9 sh: -0.25 c/sh at 1 s and -0.01 at 5 s (research mid, at the trade's research
time); -$2.59 a day at 1 s.

| Order age at the fill (research trade time - ack) | Shares | c/sh 1 s |
|---|---|---|
| < 100 ms | 261.4 | +0.05 |
| 100-500 ms | 237.0 | +0.30 |
| >= 500 ms | 298.4 | -0.94 |

**The replay on the same orders.** Recall = share of live's eligible fill shares that the replay credits from the right
taker order. Precision = share of the replay's shares on live's orders that come from it.

| Config | Replay sh | Replay / live | Recall | Precision | Orders filled: replay / live / both | c/sh 1 s: all / REAL / other | c/sh 5 s | Cancel-pending sh: replay / live |
|---|---|---|---|---|---|---|---|---|
| M 0 P1 | 1,249.0 | 1.47 | 0.397 | 0.253 | 142 / 86 / 59 | -0.37 / +0.11 / -0.53 | +0.73 | 438.2 / 285.9 |
| M 50 P1 | 1,351.1 | 1.59 | 0.353 | 0.208 | 153 / 86 / 58 | -0.66 / +0.24 / -0.89 | -0.14 | 546.5 / 218.0 |
| M 250 P1 | 1,668.9 | 1.96 | 0.165 | 0.079 | 202 / 86 / 39 | -2.36 / +0.32 / -2.59 | -2.31 | 519.5 / 168.0 |
| L 50 P1 | 1,297.2 | 1.53 | 0.341 | 0.209 | 147 / 86 / 57 | -0.70 / +0.25 / -0.95 | -0.12 | 532.6 / 218.0 |
| U 50 P1 | 11,435.7 | 13.46 | 0.488 | 0.034 | 1,680 / 86 / 70 | +0.90 / -0.85 / +0.96 | +1.35 | 2,937.1 / 218.0 |
| M 50 P0 | 1,276.9 | 1.50 | 0.394 | 0.246 | 146 / 86 / 46 | +0.66 / +0.14 / +0.83 | +1.07 | 0.0 / 218.0 |

By order age at M 50 P1: orders under 100 ms old take 875.4 replay sh (65% of the replay's shares), 12.3% of them REAL;
orders 500 ms or older take 195.9 sh, 47.7% REAL. At M 0: 8.3% and 46.8%.

**Where live's own fills went in the replay (M 0 P1).**

| Outcome | Legs | Shares |
|---|---|---|
| Credited (same taker order) | 42 | 321.8 |
| Research trade time after the order left (cancel ack or final fill notice) | 13 | 155.9 |
| Research trade time before the ack | 11 | 100.0 |
| Not credited: volume at our price up to the fill < queue ahead at activation | 19 | 162.5 |
| Not credited: volume at our price up to the fill >= queue ahead | 6 | 51.7 |
| Order already filled by other trades | 1 | 5.0 |

**Lag bound.** For each own leg, U = research trade time - our decision. The trade cannot have matched before our
decision, so its match-to-report lag is below U. Share-weighted: p25 84 ms, p50 191 ms, p75 765 ms. 11.3% of own-leg
shares have U < 0 (the tape's time is before our decision, so it is wrong there), 5.0% have 0-50 ms, 41.2% 0-250 ms.

**Declared readings, as computed.**

| Reading | Result | Numbers |
|---|---|---|
| R24 volume | OVER-CREDITS in all 6 configs | replay / live 1.47-1.96 (M, L), 13.46 (U) |
| R25 identity | FAILS in all 6 | precision 0.03-0.25, under 0.3 everywhere |
| R26 young orders | YOUNG-PHANTOM | REAL share under 100 ms 8.3% / 12.3% against 46.8% / 47.7% at 500 ms or more (M 0 / M 50). Null, 500 draws within run x side: p99 +0.273 against observed +0.353; 0.2% of draws at or above |
| R27 lag bound | CONSTANT-250-CONTRADICTED | 41.2% of own-leg shares have 0 <= U < 250 ms (threshold 25%); 0-50 ms 5.0% |
| R28 value | MATCHES at M 0, M 50, L 50; REPLAY-UNDERSTATES at M 250; REPLAY-OVERSTATES at U 50 | replay minus live at 1 s: -0.12 / -0.41 / -0.45; -2.11; +1.15 c/sh |
| R29 cancel-pending | PEND-OVERCREDITED | 546.5 against 218.0 sh (M 50 P1) |

## What it says

1. The fill model does not reproduce live's fills order by order. At M 50 it credits live's orders with 1.59 times the
   shares live got. 21% of those shares come from the taker orders that filled live, and it credits 35% of live's fill
   shares. It fills 153 orders against live's 86, 58 of them in common.
2. The excess is in young orders and cancel-pending time. 65% of the replay's shares on live's orders are on orders
   under 100 ms old, and 88% of those are not the trades that filled live (M 50). The replay credits 2.5 times live's
   shares on trades at or after the cancel request.
3. No constant trade-time shift fixes it. A 250 ms shift is contradicted by live's own fills: 41% of their shares have a
   research trade time within 250 ms of our decision. It also makes the replay worse (precision 0.08). The tape's trade
   time is wrong in both directions: 11% of our fill shares sit before our decision, 20% after the order left.
4. The shown-size queue is contradicted for a fifth of live's fills: 19 legs (162.5 sh) were filled live although less
   volume traded at our price than the queue the replay put ahead of the order.
5. Per share, the replay values live's orders at 0-50 ms (-0.37 to -0.70 c/sh at 1 s) within 0.5 c/sh of live's actual
   -0.25; at 250 ms it is 2.1 c/sh too low. It gets there with the wrong fills: the REAL part is +0.11 to +0.25 c/sh,
   the rest -0.53 to -0.95.
6. With the trade clock fixed, the value of the same orders changes sign with the order's own path: -0.66 c/sh at 1 s
   from ack to cancel ack (P1), +0.66 from decision to cancel request (P0). The two paths differ by live's median
   latencies, about 34 ms at the start and 20 ms at the end.
7. Live's real fills on young orders were not adverse: +0.05 c/sh at 1 s under 100 ms, +0.30 at 100-500 ms. Fills on
   orders 500 ms or older were (-0.94).

## What this changes

- No replay number in the strategy review (sections 4, 6.3, 6.5) or in round i is validated at the order level. The
  review's point that the fill clock decides the sign stands, sharpened: neither 0-50 ms nor 250 ms is a valid constant.
  Per share the 0-50 ms columns are closer to live; the volumes are not.
- The birth-step candidate (review 6.5, arm B-g90) is made of young orders, the part this check finds 88-92% phantom.
  The replay cannot rank it. Its test is live fills with exact match clocks (B1).
- B1 needs exact match clocks, which the historical tape cannot supply (fill-clock audit). The next measurement is a
  live run with the A4b wallet capture (USER decision D2: a new Tokyo tag from pmmmrust main 79a05d3).
- Point 7 agrees with the review's finding that resting longer costs (6.3 R11, 6.5 R17), now on live's own fills.

## Limits

One consumed partial Sunday; 86 filled orders and 92 eligible legs; point estimates. The research book contains live's
own orders, so the queue ahead at activation can include our other orders and makers who joined after us. The fill rule
is the replay's; the venue's matching rule is not known. Trade times are the replay's (WS pending message, else the E2
estimate); 2,722 untimed legs are excluded. tokyo-5 is not covered.
