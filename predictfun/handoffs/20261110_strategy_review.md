# Predict.fun MM: strategy review

As of 2026-11-10 13:22Z (session ce68ebe0), revision 4. Scope: pm_ladder_v3 (frozen 2026-11-07, b261091d) and the
work around it.

Questions from the user: (1) does "capture spread from users' trading, control the remaining inventory risk" hold for
the current strategy; (2) what the live-vs-replay gap on short-lived orders and the pull buffer mean; (3) how to
optimize. Follow-up: do we already have "enter at the line, pull only when the guard is clearly negative", what
exactly is the proposed mechanism, and what else is worth digging into (sections 6 and 7).

No strategy code, frozen artifact or receipt was changed. Seven read-only diagnostics were added (section 10).

Revision 2. The 06:27Z version recommended a two-line gate for the touch order: enter at the line, leave only when the
guard is clearly negative. Two further diagnostics, a band sweep and an order-level queue replay (sections 6.2 and
6.3), do not support it for an order that joins the back of the queue. Sections 1 and 5-10 are corrected. Sections 2-4
are unchanged apart from two wording fixes (the rebate formula in 3.1, the 42.6% row in 3.3).

Revision 3 (10:01Z). Section 6.5 is new: the queue-priority candidates of section 7, item 4 were measured at the order
level (`pf_review_level_birth.py`, committed before its first run). Early entry does not earn priority at our latency,
and no position pays without the guard. Sections 1, 6.4, 7, 8, 9 and 10 are updated; sections 2 to 6.3 are unchanged.

Revision 4 (13:22Z). B1a is done (`handoffs/20261110_b1a_live_orders.md`). On live's own orders the replay's fill
model credits 1.47-1.96 times live's shares under the M and L rules, and 8-25% of them come from the trades that filled
live. A constant 250 ms trade-time shift is contradicted by live's fills. Section 7 items 2, 3 and 9 now point to it and
to backlog B11 / B12; nothing else changed.

## Evidence labels

| Label | Population | Status |
|---|---|---|
| P | Public market data, 8 read days (10-23..27, 10-31, 11-01, 11-02); EXPLORATION doc §32-34, §39, §57 | Valid; in-sample description |
| R | v3 replay, 9 consumed days, 2,592 markets per timing, J (central) / X (the -50 ms bound); `20261110_inventory_cost_results.md`, `markout_by_age.json` | Fill model not calibrated (section 4) |
| L | Live, Tokyo, 220 complete windows of 2026-11-08 (a Sunday), cut 23:00Z; cohort outputs in `live_vs_replay/` | One partial day; descriptive |
| D | New diagnostics on the L cohort, first minute: 41,471 timed public touch trades, 640,523 sh. Section 6.3 replays the timed maker legs: 94,783 legs, 1,074,957 sh; 753 untimed legs (9,131 sh) are excluded with a status | Guard out of sample; one partial day, so point estimates only, no interval (G < 5) |

Terms. *Touch trade*: a trade at the best price of the book before the trade. *Markout*: the maker's gain per share from
the trade price to the mid N seconds later (it includes the half spread earned), share-weighted, c/sh. *Whole-market
value*: markout times all shares traded, i.e. what all makers together earned; an order at the back of the queue does
not get it. *Gate on*: a side's guard value v is above the minute's pull line (PM90, 0.0492 c in minute 0), so v3 quotes
that side. *Trade report*: our receipt of the WS message that shows the trade; it lags the match by an unknown amount.
*Pro rata*: valued as if every maker at the price received a proportional share of each trade. *Band*: the range of v
between the enter line and a lower exit line.

## 1. Short answer

| Question | Answer |
|---|---|
| Does the logic hold? | Not as plain spread capture: the average resting order here earns +0.15 c/sh. It holds as selective quoting: the guard separates good from bad flow before the trade (+0.42 against -0.03 c/sh at 1 s, all makers pro rata). A 5-share order at the back of a queue of about 360-390 shares gets little of that: -0.05 to +0.10 c/sh at 1 s and +0.50 to +0.89 at 5 s in an order-level replay (section 6.3). Inventory is limited by size only, because the gate is on for one side at a time. |
| Live vs replay | The replay's short-lived fills are mostly not opportunities we had. The replay stamps a trade at a late message and credits orders placed after the match. 83-85% of v3's replay value at 1 s is in such orders. |
| Pull buffer | Sound churn control, no P&L. In form it is already a two-line rule: a band of 0.01 c below the line, at most 100 ms, the touch order of up to 5 sh. It was studied for mechanics only and is not in the live code. In the order-level replay it places 21% fewer orders at about the same value. It cannot close the gap. |
| Two lines with a wide band (the 06:27Z proposal) | Not supported; withdrawn. A 5-share order that joins the back of the queue and rests through the band is filled on the adverse part of the flow: its fills in the band make -0.32 to -0.51 c/sh at 1 s for bands of 0.08 c and wider, although all makers together make +0.24 to +0.29 there pro rata. Both wide-band arms are below v3 in $/day at every assumed trade time. Same result with v3's own price rule. |
| How to optimize | Measure rebates and rewards; calibrate fills (B1a, approved); check that 11-08 is typical. Queue position decides the sign of resting: the widest band makes -0.43 c/sh at 1 s with no credit for cancels ahead, -0.24 under the replay's queue rule, +0.73 at the front. No rule tested buys priority at our latency (6.5): a level born as the best shows 52 sh at birth and 155 sh 100 ms later, our order goes live behind 229-423 sh, and without a guard a 5-share order loses 0.8-2.6 c/sh in every position tested (with no queue ahead: -0.96 to +0.30 at 1 s). At the PM90 line the arms are positive if trades happen 0-50 ms before their report (15 of 16 cells) and negative at 250 ms (all cells), so the fill clock comes first. One candidate from 6.5: the touch order enters only at the step its price becomes the best (+0.58 c/sh at 1 s on 89 fills; declared reading MIXED). Every candidate needs at least 5 untouched UTC days. |

## 2. Development so far

| Step | Result | Status |
|---|---|---|
| Rounds before 2026-11-05 | Replay numbers from engines with errors | VOID, deleted; not quoted |
| Round i, fixed engines (docs §60-62) | Of more than 100 comparisons only "per-minute lines + ladder" is above 0 at both timings: mark at stop +$63.6/day J, +$52.0 X, 7 of 8 days. Week 1 +92.6 / +76.9, week 2 +8.4 / -0.4; 11-03 -12.2 / -33.1. Inventory cap and reduce-only: no gain in 36 comparisons | In-sample; value falls with distance from the fit |
| Freezes | v1 (38535aed), v2 daily refit (636450b5), v3 = v1 + CAP10 + RO60 + S240 (b261091d) | C1 forward race due, not run; C2 after 2026-11-14 02:00Z |
| Live, 11-08 | +$2.90 on 220 windows, 99 fill legs; replay on the same windows 386 / 337 fills | Decisions match; fills do not (section 4) |
| A1-A6 | Execution fixes, pmmmrust main 79a05d3 | In code, not deployed |
| B3 loop | Pull buffer, re-entry delay, incremental size, 50 ms updates | Mechanics only; no promotion |
| B9 | Four inventory-cost variants | None keeps 1 s value and mark at stop while lowering terminal inventory |

## 3. Does "capture spread, control inventory" hold?

### 3.1 The venue pays little for plain spread capture (P)

First minute, c/sh:

| Makers | sh/day | spread | price move against them in 1 s | net at 1 s |
|---|---|---|---|---|
| top 5 | 859,445 | +1.26 | -0.83 | +0.43 |
| ranks 6-20 | 790,909 | +0.96 | -0.58 | +0.38 |
| rest | 833,713 | +1.03 | -1.39 | -0.36 |
| all | 2,484,068 | +1.09 | -0.94 | +0.15 |

- All makers' net at 1 s by minute: +0.15 / +0.13 / +0.10 / +0.04.
- The sign depends on whether the trade clears the level. At-best fills in trades that leave the level standing (80% of
  at-best volume) net +0.65; in trades that clear it (20%) -1.23 (§33).
- The top 5 are not first in the queue more often than their volume share (30.7% of front shares, 32.0% of volume). They
  differ by filling behind the best (39% of their volume, 27.3 price levels per window), by losing less when a level
  clears (84% of their at-best volume is in trades that leave the level standing; the rest of the makers 73%), and by
  ending flat (leftover 11% of gross). Marked at the window's last mid they keep +0.10 c/sh (§34, §39).
- Fills behind the best pay more: +0.44 c/sh at 1 s one tick back and +1.06 two or more back, against +0.28 at the best
  (minute 0, §32; ms-timed legs only, about 55% of volume, which overstates the at-best number).
- On 11-08 all first-minute touch flow is worth +0.03 c/sh at 1 s, -0.09 at 5 s, -0.32 at 60 s, -0.85 at settlement (D).
- Taking costs about 1.5 c/sh at 50 c (fee 0.02 x min(p, 1-p) plus half a tick), more than any edge above. Entry and
  exit both have to be passive.
- Not measured: the maker rebate (25% of the taker fee under Predict's September promotion, unverified for November:
  0.25 x 0.02 x min(p, 1-p), i.e. 0.25 c/sh at 50 c and about 0.19 c/sh averaged over the prices traded) and liquidity
  rewards (listed terms: hourly rate 3000, within 6 c, minimum 100 sh; unit and payout not verified; v3's 5 / 25 sh
  quotes are below the minimum). Either could exceed the +0.15 c/sh trading edge.

### 3.2 The guard selects flow, and it does so before the trade (D)

Touch flow by the maker side's gate state, read at several leads before the trade report:

| Guard read | gate on: share of volume | on: 1 s / 5 s / 60 s | off: 1 s / 5 s / 60 s | on minus off at 1 s |
|---|---|---|---|---|
| 50 ms before | 13.0% | +0.70 / +1.53 / +1.76 | -0.07 / -0.33 / -0.64 | +0.76 |
| 250 ms before | 13.4% | +0.42 / +1.24 / +1.35 | -0.03 / -0.29 / -0.58 | +0.45 |
| 500 ms before | 13.0% | +0.24 / +0.72 / -0.12 | +0.00 / -0.21 / -0.34 | +0.24 |
| 1 s before | 13.3% | +0.22 / +0.61 / +0.47 | -0.00 / -0.20 / -0.45 | +0.22 |

Declared reading R1 = PRE-TRADE (difference at 250 ms >= 0.30 and at 500 ms >= 0.20). The 50 ms row may be read after
the match, so 250 ms is the row to rely on. Half of the difference is gone at 500 ms.

By guard value relative to the line, read 250 ms before the report:

| v - line (c) | share of volume | 1 s | 5 s | 60 s |
|---|---|---|---|---|
| below -0.30 | 15.2% | -0.53 | -1.26 | -2.05 |
| -0.30..-0.15 | 12.4% | -0.23 | -0.69 | -1.13 |
| -0.15..-0.10 | 10.3% | +0.09 | -0.17 | +0.80 |
| -0.10..-0.05 | 19.0% | +0.11 | -0.00 | -0.88 |
| -0.05..-0.02 | 17.8% | +0.09 | -0.18 | -0.65 |
| -0.02..0 | 11.7% | +0.34 | +0.63 | +1.29 |
| 0..+0.02 | 8.3% | +0.37 | +1.44 | +1.58 |
| +0.02..+0.05 | 3.2% | +0.40 | +1.10 | -0.28 |
| +0.05..+0.10 | 1.2% | +0.69 | +1.44 | +5.65 |
| above +0.10 | 0.8% | +0.69 | -0.56 | -0.88 |

Rank correlation of bucket order with the 1 s markout: +0.976 over the 10 buckets. Three things follow.

- The guard is informative out of sample on this day, and the information exists before the trade.
- The loss is concentrated in the bottom 28% of flow (v more than 0.15 c below the line). Between that and the line the
  flow is about flat at 1 s.
- The gate-on advantage grows after the fill (+0.42 at 1 s, +1.24 at 5 s). It is a short-horizon price forecast
  collected with passive orders, more than it is spread.

### 3.3 v3 does not collect it

| Fact | Value | Source |
|---|---|---|
| Gate on, share of a side's time | 12.5% | D |
| Gate spells per side per first minute | 31.9 | D |
| Spell length p50 / p75 / p90 | 100 / 250 / 500 ms | D |
| Live touch order life, p50 (two runs) | 49 / 55 ms | L, `who.json` |
| Size ahead of us when we join, p50 | 354 sh | L, `queue_misses.json` |
| Traded at our price while we rested, p50 | 11 sh | L, `queue_misses.json` |
| Age of the front order at the trades we missed, p50 | 2.0 s | L, `front_age.json` |
| Share of the level's life our side was pulled by the guard, before a missed trade | 47.5% | L, `queue_history.json` |
| First-minute touch volume in trades where we rested unfilled / were filled | 6.1% / 1.8% | D |
| The 45 first-minute touch trades that filled us (36 windows) | +0.07 c/sh at 1 s, +1.37 at 5 s; 1 s markout at least +0.5 c in 42.6% of shares, against 75.3% for gate-on flow read 50 ms before (73.4% at 250 ms) | D |
| Live ledger, all 99 legs, 849.9 sh | -0.84 c/sh at 1 s | L |
| Live P&L | +$2.90 (two runs: +14.59, -11.69) | L, `compare_summary.json` |
| Our share of maker volume | 0.023% | L, `who.json` |

Mechanism. The guard was fitted to the value of a fill by any touch trade, as if we received a proportional share of
every trade. The policy joins the back of the queue and leaves about 100 ms later. An order at the back is reached
mainly by trades that clear the level, the -1.23 c/sh kind in 3.1. Live filled 3.9 sh per window in total; 377 sh per
window traded at the touch in the first minute while our gate was on.

The 45-trade row has a small n and recovers by 5 s, so the 1 s number alone overstates the damage. The live ledger's
5 s and 60 s markouts were not tabulated (section 9).

### 3.4 Inventory is limited by size, not managed

| Fact | J / X | Source |
|---|---|---|
| Share of a side's gate-on time with the other side also on | 4.0% (reading R4 = ONE-SIDED) | D |
| Filled shares matched with an opposite fill in the same window | 51% / 48% | R |
| Matched (round-trip) P&L as a share of settlement P&L | 26% / 55% ($134.02 of $514.06; $173.54 of $317.63) | R |
| Mean \|inventory\| at the stop | 6.56 / 5.72 sh per window | R |
| Carry from the stop to settlement | -$163.06 / -$113.54 | R |
| Daily P&L: mean / standard deviation | $57.1 / $134.5; $35.3 / $103.1 | R |
| Peak \|inventory\| against a cap of 10 sh | 75 / 65 sh | R |
| Inventory time (60-240 s) with a legal closing quote | 1.25% | R, exit-feasibility audit |
| Inventory time the guard line blocks the closing side | about 86% | R, exit-feasibility audit |

v3 does not earn a two-sided spread. It takes one side when the guard favours it and carries about half of what it
fills to settlement. The closing side is blocked for the same reason the entry side is selective: the same high line
applies to both. Real top-5 makers end with 11% of gross (P).

Verdict on the logic: the spread is there, but it is paid to orders that (a) are not present when a level clears and
(b) rest behind the best for large takers. Inventory control cannot come from the entry signal, which is one-sided.

## 4. Live vs replay

Same 220 windows (L). Decisions agree on every aligned decision (21,495 of 21,495 J; 20,569 of 20,569 X). Fills do not.

| | Live | Replay J | Replay X |
|---|---|---|---|
| Fills | 99 legs | 386 | 337 |
| Shares | 849.9 | 2,343.1 | 1,841.4 |
| P&L | +$2.90 | -$10.92 | -$19.75 |

Where the replay's fills went in live (share of replay filled shares, J / X): live rested and was not filled 45.9 /
48.2%; live order refused by the net cap 23.7 / 18.2%; live was not quoting the side 15.3 / 10.8%; live filled too
14.6 / 20.7%. A1 (in code, not deployed) removes most cap refusals.

Replay fills by order age, 9 days (R):

| Order age at the fill | J: share of shares | J: c/sh at 1 s | X: share | X: c/sh at 1 s |
|---|---|---|---|---|
| cancel already sent | 42.2% | +1.16 | 41.6% | +1.05 |
| under 50 ms | 29.4% | +1.98 | 26.3% | +1.86 |
| 50-100 ms | 10.6% | +1.95 | 11.7% | +1.79 |
| 100-200 ms | 9.4% | +2.00 | 10.3% | +1.93 |
| 200-500 ms | 5.0% | +0.93 | 6.0% | +1.00 |
| over 500 ms | 3.4% | -0.46 | 4.0% | -0.52 |

- Orders under 100 ms old or already being cancelled hold 82% / 80% of replay filled shares and 85.4% / 82.9% of the
  value at 1 s. Live has no fill on an order under 100 ms old.
- Live's share of the replay fills where it held the same order rises with the order's life: 2.3% under 50 ms, 6.1% at
  50-100 ms, 31.8% at 100-200 ms, 50.8% at 200-500 ms, 77.9% over 500 ms (`lifetime_summary.json`).
- For the replay fills live missed (n 190), the trade report arrived a median 36 ms after live's order was accepted;
  85.8% arrived within 60 ms of acceptance or after the order was gone. For fills live also got (n 119): median 162 ms,
  24.4% (`feed_timing.json`).
- The replay's fills gain +0.52 / +0.38 c/sh from the price move in the first second. Every real maker group loses
  0.58 to 1.39 c/sh there (P).

Reading (an inference; match times are not observed): the replay stamps a trade at its report, which is later than the
match. An order placed between the match and the report is credited with a fill it could not have had, at a price that
already reflects the trade. That explains the young-order fills, their favourable markouts, and why they do not exist
in live. "Short-lived orders are hard to capture" is therefore the wrong frame: most were never available.
The replay's only fills that are safe from this, on orders over 500 ms old, lose 0.46 / 0.52 c/sh at 1 s.

Consequence: v3's in-sample replay P&L ($57 / $35 per day at settlement) is not evidence for the strategy. B1 / B1a
(calibration on live outcomes and a historical reference) stay first. The recorded rule stands: do not fix this by
fitting a delay or a minimum resting age.

## 5. The pull buffer (B3)

What it is: keep an accepted touch order of up to 5 sh for up to 100 ms while v is within a band below the line.
Replay mechanics on the 220 windows, no fills modelled (`20261109_soft_pull_results.md`):

| Band below the line (c) | requests | time at the original price | time resting off-target | active share-time |
|---|---|---|---|---|
| 0.005 | -4.43% | +1.61% | +18.09% | +1.89% |
| 0.01 (primary) | -6.99% | +2.65% | +33.87% | +3.40% |
| 0.02 | -9.08% | +3.59% | +56.46% | +5.35% |
| 0.04 | -9.75% | +3.93% | +72.00% | +6.54% |

At the primary band 4,783 of 13,077 touch pulls are kept, 2,376 of them reopen at the same price within the 100 ms,
and the mean age of the oldest active quote moves from 515 to 568 ms.

What the new diagnostics add (D):

| Touch flow | share of volume | 1 s | 5 s | 60 s |
|---|---|---|---|---|
| Gate off for up to 100 ms and v within 0.01 c below the line, four leads (561-625 trades each) | 1.8-2.1% | +0.13..+0.49 | +0.76..+1.35 | -3.00..+2.31 |
| Gate off for up to 100 ms, any v (declared reading R2', lead 250 ms) | 4.6% | +0.21 | +1.00 | -0.25 |
| Gate off for 100-500 ms, any v (lead 250 ms) | 10.2% | +0.06 | -0.03 | -1.15 |

- The flow the buffer sits through is not toxic at 1-5 s. R2' = GRACE-FAVOURABLE, but by 0.013 c over its threshold and
  not monotone in the lead (+0.42, +0.08, +0.21, +0.35 at 50, 150, 250, 500 ms): read it as "not harmful", not as an
  edge.
- Holding longer by the clock is not supported: flow 100-500 ms after the gate went off is flat at 1-5 s and negative
  at 60 s.
- It cannot close the live-vs-replay gap. It covers about 2% of touch volume against 13% with the gate on. It adds at
  most 100 ms to orders that are behind fronts 2 s old. Most of the gap is fills that did not exist (section 4).
- It barely changes residence. The band is half of one ordinary guard step (v moves 0.019 c per 50 ms at the median).
  Gate-on volume that trades at least 1 s into the state: 17.9% for v3, 23.8% with the buffer; spell p50 100 ms
  against 150 ms (section 6.2).
- In the order-level replay of section 6.3 the buffer places 21% fewer orders (38.4 to 30.3 per side-minute; 34.1 to
  26.9 with v3's price rule), fills about 11% more shares (961 to 1,068; 633 to 700) and leaves the value where it
  was: -$0.86 against -$0.58 per day at 1 s and +$6.14 against +$6.23 at 5 s (+$0.81 against +$0.84 and +$7.68
  against +$7.41 with v3's price rule).

Keep it as churn control with A2 (incremental size: requests -4.5%). Do not expect P&L from it. Whether a wider band
would pay is a separate question; section 6 answers it.

## 6. Two lines instead of one: what exists, the band, an order-level test (D)

A two-line rule turns a side on when v rises above an enter line and keeps the resting order until v falls below a
lower exit line. The 06:27Z version proposed it for the touch order: enter at PM90 as now, exit at PM50 (-0.048 c in
minute 0, "H2") or at -0.10 c ("H1"), no time limit. Between the lines (the band) the resting order is kept and
nothing is placed, enlarged or re-priced.

### 6.1 What already exists

| Item | Rule | Status |
|---|---|---|
| Frozen v3 and the live Rust build (`pmmmrust` `crates/pf-strategy/src/ladder.rs`, main 79a05d3) | One line per side (PM90): the order is pulled when v is at or below the line and placed again when v is above it. No band, no hold | What runs |
| Pull buffer B3, and LC50 (`20261109_soft_pull_results.md`) | Two lines in form: an accepted touch order of up to 5 sh is kept while v is within 0.01 c below the line (0.005 / 0.02 / 0.04 as sensitivity), for at most 100 ms (LC50: 50 ms). No new order in the band. Not the rung | Mechanics only, 220 windows of 11-08; no fill or P&L result; not in the live code |
| Re-entry delay 25 / 50 / 100 ms (`20261109_reentry_results.md`) | Wait before quoting again | Rejected: requests -8.5 / -13.8 / -19.5%, time at the target price -8.2 / -11.2 / -15.8% |
| A2 incremental size (`20261109_layered_size_keep_queue.md`) | A size-up adds a second order; the first keeps its place | In code (pmmmrust 70b965b, f20fdb1), not deployed; requests -4.5% |
| `LH` | A rung resting behind the best is pulled only by the Binance guards | Rejected in sample: J -46.0, X -57.8 $/day against pm + ladder, 0 of 8 days (§55). The base's value is mostly in orders under 100 ms old (section 4), so the comparison needs a re-read with the age split |
| `LL<key>` | A separate single pull line for the rung | Switch exists; no recorded run with a line other than the default |
| `IM<key>` | In a spread of 2 ticks or more, quote one tick inside when v is above a higher line | In sample (§55): J +25.4 (+2.0..+48.7), X +11.6 (-4.1..+27.2) $/day on pm; not in v3 |
| B5, B6, B7 (backlog) | Ladder retention; joining new levels first and resting longer; a guard target matched to how long orders rest | Open, not specified, after B1 |

Nothing that runs has two lines. The buffer has the form, with a band of 0.01 c (exit line +0.039 c in minute 0, still
above zero) and a 100 ms clock. Sections 6.2 and 6.3 test the same form with bands of 0.08-0.15 c and no clock.

v3's price rule is already a book filter. With a one-tick spread (96% of the time in minute 0) the MICRO target with G4
places the touch order only on the side whose best level is at least as large as the other side's. G3 withholds when
the next level behind is two or more ticks away.

### 6.2 The band: residence and value for all makers, pro rata

Lowering one line does not stop the flicker. A single line at any level switches 32-63 times per side per minute,
because the guard's noise crosses it (`gate_followup.txt`):

| Single line | on, share of side-time | spells per side-minute | spell p50 |
|---|---|---|---|
| PM90 (v3) | 12.5% | 31.9 | 100 ms |
| PM70 | 42.4% | 62.8 | 100 ms |
| PM50 | 63.1% | 53.0 | 150 ms |
| v > -0.10 c | 74.2% | 40.4 | 200 ms |

With a band below the enter line (PM90, +0.049 c in minute 0). State read 250 ms before the trade report; 41,471
trades, 640,523 sh, 220 windows; `gate_bands.txt`, as of 2026-11-10T06:44Z:

| Exit line (c), clock | on, share of side-time | spells per side-minute | spell p50 | state-on volume traded at least 1 s into the state | touch volume in state | c/sh at 1 s / 5 s | all makers, $/day at 1 s / 5 s | band flow: share of volume (trades) | band flow: c/sh at 1 s / 5 s |
|---|---|---|---|---|---|---|---|---|---|
| +0.049, no band (v3) | 12.5% | 31.9 | 100 ms | 17.9% | 13.4% | +0.422 / +1.236 | 472 / 1,384 | 0 | n/a |
| +0.039, 50 ms (LC50) | 13.7% | 24.9 | 150 ms | 22.6% | 14.4% | +0.406 / +1.218 | 492 / 1,475 | 1.1% (448) | +0.219 / +0.998 |
| +0.039, 100 ms (B3) | 14.0% | 23.3 | 150 ms | 23.8% | 14.9% | +0.387 / +1.263 | 483 / 1,575 | 1.5% (525) | +0.081 / +1.499 |
| +0.039, none | 14.3% | 22.5 | 150 ms | 24.9% | 15.2% | +0.386 / +1.242 | 492 / 1,582 | 1.8% (587) | +0.130 / +1.291 |
| +0.009, none | 19.8% | 12.5 | 400 ms | 45.5% | 21.1% | +0.355 / +1.003 | 628 / 1,776 | 7.8% (2,596) | +0.239 / +0.602 |
| -0.031, none | 26.4% | 9.7 | 700 ms | 60.8% | 27.7% | +0.353 / +0.827 | 820 / 1,922 | 14.4% (4,914) | +0.288 / +0.446 |
| -0.048 (PM50), none: H2 | 29.0% | 9.1 | 800 ms | 65.6% | 30.2% | +0.323 / +0.709 | 819 / 1,795 | 16.8% (5,747) | +0.246 / +0.291 |
| -0.100, none: H1 | 36.1% | 7.7 | 1,100 ms | 74.0% | 36.5% | +0.307 / +0.679 | 939 / 2,079 | 23.2% (8,207) | +0.241 / +0.358 |
| -0.101, 100 ms | 16.5% | 17.1 | 300 ms | 25.5% | 17.2% | +0.405 / +1.242 | 583 / 1,786 | 3.8% (1,427) | +0.347 / +1.263 |
| -0.101, 1 s | 26.4% | 9.6 | 1,100 ms | 54.2% | 27.6% | +0.387 / +0.882 | 897 / 2,043 | 14.3% (4,990) | +0.355 / +0.550 |
| never leave | 82.3% | 1.0 | 54.2 s | 98.0% | 82.4% | +0.043 / +0.060 | 300 / 415 | 69.1% (28,394) | -0.030 / -0.167 |

- Residence needs a wide band and no short clock. R8 = BAND-BOUND: a 0.04 c band with no clock has a spell p50 of
  400 ms and 45.5% of state-on volume at least 1 s into the state. R9: the smallest band that reaches 60% is 0.08 c.
  A 0.15 c band with a 100 ms clock gives 25.5%, about the buffer's 23.8%.
- Flow in the band is positive pro rata: +0.24 to +0.29 c/sh at 1 s for bands of 0.08 c and wider. Declared readings
  for H1 from the 06:27Z version: R6 = HOLD-OK (+0.241 / +0.358 at 1 s / 5 s), R7 = RESIDENT (74.0%).
- History matters pro rata. With v between -0.10 c and the enter line, flow in H1's kept state (the gate was on and
  v has not fallen below -0.10 c since) makes +0.24 / +0.36 at 1 s / 5 s; flow in the same range of v with the state
  off makes +0.08 / -0.16.
- The tail is adverse. R10 = TAIL-ADVERSE: after a first gate-on in the window, flow with v at least 0.15 c below the
  line (18.2% of volume, 9,580 trades) makes -0.439 / -0.786 / -1.750 c/sh at 1 s / 5 s / 60 s.
- H1's band flow turns negative later: -0.34 c/sh at 60 s, -2.40 at settlement. The other side is also on for 4.0%
  (v3), 7.4% (H2), 13.3% (H1) of a side's on-time.
- A third arm of the 06:27Z version (H3: enter at PM80) is in `gate_hysteresis.txt`; it was not taken further.

Limit. These are values for all makers pro rata. They show that the state would last seconds on flow that is positive
on average. They do not show what one order at the back of the queue would be filled on. Section 6.3 measures that.

### 6.3 Order-level test: one 5-share order through the same rules

`pf_review_queue_reach.py` puts one 5-share touch order per side per window through each rule in the first 60 s and
replays every timed maker leg against it. 220 windows, 440 window-sides, 0.764 window-days; 11-08, consumed, point
estimates only; `queue_reach.txt`, as of 2026-11-10T08:24Z.

- Clock: the replay's (order live after 40 ms, cancel effective after 22 ms, own fill known after 100 ms).
- Entry (configuration JOIN): when v is above the enter line and no order rests, the order joins the known best, and
  follows it while v stays above the line. In the band it is kept, also one tick behind the best; it is pulled when
  it is two or more ticks behind or alone in front. Nothing is placed, enlarged or re-priced in the band.
- Queue: the size shown at the price when the order goes live is ahead of it. Rule M is the replay's fill model
  (trades at our price use up the queue; a shrinking level credits cancels ahead; a trade through our price fills
  us). Rule L never credits cancels ahead. Rule U puts the order at the front: an upper bound, not a runnable arm.
- Trade time: each trade is placed 0, 50 or 250 ms before its report, because the match time is not observed
  (section 4). Not fitted. Primary configuration: M, 50 ms.
- Robustness V: v3's price rule (MICRO target, G4, G3) in place of "join the best". Declared after the JOIN table was
  seen and before its own table.
- Design, arms and readings R11-R14 were fixed in the script's header before the first table. Not modelled: the
  Binance guards (G1, G2), CAP10 / RO60, the 25-share size-up, the rung, the throttle.

JOIN, primary configuration:

| Arm | orders per side-minute | order life p50 / p90 (ms) | resting, share of side-time | fills | shares | filled by a trade through the price | c/sh at 1 s / 5 s | $/day at 1 s / 5 s |
|---|---|---|---|---|---|---|---|---|
| v3 (one line) | 38.4 | 82 / 332 | 11.2% | 274 | 960.9 | 58.7% | -0.046 / +0.495 | -0.58 / +6.23 |
| buffer B3 (0.01 c, 100 ms) | 30.3 | 132 / 532 | 13.0% | 297 | 1,067.9 | 54.6% | -0.062 / +0.439 | -0.86 / +6.14 |
| band 0.08 c | 18.8 | 232 / 2,152 | 24.6% | 645 | 2,388.0 | 42.8% | -0.149 / -0.054 | -4.65 / -1.68 |
| H2 (exit PM50) | 18.5 | 232 / 2,382 | 26.8% | 712 | 2,641.2 | 41.5% | -0.197 / -0.193 | -6.81 / -6.67 |
| H1 (exit -0.10 c) | 18.0 | 282 / 2,982 | 31.6% | 937 | 3,482.8 | 39.4% | -0.243 / -0.164 | -11.09 / -7.46 |
| never leave | 16.7 | 432 / 4,190 | 40.7% | 2,062 | 7,175.6 | 47.0% | -0.989 / -1.018 | -92.90 / -95.60 |

V (v3's price rule), same configuration:

| Arm | orders per side-minute | order life p50 / p90 (ms) | resting, share of side-time | fills | shares | filled by a trade through the price | c/sh at 1 s / 5 s | $/day at 1 s / 5 s |
|---|---|---|---|---|---|---|---|---|
| v3 (one line) | 34.1 | 82 / 382 | 10.0% | 183 | 633.2 | 64.1% | +0.101 / +0.894 | +0.84 / +7.41 |
| buffer B3 (0.01 c, 100 ms) | 26.9 | 132 / 532 | 11.5% | 195 | 700.2 | 59.6% | +0.088 / +0.838 | +0.81 / +7.68 |
| band 0.08 c | 17.9 | 182 / 1,682 | 18.6% | 317 | 1,176.2 | 51.5% | -0.092 / +0.232 | -1.41 / +3.57 |
| H2 (exit PM50) | 17.8 | 182 / 1,732 | 19.1% | 327 | 1,211.2 | 52.0% | -0.112 / +0.194 | -1.77 / +3.08 |
| H1 (exit -0.10 c) | 17.6 | 182 / 1,826 | 19.6% | 352 | 1,303.1 | 52.7% | -0.210 / +0.053 | -3.59 / +0.90 |
| never leave | 17.4 | 232 / 1,882 | 20.2% | 429 | 1,561.0 | 57.9% | -0.964 / -0.927 | -19.71 / -18.93 |

Queue ahead when the order goes live: p50 362-371 sh in every JOIN arm, 385-391 sh under V.

Fills by the state at the fill: shares (fills), c/sh at 1 s / 5 s:

| Arm | JOIN: v above the enter line | JOIN: in the band | V: v above the enter line | V: in the band |
|---|---|---|---|---|
| v3 (one line) | 782.8 (223), -0.072 / +0.241 | 5.0 (1) | 476.9 (141), +0.217 / +0.901 | 5.0 (1) |
| buffer B3 | 824.8 (229), -0.058 / +0.254 | 91.0 (23), +0.132 / +1.972 | 503.9 (143), +0.206 / +0.873 | 56.0 (14), +0.527 / +1.999 |
| band 0.08 c | 1,014.4 (277), +0.030 / +0.496 | 1,214.3 (326), -0.316 / -0.567 | 553.6 (154), +0.185 / +1.047 | 474.7 (126), -0.397 / -0.883 |
| H2 (exit PM50) | 1,044.4 (285), +0.053 / +0.462 | 1,426.5 (380), -0.369 / -0.691 | 553.6 (154), +0.185 / +1.047 | 499.7 (132), -0.402 / -0.894 |
| H1 (exit -0.10 c) | 1,117.0 (308), +0.119 / +0.556 | 2,168.9 (577), -0.414 / -0.541 | 558.6 (156), +0.187 / +1.043 | 581.6 (155), -0.507 / -1.115 |
| never leave | 1,240.0 (344), +0.255 / +0.786 | 5,756.1 (1,669), -1.236 / -1.333 | 558.6 (156), +0.187 / +1.043 | 804.6 (220), -1.549 / -2.075 |

The rest of each arm's fills came while a cancel was pending (JOIN 42-52 fills, V 37-53 fills per arm). v3 has no
band; its one fill in that column is a trade at the instant of the pull decision.

Sensitivity, c/sh at 1 s ($/day at 1 s):

| Arm | M, 0 ms | M, 50 ms (primary) | M, 250 ms | L, 50 ms | U, 50 ms (front; upper bound) |
|---|---|---|---|---|---|
| JOIN: v3 | +0.314 (+4.01) | -0.046 (-0.58) | -1.750 (-27.75) | -0.060 (-0.70) | +0.898 (+117.31) |
| JOIN: buffer B3 | +0.322 (+4.56) | -0.062 (-0.86) | -1.694 (-31.09) | -0.096 (-1.24) | +0.869 (+128.93) |
| JOIN: band 0.08 c | +0.016 (+0.48) | -0.149 (-4.65) | -1.332 (-54.23) | -0.273 (-7.19) | +0.771 (+175.48) |
| JOIN: H2 | -0.047 (-1.60) | -0.197 (-6.81) | -1.309 (-59.11) | -0.341 (-9.81) | +0.753 (+176.22) |
| JOIN: H1 | -0.099 (-4.30) | -0.243 (-11.09) | -1.235 (-72.43) | -0.428 (-15.43) | +0.725 (+179.63) |
| JOIN: never leave | -0.815 (-74.89) | -0.989 (-92.90) | -1.185 (-118.29) | -1.487 (-111.53) | +0.475 (+138.05) |
| V: v3 | +0.721 (+6.10) | +0.101 (+0.84) | -2.338 (-25.18) | +0.090 (+0.71) | +0.959 (+107.69) |
| V: buffer B3 | +0.698 (+6.55) | +0.088 (+0.81) | -2.243 (-27.53) | +0.054 (+0.47) | +0.927 (+118.56) |
| V: band 0.08 c | +0.252 (+3.82) | -0.092 (-1.41) | -1.818 (-39.55) | -0.173 (-2.41) | +0.855 (+153.93) |
| V: H2 | +0.253 (+3.92) | -0.112 (-1.77) | -1.811 (-40.60) | -0.195 (-2.77) | +0.849 (+154.70) |
| V: H1 | +0.223 (+3.63) | -0.210 (-3.59) | -1.808 (-42.44) | -0.303 (-4.60) | +0.835 (+154.68) |
| V: never leave | -0.278 (-5.21) | -0.964 (-19.71) | -1.784 (-45.71) | -1.125 (-20.62) | +0.729 (+139.93) |

Declared readings, computed in the script:

- R11 = RESIDENCE-COSTS. H2 and H1 against v3 under rule M, at trade-time shifts 0 / 50 / 250 ms, at 1 s and 5 s: all
  12 differences are negative, -$5.61 to -$44.68 per day. R11v (v3's price rule) the same: -$2.19 to -$20.10.
- R14 = HOLD-FILLS-ADVERSE. Fills taken in the band: H2 380 fills at -0.369 / -0.691 c/sh (1 s / 5 s), H1 577 fills
  at -0.414 / -0.541. R14v the same: 132 fills at -0.402 / -0.894, 155 fills at -0.507 / -1.115.
- R13 = QUEUE-DECISIVE. H1 at 1 s makes -0.428 c/sh under rule L and +0.725 at the front. R13v the same: -0.303 and
  +0.835.
- R12 = MIXED. H1's fills on orders at least 500 ms old (662 fills, 2,466 sh) make -0.307 c/sh at 1 s, and only 27.2%
  of those shares were filled by a trade through the price. Old orders at the back are adverse without being swept.

Reading. A share at the back of a ~360-share queue is reached after the queue ahead has been used up or by a trade
through the price; both happen when the level is about to go (3.1: -1.23 c/sh in trades that clear a level, +0.65 in
trades that leave it standing). The pro rata value of the band flow goes to the front of the queue. Above the enter
line the order at the back still makes about zero to slightly positive (-0.07 / +0.24 under JOIN, +0.22 / +0.90
under V, at 1 s / 5 s). In the band it does not.

Two post hoc cuts, not declared readings; inputs to the design of the next test:

- By the queue ahead when the order went live (JOIN, v3 arm): 0-25 sh ahead -1.903 c/sh at 1 s (12 fills), 25-100 sh
  -0.644 (51), 100-300 sh +0.090 (100), over 300 sh +0.280 at 1 s and +1.361 at 5 s (111). A thin level ahead is a
  level about to clear. v3's price rule already avoids the smaller side. Not computed for V.
- Value of the difference per added share against v3 (rule M): -0.20 to -0.32 c/sh at 1 s for bands of 0.08 c and
  wider at shifts 0 and 50 ms, -1.04 to -1.07 at 250 ms. A rebate of 0.20 (0.25) c on every filled share, the
  formula rate at 40 c (50 c), unverified, would make 1 (2) of 18 cells beat v3.

Limits.

- One partial Sunday, consumed; no interval.
- The queue rule is not validated (B1a); the test brackets it with L and U.
- Every level moves with the assumed trade time: the v3 arm goes from +0.314 to -1.750 c/sh at 1 s between 0 and
  250 ms (V: +0.721 to -2.338).
- Maker legs that the research tape could not time from the WS (their time is an estimate) are 32.0% of timed
  first-minute shares and 55.9% (V: 61.5%) of the v3 arm's filled shares. They are the adverse ones (matched touch
  trades: -0.498 c/sh at 1 s against +0.054 for WS-timed legs), and their times are the least certain.
- 60 s and settlement values do not separate the arms at 183-937 fills of 5 sh: v3 -1.171 / -2.164 c/sh and H1
  -0.083 / -0.791 under JOIN; v3 -0.057 / -0.529 and H1 -0.164 / -6.249 under V.
- The Binance guards and the rest of the engine are not modelled. The V price rule is checked against hand-computed
  cases, not against the engine's own decisions, which are not stored per decision.

### 6.4 What this changes

- The two-line touch gate is not supported at the order level. It is withdrawn as a recommendation. $/day at 1 s for
  v3, buffer, band 0.08 c, H2, H1: -0.58 / -0.86 / -4.65 / -6.81 / -11.09 under JOIN and +0.84 / +0.81 / -1.41 /
  -1.77 / -3.59 with v3's price rule.
- A full-engine run with a band token would close the remaining gap (Binance guards, sizes, rung, throttle). On this
  evidence it is not worth building.
- What pays is priority in a level that stays. The front-of-queue column is an upper bound: +0.73 to +0.96 c/sh at
  1 s for the arms from v3 to H1, against -0.24 to +0.10 at the back.
- Candidates that could earn priority (entering a new level first; rungs behind the best) were measured at the order
  level in revision 3 (section 6.5): not supported. A guard fitted to the value of the order we can place is still
  open (section 7, item 5).

### 6.5 Queue priority at the order level: new levels, a rung, improving inside the spread (revision 3)

Script `pf_review_level_birth.py`, committed before its first run (8f448870, 09:42:12Z); output as of 2026-11-10
09:47Z. Same cohort, clock, fill rule, queue rules and trade times as 6.3: 220 windows, 440 window-sides, 0.764
window-days, first 60 s, one 5-share order per side. Readings R15-R23 were written in the header before any table and
are evaluated in code. The cuts under "Post hoc" were added after the first table was seen and are marked.

Controls, evaluated before any table is printed. The join lane with v3's lines equals the 6.3 simulator fill for fill
on 2,200 window-side x config cells (0 differ) and reproduces the committed 6.3 rows (5 configs x 11 fields, 0
differ). With the cancel latency set to 30 ms, 40 of 40 window-sides differ, so the regression can fail. Eleven
planted errors are each caught by a synthetic case. Statuses: 753 untimed legs (9,131 sh) excluded, as in 6.3; 50 of
528,000 grid steps have no guard value (no order there); no window without frames or outcome.

| Lane | Rule |
|---|---|
| J (join) | Join the known best whenever there is no order. J-v3 is 6.3's JOIN v3 arm |
| B (birth) | Enter only at the grid step where the known best has just improved: a level born as the best. Keep while at the best or one tick behind |
| L (late) | Lane B's levels, entered 2 s after the birth. Control for R21 |
| R (rung) | One tick behind the known best, kept by the engine's LD2 rule. R-g90 is v3's rung rule |
| I (improve) | One tick inside a known spread of two ticks or more, the engine's `IM` rule |

Guards: none (always on), g10 (line -0.10 c), g50 (PM50), g90 (PM90). B-90/10 enters above PM90 and leaves at
-0.10 c.

**New levels, true time, both sides.** 11,481 levels were born as the best: 26.09 per side-minute, 72.0% of them one
tick better than the previous best. The best fell back 22.34 times per side-minute.

| | At birth | +100 ms | +200 ms | +500 ms | +1 s | +2 s |
|---|---|---|---|---|---|---|
| Shown size at the level, p50, levels still the best (sh) | 52.4 | 154.8 | 228.4 | 242.5 | 226.8 | 225.4 |
| Still the best (%) | 100 | 71.0 | 55.8 | 40.7 | 29.5 | 18.4 |

Life p50 293 ms, p90 3,553 ms. 40.2% of born levels trade while they are the best (33.7% within 1 s), 25.5 sh per
birth. Of the at-best maker volume in the first minute (732,649 sh), 50.6% trades at a level born as the best (13.7%
at a level under 0.5 s old) and 49.3% at a level that became the best by a fall-back.

**Arms.** Queue rule M unless stated; three numbers are trade-time shifts of 0 / 50 / 250 ms. $/day and the matched
share are at M 50. L-g90 (5 fills) is left out. "-0.00" is a negative number above -0.005.

| Arm | Orders per side-min | Queue ahead when live, p50 (sh) | Fills | Shares | c/sh 1 s, M 0 / 50 / 250 | c/sh 5 s, M 0 / 50 / 250 | c/sh 1 s, L 50 / U 50 | $/day 1 s / 5 s | Matched % |
|---|---|---|---|---|---|---|---|---|---|
| J-v3 | 38.4 | 363 | 274 | 960.9 | +0.31 / -0.05 / -1.75 | +0.96 / +0.50 / -1.72 | -0.06 / +0.90 | -0.58 / +6.23 | 33.3 |
| J-none | 48.8 | 158 | 9,077 | 30,612.9 | -1.20 / -1.32 / -1.67 | -1.45 / -1.60 / -2.01 | -1.62 / -0.07 | -528.80 / -639.92 | 81.6 |
| B-none | 23.9 | 229 | 3,468 | 11,693.1 | -0.85 / -1.01 / -1.25 | -0.84 / -1.05 / -1.41 | -1.31 / +0.30 | -155.17 / -160.78 | 83.8 |
| B-g10 | 16.1 | 304 | 928 | 3,299.7 | -0.03 / -0.20 / -1.18 | +0.32 / -0.00 / -1.01 | -0.28 / +0.68 | -8.62 / -0.12 | 51.7 |
| B-g50 | 14.4 | 325 | 514 | 1,830.9 | +0.18 / -0.01 / -1.26 | +0.49 / +0.02 / -1.31 | -0.05 / +0.79 | -0.16 / +0.47 | 38.6 |
| B-g90 | 7.2 | 423 | 89 | 301.7 | +0.99 / +0.58 / -1.10 | +1.81 / +1.10 / -1.04 | +0.60 / +1.17 | +2.30 / +4.34 | 25.2 |
| B-90/10 | 7.2 | 423 | 312 | 1,145.2 | +0.08 / -0.24 / -1.58 | +0.43 / -0.01 / -1.62 | -0.34 / +0.88 | -3.54 / -0.17 | 29.6 |
| L-none | 4.5 | 235 | 1,218 | 4,163.2 | -1.23 / -1.38 / -1.44 | -1.44 / -1.65 / -1.67 | -1.81 / +0.05 | -75.14 / -89.81 | 66.8 |
| R-none | 22.7 | 175 | 4,623 | 15,774.0 | -1.61 / -1.74 / -2.17 | -1.96 / -2.09 / -2.61 | -2.13 / -0.96 | -358.97 / -430.92 | 71.9 |
| R-g10 | 47.9 | 225 | 784 | 3,002.9 | -0.25 / -0.54 / -2.14 | -0.38 / -0.70 / -2.34 | -0.64 / +0.09 | -21.32 / -27.59 | 43.8 |
| R-g50 | 59.0 | 229 | 468 | 1,849.8 | -0.19 / -0.47 / -2.48 | -0.15 / -0.40 / -2.65 | -0.51 / +0.17 | -11.28 / -9.67 | 36.9 |
| R-g90 | 34.2 | 291 | 75 | 282.9 | +0.53 / +0.28 / -3.15 | +0.25 / +0.34 / -3.39 | +0.28 / +0.78 | +1.05 / +1.27 | 28.3 |
| I-none | 9.1 | 117 | 851 | 2,848.6 | -0.84 / -0.93 / -1.15 | -1.06 / -1.29 / -1.51 | -1.19 / +0.17 | -34.72 / -48.11 | 52.5 |
| I-g50 | 4.6 | 188 | 190 | 712.2 | -0.16 / -0.23 / -0.93 | -0.01 / -0.31 / -0.98 | -0.26 / +0.71 | -2.13 / -2.92 | 25.1 |
| I-g90 | 3.3 | 211 | 46 | 173.2 | +0.55 / +0.36 / -0.54 | +0.93 / +0.27 / -1.37 | +0.36 / +1.05 | +0.82 / +0.62 | 17.3 |

**Declared readings, as computed.**

| Reading | Result | Numbers |
|---|---|---|
| R15 a new level | B-none BIRTH-COSTS; B-g90 MIXED | B-none -0.84 to -1.41 c/sh in 8 of 8 cells. B-g90 +0.58 to +1.81 at shifts 0 and 50 (M and L), -1.10 and -1.04 at 250 ms |
| R16 guard at a new level | GUARD-ADDS | B-g90 minus B-none at 1 s: +1.84 / +1.60 / +0.14 c/sh |
| R17 residence at a born level | RESIDENCE-COSTS | B-90/10 minus B-g90: -$2.05 to -$27.41 a day in 6 of 6 cells. Same direction as R11 in 6.3 |
| R18 rung | R-none RUNG-COSTS; R-g90 MIXED | R-none -1.61 to -2.61 in 8 of 8 cells. R-g90 +0.25 to +0.53 at shifts 0 and 50, -3.15 and -3.39 at 250 ms |
| R19 improve | I-none IMPROVE-COSTS; I-g90 n/a | I-none -0.84 to -1.51 in 6 of 6 cells. I-g90 has 46 fills, under the declared minimum of 50 |
| R20 queue rule | B-none QUEUE-DECISIVE; R-none QUEUE-ROBUST | 1 s, L 50 against U 50: B-none -1.31 / +0.30; R-none -2.13 / -0.96 |
| R21 priority | PRIORITY-NOT-SHOWN | Same levels. B-none's orders 2 s or older: -0.824 / -0.955 / -1.005 c/sh at 1 s on 3,897-3,933 sh. Orders entered 2 s late: -1.227 / -1.379 / -1.437 on 4,007-4,291 sh. The older orders are +0.40 to +0.43 c/sh better on fewer shares, and both lose |
| R22 two-sided | MORE-TWO-SIDED for B-none, R-none, I-none | Matched share 83.8% / 71.9% / 52.5% against 33.3% for J-v3 |
| R23 scale | No arm is above J-v3 in all 6 $/day cells | B-g90 4 of 6; B-g50, R-g90, I-g90 3; I-g50 2; B-g10 1; the other arms 0 |

**What it says.**

1. Entering a new level early does not buy priority at our latency. The order goes live 44 ms after a decision made
   on a book seen 53 ms late, and a born level fills from 52 sh to about 155 sh in its first 100 ms. The queue ahead
   when the order goes live is p50 229 sh with no guard and 423 sh at PM90, against 363 sh for the v3 join. A short
   queue at entry does not mark a good fill: with 0-25 sh ahead the fills make -0.64 to -2.32 c/sh at 1 s in all 7
   arms that have 20 or more of them, and that is the worst bucket in 4 of the 7.
2. Without a guard every position loses. All 40 cells of the five unguarded arms (1 s and 5 s; M 0, M 50, M 250,
   L 50) are negative, -0.84 to -2.61 c/sh. With the queue ahead set to zero (U 50) the 1 s value is -0.07 (join),
   +0.30 (birth), +0.05 (late), +0.17 (improve) and -0.96 (rung) c/sh. Plain spread capture is not available to an
   always-present 5-share order, the front of the queue included. Fills taken while our price was in front of the
   true best (the others at the price had left; our cancel was not yet effective, or the rung's rule kept the order)
   make -3.16 to -3.57 c/sh at 1 s in the unguarded join, birth and rung arms.
3. The guard is what adds (R16), and holding below the line costs again (R17).
4. The rung under v3's own rule (R-g90) is the arm most exposed to the assumed trade time: +0.28 c/sh at 1 s at a
   shift of 50 ms, -3.15 at 250 ms. 87.6% of its filled shares are trades through its price, and its sign differs
   between odd and even windows (+0.51 | -0.02). Without a guard the rung loses under every queue rule.
5. Improving inside the spread: 23.3% of the unguarded placements are refused as crossing, the unguarded arm loses,
   and at PM90 there are 46 fills. This lane depends most on our own order being absent from the public book.
6. Two-sided flow comes only with the arms that are always on, and those lose 0.93 to 1.74 c/sh at 1 s. The guarded
   arms have a matched share of 17-52%. Quoting more does not net inventory at an acceptable price.
7. At the PM90 line 15 of 16 cells are positive at trade-time shifts of 0 and 50 ms (four lanes, 1 s and 5 s). At
   250 ms all 32 cells of the 16 arms are negative; the best is -0.54 c/sh. The fill clock decides the sign of every
   arm.

**Post hoc (not declared; candidates only).** Fills were split by the state of the known best at the decision step
of the order that was filled. The classifier is checked on the lanes: it puts 35,456 of 35,456 lane-B fills at a
birth step, 7,155 of 7,155 lane-L fills on a born level 2 s old, and 19,942 of 64,690 unguarded join fills at a
birth step. With exit = enter, arm B-g90 is by construction the part of the v3 join whose orders were entered at a
birth step, so the first row is a declared arm; the split into the other rows is post hoc.

| J-v3 fills by where the order was entered | Shares (fills), M 50 | c/sh 1 s, M 0 / 50 / 250 | c/sh 5 s, M 50 |
|---|---|---|---|
| At the step the level was born (= B-g90) | 301.7 (89) | +0.991 / +0.583 / -1.104 | +1.098 |
| Level born 50-450 ms before | 154.6 (39) | +0.153 / -0.169 / -0.542 | +0.388 |
| Level born 0.5-2 s before | 70.7 (19) | +0.765 / -0.387 / -1.790 | +0.131 |
| Level born 2 s or more before | 138.2 (40) | -0.260 / -0.500 / -2.586 | -0.260 |
| Best by a fall-back, under 0.5 s before | 156.9 (45) | -0.587 / -0.555 / -2.061 | -0.032 |
| Best by a fall-back, 0.5 s or more before | 138.9 (42) | -0.216 / -0.073 / -2.310 | +0.839 |
| All entries not at a birth step | 659.1 (185) | -0.090 / -0.334 / -1.936 | +0.220 |

- At 1 s the join's positive value is in the orders entered at the step a level is born. The other entries
  together are negative at 1 s at all three shifts; at 5 s they add +0.22 c/sh at M 50.
- B-g90 against J-v3: above per share in 10 of 10 cells (1 s and 5 s, five configs; arithmetic on two declared arms,
  not a declared reading). $/day at 1 s for shifts 0 / 50 / 250: +4.73 / +2.30 / -3.92 against +4.01 / -0.58 /
  -27.75. At 5 s: +8.62 / +4.34 / -3.69 against +12.26 / +6.23 / -27.23. It gives up 5 s value at shifts 0 and 50
  and removes most of the loss at 250 ms, on 31% of the shares. Same sign in odd and even windows (+0.56 | +0.62).
- Shown size at our price at the decision step does not order the fills: J-v3 -0.54 / -0.05 / +0.36 / -0.51 c/sh at
  1 s for 0-25 / 25-100 / 100-300 / over 300 sh. No size filter is supported.
- With a maker rebate of 0.19 or 0.25 c/sh (the formula; not verified on the account) the cells above zero, of 6,
  go from 1 to 3 and 4 for B-g10, from 3 to 4 for B-g50 and J-v3, and from 0 to 2 and 3 for I-g50. B-g50 fills 6.1
  times B-g90's shares at -0.01 c/sh at 1 s. No arm has a positive cell at 250 ms with either rebate.

Limits. One consumed partial Sunday; point estimates only. B-g90 has 89 fills and the post hoc rows 19-45. These are
single 5-share orders: the Binance and book guards, MICRO, G3 / G4, CAP10 / RO60, the size-up, the throttle and
inventory are not modelled, and lanes R and I are one order alone, not v3's ladder. The public book does not contain
our order. The fill rule is unvalidated until B1a. Young orders are the most exposed to the unknown match-to-report
lag. The 60 s and settlement columns are in the output and are not used in any reading.

What this changes. The two queue-priority candidates of revision 2 (entering a new level first or improving; rungs
behind the best) are measured and not supported on this day. What separates fills is the guard and, post hoc, the
state of the level when the order was entered. One candidate goes forward for a fresh-day test: the touch order
enters only at the step its price becomes the best (section 7, item 4).

## 7. Recommendations, in order

| # | Action | Why | Test or status | Backlog |
|---|---|---|---|---|
| 1 | Measure the maker rebate and the liquidity rewards on the account | By the formula the rebate is 0.19-0.25 c/sh, more than the +0.15 c/sh average trading edge. On 11-08's 849.9 live shares that is about $1.70-2.12 at 0.20-0.25 c/sh (arithmetic), against a P&L of +$2.90. The listed reward terms need 100 sh within 6 c; v3's 5 / 25 sh quotes do not qualify. If rewards pay, the objective moves from fills to resting size | The user's account data; keep out of P&L until measured | T5, D3 |
| 2 | Calibrate the replay fill model (B1a, then B1). Until then report every replay result split by order age at the fill, and treat value in orders under 100 ms old or being cancelled as unconfirmed. Re-read `LH`, `IM95` and the ladder with that split | Section 4: 83-85% of v3's replay value sits there. Section 6.3: the assumed match time alone moves the same arm from +0.31 to -1.75 c/sh at 1 s (+0.72 to -2.34 with v3's price rule), and the queue rule decides the sign of resting. Section 6.5: at a shift of 250 ms all 32 cells of the 16 arms are negative, and v3's rung rule is the most exposed (+0.28 c/sh at 1 s at 50 ms, -3.15 at 250 ms) | A reporting rule, not a fitted delay. B1a DONE (revision 4, `handoffs/20261110_b1a_live_orders.md`): the fill model is not validated at the order level (precision 0.08-0.25 under M and L; young-order replay fills 88-92% not live's), and no constant shift fits. B1 needs a new live capture with exact match clocks | B1, B1a, B2 |
| 3 | Run sections 3.2, 6.2 and 6.3 on the other consumed days outside the guard's fit: 11-03 (Tuesday), 11-07 (Saturday) | 11-08 is one Sunday; weekends have more gate-on time (§61); replay value fell from week 1 to week 2 | If the 250 ms on-minus-off difference is absent on 11-03, stop: no execution change can help. Needs guard values and timed legs built for those days | B11 |
| 4 | Queue priority: measured in 6.5 and not found. Entering a new level early, a rung without the guard, improving inside the spread and holding for residence are not supported on this day. One candidate remains: the touch order enters only at the grid step where its price becomes the best (arm B-g90), and does not join a level later or after a fall-back | 6.5: the order goes live behind 229-423 sh, and the unguarded arms lose in 40 of 40 cells. B-g90: +0.58 c/sh at 1 s and +1.10 at 5 s (M 50, 89 fills), above the v3 join per share in 10 of 10 cells, -$3.92 a day at 1 s at 250 ms against -$27.75. Its declared reading is MIXED (negative at 250 ms), and the comparison with the other entries is post hoc | After item 2. First repeat 6.5 on 11-03 and 11-07 (item 3). Then the full engine with an entry condition on the touch order and the age split, then at least 5 untouched UTC days. It changes the frozen rule: a new candidate, to be flagged | B5, B6 |
| 5 | Refit the guard on the value of an order we can place: an order placed now at the back of the queue (0 if unfilled; one row per decision) | The present label values every touch trade pro rata. In the band the two differ in sign: +0.24 to +0.29 c/sh pro rata, -0.32 to -0.51 for the order (1 s). 6.5, post hoc: the state of the level at the decision (born or by a fall-back, age) separates the join's fills, so it belongs among the inputs | After item 2, because it needs a validated queue rule. It changes the model: flag it against the pinned comparison | B7 |
| 6 | Reducing side: solve the line, the legal price and the minimum size together | 3.4: the entry gate blocks the closing side about 86% of inventory time; single relaxations failed (RX5, B9). No exit is clearly cheap: a passive fill from the back of the queue in the band costs 0.32-0.51 c/sh at 1 s and 0.54-1.12 at 5 s (6.3); taking costs about 1.5 c/sh; carrying to settlement cost 0.96 c (J) / 0.77 c (X) per share held at the stop in the replay (arithmetic: $163.06 / (2,592 x 6.56 sh), $113.54 / (2,592 x 5.72 sh); uncalibrated fills, indicative) | The research session's next step. Judge by terminal inventory and daily variance at equal 1 s value, not by the mean alone | B9 follow-up |
| 7 | Pull buffer: keep as churn control only. Do not widen it into a general two-line gate | Sections 5 and 6.3 | No further tuning | B3 (closed) |
| 8 | Run C1 as declared; C2 after 2026-11-14 02:00Z | Declared forward races | Read them as replay under an uncalibrated fill model, with the age split | C1, C2 |
| 9 | Read what the profitable makers do: split each large maker's fills by guard state, by the state of the level (born or by a fall-back, age) and by time since the level was born | 6.5: an always-present order loses in every position, yet the top 5 makers make +0.43 c/sh at 1 s on 859,445 sh a day (P). If their fills sit in states our guard already marks, the gap is speed or queue; if not, it is signal (item 5) | Read-only on the cohort's timed legs, which carry a `signer` column (to be confirmed as the maker's address). Readings declared first; descriptive | B12 |

Item 4's two original candidates were measured in 6.5 and are not supported. What remains of item 4 is one arm whose
declared reading is MIXED, plus a post hoc split. Items 5, 6 and 9 are hypotheses from data already read.

Not recommended now: larger sizes before item 2 (20 / 100 sh: J +92.5, X +64.4 $/day in sample, with no own price
impact in the replay). A Polymarket hedge was not examined here.

## 8. Size of the prize (D)

- All makers together: first-minute touch flow with the gate on was worth $472 per day at 1 s and $1,384 at 5 s on
  11-08 (220 windows scaled to 288).
- The top 5 makers' first-minute net is about $3,700 per day at 1 s (859,445 sh at +0.43 c/sh, P; arithmetic).
- One 5-share order per side in the first minute, from the back of the queue: -$0.58 per day at 1 s and +$6.23 at 5 s
  (+$0.84 and +$7.41 with v3's price rule). The rebate on the same fills would be $1.7-3.1 per day at 0.20-0.25 c/sh
  (arithmetic; rate unverified).
- The same order at the front of the queue: $108-117 per day at 1 s with v3's line, $155-180 with H1's residence. These
  are upper bounds; no rule here gets the order to the front.
- Section 6.5 confirms that: an order entered at a level's birth still goes live behind p50 229-423 sh. Entering only at
  the birth step at PM90 makes +$2.30 a day at 1 s and +$4.34 at 5 s (M 50, 5 shares), against -$0.58 and +$6.23 for
  the join.

At this size the trading edge from the back of the queue is about -$1 to +$7 a day. Items 1 and 2 decide whether the
build is worth more work; item 4 no longer offers a way to the front.

## 9. Not verified

- Rebates and liquidity rewards (T5, D3).
- The match time behind each trade report. The section 4 reading is an inference, and every level in 6.3 depends on it.
- The queue rule. 6.3 brackets it (L, M, U); B1a tests it.
- 6.5 measures new-level entry, the rung and improving as single 5-share orders on one day. Not verified: the same
  rules inside the full engine (the ladder, sizes, Binance guards), on other days, and with our own order in the book.
- Whether `signer` in the cohort's timed legs is the maker's address (item 9).
- 6.3 leaves out the Binance guards, the cap and reduce-only rules, the size-up, the rung and the throttle. Its v3
  price rule is checked against hand-computed cases only.
- Live fills' 5 s and 60 s markouts. The ledger sits in the raw capture directory, which also holds 11-09 windows; it
  was not read.
- Everything labelled D is one partial Sunday. The guard's skill on weekdays and in later weeks is open (item 3).
- The Polymarket lead-lag study (HANDOFF 07:47Z) is the research session's; it was not checked here.

## 10. New diagnostics: declared readings, self-tests, reproduce

All seven read only derived cohort outputs (`touch_trade_reasons.parquet`, `queue_misses.parquet`,
`research/step5/20261108/preds_t60k.parquet`, `research/step4/minute_lines_t60k.json`); the queue replay and the level-birth script also read the
cohort's book frames, registry, clock file and timed maker legs under `research/step1/20261108/`. Readings were written
in each script's header before its table was seen, except the cuts marked post hoc. Outputs:
`live_vs_replay/review_20261110/` under the cohort directory (`*.json`, `*.txt`).

| Script (`predictfun/research/`) | Question | Declared readings | Self-test |
|---|---|---|---|
| `pf_review_queue_miss_age.py` | Missed queue trades by our order's age | PRE-TRADE, **not used**: the population is selected on our order surviving the trade, so it cannot test the guard | Reproduces `queue_misses.json`; a copy with 10% of rows dropped is refused |
| `pf_review_gate_information.py` | Guard state against all touch flow, by lead | R1 PRE-TRADE; R2 BETWEEN; R3 BETWEEN (30.8% of gate-on volume after 500 ms of continuous gate-on) | Gate state agrees with the existing "pulled by prediction" status on 100.00% of 33,374 trades; 42.00% with the guard's sign flipped |
| `pf_review_gate_followup.py` | Buffer zone at larger leads, line frontier, two-sidedness, spells | R2' GRACE-FAVOURABLE (marginal); R4 ONE-SIDED (3.98%); R5 PERSISTENT (74.6%) | Reproduces the previous script's lag-50 difference (+0.762825); +0.086 with trade times shuffled within the window |
| `pf_review_gate_hysteresis.py` | Two-line rules, pro rata | R6 HOLD-OK; R7 RESIDENT (74.0%) | PM90 / PM90 reproduces the previous script's gate-on row (85,552.4 sh, +0.421669); +0.043 with shuffled trade times |
| `pf_review_gate_bands.py` | Band width and clock: residence and pro rata value | R8 BAND-BOUND; R9 0.08 c; R10 TAIL-ADVERSE | Reproduces the hysteresis script's three rows to 6 decimals; a clock of 0 equals no band and an infinite clock equals no clock on every grid point; agrees with a step-by-step state machine on 40 window-sides; +0.016 with shuffled trade times |
| `pf_review_queue_reach.py` | One 5-share order through each rule, at the order level | R11 RESIDENCE-COSTS; R12 MIXED; R13 QUEUE-DECISIVE; R14 HOLD-FILLS-ADVERSE; R11v, R13v, R14v the same with v3's price rule. Two post hoc cuts, marked | Synthetic books with known answers; v3's price rule against hand-computed cases; unsorted input refused; five planted errors each caught; identity control on 22,363 touch trades (same 1 s value in 95.4%); gate-on control +0.552 against +0.422 (tolerance 0.15). The first version of the second control failed (07:47Z) and is recorded in the header |
| `pf_review_level_birth.py` | Queue priority at the order level: new levels, a rung, improving (6.5) | R15 B-none BIRTH-COSTS, B-g90 MIXED; R16 GUARD-ADDS; R17 RESIDENCE-COSTS; R18 R-none RUNG-COSTS, R-g90 MIXED; R19 I-none IMPROVE-COSTS, I-g90 n/a; R20 B-none QUEUE-DECISIVE, R-none QUEUE-ROBUST; R21 PRIORITY-NOT-SHOWN; R22 MORE-TWO-SIDED; R23 printed. Post hoc cuts, marked, with their own instrument control | Committed before its first run (8f448870). Synthetic books for every lane; unsorted input refused; eleven planted errors each caught; the join lane equals the queue replay on 2,200 window-side x config cells and reproduces its committed rows to 1e-9; with a 30 ms cancel latency 40 of 40 window-sides differ |

```bash
D=/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108/live_vs_replay/review_20261110
R=predictfun/research
$R/run_light.sh review_queue_miss_age   -- python3 $R/pf_review_queue_miss_age.py   $D/queue_miss_age.json
$R/run_light.sh review_gate_information -- python3 $R/pf_review_gate_information.py $D/gate_information.json
$R/run_light.sh review_gate_followup    -- python3 $R/pf_review_gate_followup.py    $D/gate_followup.json $D/gate_information.json
$R/run_light.sh review_gate_hysteresis  -- python3 $R/pf_review_gate_hysteresis.py  $D/gate_hysteresis.json $D/gate_followup.json
$R/run_light.sh review_gate_bands       -- python3 $R/pf_review_gate_bands.py       $D/gate_bands.json $D/gate_hysteresis.json
$R/run_light.sh review_queue_reach      -- python3 $R/pf_review_queue_reach.py      $D/queue_reach.json
$R/run_light.sh review_level_birth      -- python3 $R/pf_review_level_birth.py      $D/level_birth.json
```

Each run's stdout was saved as `$D/<name>.txt`. Copies of all fourteen output files, with their SHA-256 sums, are in
`predictfun/reviews/strategy_review_20261110/`.
