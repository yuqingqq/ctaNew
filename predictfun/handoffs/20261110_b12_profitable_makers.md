# B12: what the profitable makers do (2026-11-10)

As of 2026-11-10 16:47Z (session ce68ebe0). Item B12 of `handoffs/20261109_TODO.md` (strategy review item 9). Script
`research/pf_b12_makers.py`, committed and pushed before its first run (d5f68cf2; output 16:43Z). Post hoc split
`research/pf_b12_posthoc.py`, written after reading that output (16:46Z), marked post hoc below. Output in the cohort
directory `live_vs_replay/b12_20261110/`, copies in `reviews/b12_20261110/`.

Public tape only; our fill model is not used. Descriptive. One consumed partial day (the 220 windows of 2026-11-08, cut
23:00Z, 0.764 window-days): point estimates only.

## Question

The top makers net about +0.4 c/sh at 1 s in the first minute, while an always-present 5-sh order of ours loses in every
position (review 3.1 and 6.5). Are their fills in states our guard already marks (then the gap is speed or queue), or
not (then it is signal, B7)?

## Input checks (before the script was written)

- `signer` in the timed legs is the maker's address. On chain, the funder (`maker`) equals the `signer` on all 19,671
  orders of the 10-23 14:00-14:15 slice (1,645 addresses, one to one). The REST `signer` is among the transaction's
  chain makers for 11,492 of 11,492 legs (2.9% against a wrong transaction). Our account is one id (200). It holds the
  fill-clock audit's 99 own legs and 849.867 sh, exactly live's REST total; with the column shuffled this check fails.
- Maker fee in the public match records: each maker leg carries a `fee` field, 0 on all 41,804 legs of 10-23 14h and
  all 45,351 legs of 11-08 03h and 15h. No leg is negative. A maker rebate, if any, is not paid inside the match record;
  only the account can show it (D3 stays a USER item). The taker fee, where charged in collateral, equals 0.02 x min(p,
  1-p) per share (p50 ratio 1.00).

## Design

Maker legs of the first minute with a research trade time (one time per taker order, as the review's replay). Makers
ranked by shares in 0-240 s over the 220 windows, ours excluded: top 5, ranks 6-20, rest. O = ranks 6 and below. Per
leg: value at 1 s and 5 s from the trade price, split into spread (to the mid just before) and move. Also, from the
frame before the trade, the guard state of the maker's side (ON if v > PM90, read at the trade time minus 0 / 50 /
250 ms), the depth behind the best and the level's state (born as best or fall-back, age). Finally, whether the taker
order went through the level (swept). Readings R30-R36 and the null N1 are declared in the script header.

Controls, all passed before any table: the copied level code reproduces pf_maker_behavior's seven cases. Guard lookup,
sweep flag and the decomposition identity pass, and three planted decompositions recover their parts. The null flags a
planted +1 c/sh skill and not its absence. Refusals pass, and a NaN reads as n/a. On the data: K1 the first-minute tape
statuses equal the replay's (753 untimed legs / 9,131.4 sh; 94,783 timed / 1,074,957.4 sh); K2 the own id as above; K3
the minute-0 PM90 line is 0.0492 c.

Statuses: 753 untimed legs excluded. No leg lacks a mid, no window lacks frames or guard values. At the best (lag 50),
the warm-up placeholder covers 2.0% of the top 5's volume and 4.8% of O's; 'sweep unknown' (no taker order id) 8.4% and
6.4%. All are reported, none is in the decomposition.

## Results

**The makers on 11-08** (first minute, all depths, c/sh):

| Makers | n | sh/day | spread | move 1 s | net 1 s | net 5 s | $/day at 1 s |
|---|---|---|---|---|---|---|---|
| top 5 | 5 | 462,118 | +1.33 | -0.96 | +0.36 | +0.27 | +1,676 |
| ranks 6-20 | 15 | 463,040 | +0.87 | -0.68 | +0.19 | +0.09 | +877 |
| rest | 373 | 481,098 | +1.12 | -1.45 | -0.33 | -0.46 | -1,590 |
| all but ours | 393 | 1,406,256 | +1.11 | -1.04 | +0.07 | -0.04 | +963 |
| ours | 1 | 961 | +0.94 | -1.37 | -0.44 | -0.26 | -4 |

The pattern of the 8 public days (3.1: +0.43 / +0.38 / -0.36) holds on this day. All five top makers are positive at
1 s (+0.12 to +0.71).

**By depth** (top 5 against O; spread / move / net at 1 s, c/sh; split post hoc):

| Depth | top 5: share of volume | top 5 | O: share | O |
|---|---|---|---|---|
| at best | 59.1% | +0.53 / -0.37 / +0.17 | 72.6% | +0.53 / -0.60 / -0.07 |
| 1 tick behind | 19.8% | +1.55 / -1.31 / +0.24 | 14.2% | +1.54 / -1.73 / -0.19 |
| 2+ behind | 19.2% | +3.76 / -2.76 / +1.00 | 10.9% | +3.77 / -3.89 / -0.12 |
| inside the spread | 1.9% | -0.99 / +2.29 / +1.30 | 2.3% | -0.86 / +1.47 / +0.61 |

The top 5 make $1,104 of their $1,676 a day behind the best ($886 two or more ticks back) and $459 at it. The rest of
the makers lose behind the best (-0.56 and -0.54 c/sh). The spread is the same for every group at every depth; the
difference is the move after the fill.

**At the best, by our guard state** (read at the trade time minus 50 ms; net at 1 s, move in brackets; split post hoc):

| Makers | OFF share of volume | ON | OFF |
|---|---|---|---|
| top 5 | 86.1% | +0.94 (+0.42) | +0.02 (-0.51) |
| ranks 6-20 | 83.3% | +0.75 (+0.24) | +0.03 (-0.48) |
| rest | 88.1% | +0.63 (+0.07) | -0.38 (-0.92) |
| ours | 34.7% | -0.25 (-0.76) | -1.98 (-2.48) |

The spread at the best is +0.50 to +0.56 c/sh for every group and state. The guard separates good flow from bad for
everyone (ON beats OFF by 0.7 to 1.0 c/sh). The top makers trade 86% of their at-best volume while our guard reads OFF,
as much as everyone else. Our own fills (86 legs) have the worst move in both states.

**Where the at-best gap comes from** (top 5 against O, c/sh at 1 s):

| Guard lag | D | MIX (all) | MIX guard | MIX sweep | WITHIN |
|---|---|---|---|---|---|
| 0 ms | +0.20 | +0.01 | -0.00 | +0.01 | +0.19 |
| 50 ms | +0.20 | +0.01 | -0.00 | +0.01 | +0.19 |
| 250 ms | +0.20 | +0.01 | -0.01 | +0.01 | +0.19 |

Null N1 (500 permutations of the top-5 label within window, side and cell): observed WITHIN +0.19, null p50 +0.01, p99
+0.09; no draw at or above. Post hoc: WITHIN = spread +0.01 + move +0.18. It sits in the trades that do not go through
the level: moves of -0.43 against -0.62 with the guard OFF (73.5% of the top 5's volume) and +0.53 against +0.25 with it
ON. In trades that go through the level the moves are equal (-0.97 against -1.01; -0.15 against -0.14).

**Speed at new levels.** The top 5 take 25.7% of at-best volume on levels under 1 s old and 26.4% of all age-classified
at-best volume (ratio 0.97). They take 25.6% of born-as-best volume against 28.5% overall (0.90).

**Declared readings, as computed.**

| Reading | Result | Numbers |
|---|---|---|
| R30 edge | TOP-EARNS | top 5 +0.36, O -0.08 c/sh at 1 s (difference +0.44) |
| R31 source | WITHIN-STATE at lags 0, 50, 250 | D +0.20; WITHIN +0.19; MIX +0.01 |
| R32 guard off | lag 0 OFF-ADVERSE-FOR-TOP, lags 50 / 250 OFF-NOT-ADVERSE; NO-AVOIDANCE at all three | top 5 OFF -0.00 / +0.02 / +0.10 c/sh; OFF share 86.5 / 86.1 / 86.4% against O 85.8 / 85.7 / 84.9% |
| R33 depth | DEPTH-FED | behind the best +$1,104 a day, at the best +$459 |
| R34 speed | PROPORTIONAL | 0.97 |
| R35 birth | PROPORTIONAL | 0.90 |
| R36 clock | GUARD-CLOCK-SENSITIVE | only R32's sign changes, around zero (-0.001 at lag 0) |
| N1 null | WITHIN-REAL | +0.19 against p99 +0.09 |

## What it says

1. The top makers earn on this day as on the 8 public days: +0.36 c/sh at 1 s on 462,118 sh a day, +$1,676 a day in
   the first minute. The rest of the field loses.
2. Their edge is not in states our guard marks. They trade while our guard reads OFF as much as everyone else (86% of
   their at-best volume). The guard-state mix explains none of their at-best advantage (0.00 of +0.20 c/sh). The
   advantage is within the same guard state and trade type (+0.19, above the null).
3. They do not earn by price. Their spread is the same as everyone's at every depth. They earn by which trades fill
   them: the move after their fills is smaller, at the best (+0.18 c/sh) and behind it (-2.76 against -3.89 two ticks
   back). The difference sits in trades that leave the level standing; in trades that go through it, everyone is hit
   alike.
4. They are not faster to new levels or more often on levels born as best (ratios 0.97 and 0.90).
5. Two thirds of their first-minute dollars are behind the best. Depth does not pay by itself: ranks 21 and below lose
   behind the best. The difference is again the move after the fill.
6. Our own first-minute fills (86 legs, 734 sh) have the worst move of any group, in both guard states: -0.76 c/sh with
   the guard ON, where the field makes +0.07 to +0.42. With the guard OFF, where v3's rule is to pull (so these fills
   came before a pull took effect), it is -2.48 against the field's -0.48 to -0.92.

Which trades fill a maker depends on its place in the queue, on how fast it cancels before informed flow, and on what
it knows. The public tape cannot separate the three. It does exclude two explanations: guard-state selection and speed
to new levels.

## What this changes

- Signal (B7): the guard already separates flow for everyone (ON beats OFF by 0.7 to 1.0 c/sh). The top makers' extra
  is within the guard state and comes from smaller moves after their fills. That needs either an input our guard lacks
  or a better queue place. B12 cannot tell which. A guard refit (B7) is not supported as the first lever.
- Execution: in the states where v3 quotes (guard ON), our fills move -0.76 c/sh against the field's +0.07 to +0.42 at
  the same spread. The loss is in which trades fill us: queue place and cancel timing. This is what B1 measures; it
  needs a live run with the A4b capture (USER D2).
- Depth (B5): the profitable makers' larger share is behind the best, and it pays only for the top 20. A rung behind
  the best is worth testing only with its cancel behaviour (when it pulls) in the test, after B1.
- Rebate (D3): no rebate inside public match records (maker fee 0 on 87,155 legs, 10-23 and 11-08). Only the account can
  show whether one is paid elsewhere.
- Next on the public tape (proposed, not run): split the trades that fill the top 5 against O within the same cell by
  taker-order size and by our queue-ahead estimate (B12b). If the top 5 are filled by smaller trades, the gap is queue
  place; if by the same trades at other times, it is cancel timing or information.

## Limits

One consumed partial Sunday; point estimates. Trade times are the replay's (WS pending message, else the E2 estimate),
off in both directions for our own legs (fill-clock audit, B1a): level ages under ~250 ms and the guard state at lag 0
are uncertain, and R32 changes sign around zero between lags. The book contains our own orders. These are values of the
flow the makers received, not of an order we can place. Queue position inside a level is not observed. 'Sweep unknown'
legs (no taker order id) are excluded from the decomposition.

## Addendum 2026-11-11 02:38Z (post hoc; no new run)

Written to answer the USER's question: what separates our fills from the top makers' at the same price, and are their
profits from layered quotes. Arithmetic on the committed `b12.json` / `b12_posthoc.json` only
(`research/pf_b12_qa_derive.py`; output `reviews/b12_20261110/b12_qa_derive.txt`, not in SHA256SUMS). Descriptive; our
group is 86 legs on one day and has no null.

- Our first-minute loss is the fills taken at the best while the guard reads OFF (lag 50): 25 legs, 160.0 sh, -1.98 c/sh
  at 1 s, -$4.16 a day, 99% of our first-minute -$4.18 a day. The rest nets to about zero: ON at the best -$0.99, one
  tick behind -$0.0 (27 legs), two or more behind +$1.0 (2 legs). Live's own record has fills of this kind: 218.0 of
  849.9 live fill shares have a research trade time after our cancel request (B1a, M 50). Whether the 25 legs are among
  them is not checked here.
- With the guard ON we make -0.25 c/sh at the best where the top 5 make +0.94 (32 legs; $4.72 a day at our ON volume).
  39.2% of our at-best shares come from trades that go through the level, against 14.5% for the top 5 and 17.0% for
  ranks 6 and below (sweep-known). An order at the back of the queue is reached mostly by trades that use the level up,
  so this is consistent with our place in the queue (the book shows about 360 sh ahead when the v3 join goes live,
  review 6.3); it does not prove it. In trades that stop at the level, the top 5 make +1.05 c/sh with the guard ON,
  ranks 6 and below +0.78, we -0.41.
- Layers: 66% of the top 5's first-minute dollars are behind the best, but their per-share advantage is at every depth,
  and depth alone loses (ranks 21 and below: -0.56 one tick back, -0.54 two or more). A third of our first-minute shares
  fill one tick back (v3 has a rung there): -0.01 c/sh against the top 5's +0.24 and ranks 6 and below -0.19 (27 legs).
  Two of our legs fill two or more back.
- Ranking: the first loss to remove is the fills after the guard turns OFF (pull timing), before queue place or a
  deeper layer. B1 (live run with the A4b capture, USER D2) measures it order by order. Proposed, not run: join the 25
  legs to live's order paths (B1a) to count how many came after our cancel request.
