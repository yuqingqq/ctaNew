# B12f: Binance activity at the top makers' takes, and Predict imbalance and queue (2026-11-11)

As of 2026-11-11 06:48Z (session ce68ebe0). USER "Yea check Binance activity during those times, and imbalance etc".
Script `research/pf_b12f_activity_imbalance.py`, committed and pushed before its run (6a51ba39; output 06:46Z). Output
in `live_vs_replay/b12f_20261111/`, copies in `reviews/b12f_20261111/`.

Public tape, chain match records, and the HF collectors' Binance BTCUSDT trade and bookTicker files for 11-08 (receive
clock of this box). One consumed partial day (the 220 windows of 2026-11-08): point estimates only.

## Design

Part A: the same 2,491 taker orders of makers 48 / 51 / 661 as B12e, against 12,455 random times (same windows and
sides) and other takers. Binance features over the second before (signed in the Predict taker's favour): aggressor
flow (buy minus sell BTC), top-of-book size imbalance, trades and quote updates. Part B: all at-best maker fills with
the guard known, split by Predict's book on the maker's side (thin: size at the best at most half the other side's;
heavy: at least twice); D_thin = the thin-side gap against the rest within guard x guard-value quintile cells; heavy
levels split by reach (the share of the shown level the trade took). Readings R80-R87 in the script header. Controls
passed (B12e's, window sums, aggressor sign, side classes, the D_thin null on planted fills, label rules); event
counts equal B12e's.

## Results

**Part A, Binance at their takes**

| Second before the take | their takes | random times |
|---|---|---|
| Binance trades | 45.7 (median 4) | 21.6 |
| Binance best bid / ask updates | 426 (median 107) | 199 |
| Top-of-book size imbalance their way (-1..1) | +0.17 (58% > 0) | +0.04 (51% > 0) |
| Aggressor flow their way (BTC) | +0.50 (52% > 0) | +0.37 (43% > 0) |
| Aggressor flow over the last 250 ms (BTC) | +0.42 | +0.05 |
| Flow over the next 0.5 s (BTC) | +0.26 | +0.12 |

Differences: imbalance +0.129 (null p99 +0.025); trades +24.0 (p99 +8.9); updates +227 (p99 +21); flow over the second
+0.13 (p 0.35, SD 15.7); over 250 ms +0.37 (p 0.004, 0.13 SD); after +0.15 (p 0.11).

**Part B, Predict's book on the maker's side** (all makers except ours, at-best fills; share / net 1 s / net 5 s):

| Guard | thin side | balanced | heavy side |
|---|---|---|---|
| ON | 6% / +0.13 / +0.63 (565 legs) | 24% / +0.43 / +0.87 | 71% / +0.94 / +1.87 |
| OFF | 33% / -0.72 / -1.17 | 38% / -0.01 / -0.24 | 29% / +0.43 / +0.59 |

- D_thin within guard x guard-value cells: -0.58 c/sh (null p01 -0.18, p50 -0.10).
- Heavy levels: fills from trades that took more than half the shown level +0.30 c/sh (3,710 legs), the rest +0.68
  (15,764). Median reach on heavy levels: the field 0.199, ours 0.930.
- Ours (net 1 s, legs): guard ON thin -1.00 (2), balanced -0.75 (4), heavy -0.14 (26); OFF thin -2.32 (12), balanced
  +0.50 (2), heavy -2.00 (11).

| Reading | Result |
|---|---|
| R80 flow lead | NO-FLOW-LEAD |
| R81 Binance book lead | BN-BOOK-LEAD (+0.129) |
| R82 activity | BN-ACTIVE (2.1x the trades) |
| R85 Predict imbalance incremental to our guard value | IMBALANCE-INCREMENTAL (-0.58 c/sh) |
| R86 thin side when the guard is ON | THIN-OK-WHEN-ON (+0.13) |
| R87 heavy-level queue | NO-QUEUE-SPLIT (back +0.30, front +0.68) |

## What it says

1. Binance does show something at their takes, but not in the price: Binance is about twice as busy (trades and quote
   updates), and its best bid / ask sizes lean their way (+0.17 against +0.04). The price itself had moved under 1 bp
   before 94% of their takes (B12e v2). That is book pressure before a tick; aggressor flow does not clearly lean.
2. With B12e: they take when both books press one way, Binance's top of book and Predict's (a thin side against a heavy
   other side), and Predict's price has started to move. The feature table our models are trained on has these inputs
   (step5 rows_rl_d2: bn_imb, bn_flow250/1000/5000, bn_ntr1000, pf_imb1, pf_imb3, pf_dm250/1000/5000; the model files do
   not name which of their 32 columns are used), and our model points their way at 81% of these moments (B12e).
3. Predict's imbalance still carries information our guard value lacks: at the same value, thin-side fills are 0.58 c/sh
   worse. Most of it sits where our guard is OFF (thin -0.72, heavy +0.43). With the guard ON the thin side is 6% of the
   volume and still earns a little (+0.13, against +0.94 on the heavy side).
4. Being behind on a heavy level does not lose in general (back +0.30, front +0.68), but we sit at the very back: the
   trades that fill us took a median 93% of the shown level, against 20% for the field. Our heavy-side fills lose with
   the guard ON (-0.14, 26 legs) and badly with it OFF (-2.00, 11 legs: the late / slow-cancel fills of B12b).

## What this changes

- Guard (B7, reframed): the signals the top makers act on are book pressure on Binance and on Predict, which our
  feature table has. The test is whether the guard weights them enough: at the same guard value, Predict's thin side
  loses 0.58 c/sh more. A declared guard-input test needs untouched days for validation (at least 5).
- v3: skipping the thin side would mostly cut exposure where the guard is already OFF; with the guard ON it would give
  up fills that earn +0.13. The larger lever for us is where we sit in the queue on heavy levels (reach 0.93).
- The slow-cancel issue (B12b) still explains our guard-OFF heavy-side losses; it stays with the live session.

## Limits

One consumed Sunday; point estimates; three SMART addresses. Values are pro rata flow, which ranks moments and does not
price an order we can place. Predict trade times are reports (~0.2 s after the match); the Binance receive clock is
assumed to be the Predict collector's. The Binance imbalance is the best bid / ask sizes only. Our rows are 57 legs.
