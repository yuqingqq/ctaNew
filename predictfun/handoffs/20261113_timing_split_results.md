# TIMING-SPLIT: where the guard ladder loses - estimated-time trades (results, 2026-11-13)

Session ce68ebe0. USER: "We can run it, it still looks we don't manage deep ladder well, the timing is also not good".

Script `research/pf_timing_split.py` + `run_timing_split.sh` (298390f7, committed before the first run). 09:16-09:26Z, claude slot.

## Design

- **Check:** FINDINGS C12, skipped in every ladder study today until now.
- **Simulator:** DEEP-PULL's. Arms: the guard ladder (LDG: 30 sh x 6 rungs per side, guard trigger on every rung), the never-pulled
  ladder and J-v3. 9 consumed replay days, first 60 s, fill clocks M 0 / 50 / 250.
- **Per simulated fill:**
  - the trade's timing: WS = its taker order has a WS-timed leg, so the time is exact; E2 = estimated, about +-250 ms;
  - depth;
  - at our price or through it;
  - young (< 100 ms) or not;
  - filled while our cancel was pending;
  - the P tercile;
  - value at 1 s / 5 s, the same function as DEPTH-TIMING.
- **Real maker legs** of the same windows: by group, depth and timing.

Controls passed:
- The self-tests and the source-flag case.
- C1 at every rung.
- P aligned on all rows.
- The loss-map cells add up.

## Answer

$/day at 1 s, all / without young fills (c/sh without young in brackets), and share of shares:

| Arm, clock | All trades | Exact (WS) | Estimated (E2) |
|---|---|---|---|
| LDG, M 0 | +15 / -16 | **+102 / +86 (+0.42)**, 54% | -87 / -102 (-0.62), 46% |
| LDG, M 50 | -137 / -137 | **+59 / +54 (+0.24)**, 53% | -196 / -190 (-1.00), 47% |
| LDG, M 250 | -2,071 / -1,785 | -1,084 / -936 (-1.87), 59% | -987 / -849 (-2.48), 41% |
| Never pulled, M 50 | -9,852 / -8,548 | -5,069 / -4,342 (-1.45) | -4,783 / -4,207 (-2.08) |
| J-v3, M 50 | +14.7 / +3.7 | +5.2 / +1.7 (+0.73), 37% | +9.5 / +2.0 (+0.59), 63% |

Real maker legs, c/sh at 1 s (share of the group's shares):

| Group | At the best, exact | At the best, estimated | Behind, exact | Behind, estimated |
|---|---|---|---|---|
| Top 5 | +0.38 (67%) | -0.08 (33%) | +0.86 (73%) | +0.30 (27%) |
| Field | +0.20 (57%) | -0.58 (43%) | +0.49 (65%) | -0.43 (35%) |

LDG at M 50, fills on orders >= 100 ms old that were not pending (the bulk):

| Cell | Exact (WS) | Estimated (E2) |
|---|---|---|
| At the best | +$7.0 a day (+0.05 c/sh) | -$75.1 (-0.76) |
| Behind the best | **+$64.4 (+0.92)** | -$74.4 (-1.04) |
| Behind, by P tercile: low / mid / high | +1.07 / +0.74 / -1.78 c/sh | -0.64 / -1.46 / -2.40 c/sh |
| Behind, at our price / through it | +1.14 / +0.33 c/sh | +0.69 / **-1.51** c/sh |

Fills while our cancel was pending: -$8.7 a day (exact) and -$30.7 (estimated). Young fills are about neutral.

Readings (declared, evaluated in code):
- **R450 E2-DRIVEN.**
  - LDG's loss without young fills (-$137 a day at M 50) is entirely on estimated-time trades: -$190 there, +$54 on exact ones.
  - Estimated trades are 46% of LDG's shares.
- **R451 WS-POSITIVE.**
  - On exactly-timed trades LDG earns +$54 a day at M 50: positive on 7 of 9 days (+54, +182, +160, +58, +19, +19, -9, +1, -2).
  - At M 0 it earns +$86.
- **R452:** real makers' fills also look much worse on estimated trades, at every depth.
- **R453:** the loss map above.

## What it means

1. **On trades whose time we know, the guard ladder works.**
   - +$54 a day at the 50 ms clock and +$86 at 0 ms.
   - Its fills behind the best earn +0.92 c/sh, about the top makers' +0.86 on the same kind of trades.
   - The deep-ladder management (keep through fills, pull every rung on the guard) is not where we lose.
2. **The losses sit on estimated-time trades** (about half our shares). Two things mix there:
   - **The estimate itself.** A +-250 ms anchor error moves the measured value. Real makers' fills there also drop sharply (top 5 at the
     best +0.38 -> -0.08, field +0.20 -> -0.58).
   - **The fill decision.** The simulator decides whether our order was still there by that uncertain time. Our deep fills that swept
     through the level on those trades (-1.51 c/sh) are the largest loss, and they are worse than either real group's.
3. **The gap at the best remains:** exact-time aged fills +0.05 c/sh against the top 5's +0.38.
4. **Speed still matters.** At the 250 ms clock even the exact trades lose (-$936 a day).
5. **This qualifies today's ladder results** (FRONT-PULL-SIZE, DEEP-PULL, KEEP-PLACE). Their totals mix both kinds of trade, and the
   estimated half drives their losses.

## Next (proposals, the user's call)

- **Better trade times for the estimated half.** Check which trades get no WS timing (size, depth, how they are matched) and whether
  another public feed times them (`predictTopOfBook`, not yet recorded).
- **The live probe of the guard ladder.** Live fills carry exact match times for our own orders, which removes the estimated half for us.
  Measure: share of fills at our price vs swept, value at 1 s, at the best vs behind it.
- **The gap at the best** (exact +0.05 against +0.38): queue place / timing at the best level.

## Limits

- Replay rule M (B1a). Consumed days.
- The WS-only view is a selection (half the trades). Estimated-time trades exist live too and their true value is unknown.
- The split uses K.load_day's taker-order grouping (unidentified legs are their own groups).

Outputs: `data/predict_fun/research/timing_split_20261113/` (timing_split.json / .txt, fills_*.parquet, day_*.json, logs); copies (json / txt)
in `reviews/timing_split_20261113/`.
