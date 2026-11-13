# DEPTH-TIMING: why the top makers' fills behind the best pay and our rungs there lose (results, 2026-11-13)

Session ce68ebe0. USER: "If deep ladder doesnot work, why top makers do so? Check if it still the timing issue, I don't think it is live
vs replay issue, top makers still make money in replay and their performance is much better".

Script `research/pf_depth_timing.py` + `run_depth_timing.sh` (586efcba, committed before the first run; 80ec5108 fixed a missing column
after a 4 s crash, before any output). 07:37-07:46Z, claude slot.

## Design

Both sides on the same days, the same tape and one value function:
- **Population:** 9 consumed replay days, first 60 s, WS-timed trades whose taker order is exact and complete (ORDERS' map).
- **Real side:** every maker leg, grouped as top 5 / field.
- **Our side:** the 30-sh rungs 1-5 ticks behind the best (FRONT-PULL-SIZE's LZ-none and LFPQ rungs; rule M 50). Each fill inherits the
  real taker order's type.
- **Per fill:**
  - depth (ticks behind the best before the trade);
  - type: STOP (the taker order stopped at this price, the level held) or THROUGH (it went past);
  - value at 1 s / 5 s;
  - our P at t - 24 ms;
  - the move in the second before.

Controls passed:
- The type cases.
- C1 for every rung (the never-pull callback equals the plain lane).
- The value function is shared by both sides.

Statuses:
- 773,905 real legs.
- 221,219 of our fills classified.
- 357,611 of our fills on trades that are not WS-timed or not complete were excluded by the same rule as the real legs.

## Answer

c/sh at 1 s, depth >= 1 (share of shares by type in brackets):

| Who | All | STOP | THROUGH | Move before (c), STOP / THROUGH |
|---|---|---|---|---|
| Top 5 (real) | **+0.89** | +0.94 (64%) | +0.80 (36%) | -1.84 / -1.72 |
| Field (real) | +0.58 | +0.46 (60%) | +0.75 (40%) | -1.74 / -1.48 |
| Ours, never pulled | **-1.10** | +0.16 (37%) | -1.85 (63%) | -2.25 / -3.85 |
| Ours, FPQ at the best | -1.22 | +0.04 (34%) | -1.87 (66%) | -2.40 / -3.84 |

By depth, c/sh at 1 s:

| Who | At the best | 1 tick | 2 ticks | 3+ ticks |
|---|---|---|---|---|
| Top 5 | +0.40 | +0.58 | +0.98 | +1.77 |
| Field | +0.24 | +0.34 | +0.68 | +1.33 |
| Ours, never pulled | -0.88 | -1.16 | -1.03 | -1.03 |

Depth >= 1 by tercile of our P (P <= 0.090 / <= 0.424 / above): share of shares; c/sh STOP / THROUGH:

| Who | Low P | Mid P | High P |
|---|---|---|---|
| Top 5 | 42%: +1.45 / +1.00 | 39%: +0.92 / +0.65 | 19%: -0.01 / -0.19 |
| Field | 46%: +1.49 / +1.03 | 34%: +0.92 / +0.58 | 20%: -1.74 / -0.57 |
| Ours, never pulled | 21%: +1.41 / +0.85 | 28%: +0.71 / +0.12 | **50%: -1.35 / -3.64** |

Readings (declared, evaluated in code):
- **R420 WITHIN-TYPE.**
  - Gap top 5 minus ours +1.99 c/sh, positive on 9 of 9 days (+1.54 to +2.73).
  - 73% of it is within the same trade type and 27% the mix.
  - The largest part: their THROUGH fills +0.80 against our -1.85.
- **R421 MORE-STOPS.** 64% of the top 5's deep shares come from trades that stop at their price, against 37% of ours.
- **R422 PRESENT-WHEN-SAFE.** 42% of the top 5's deep shares come at low P, against 21% of ours. Half of ours come at high P, against 19%
  of theirs.

## What it means

1. **Deep ladders work for real makers.**
   - Behind the best, the top 5 earn +0.6 to +1.8 c/sh, rising with depth.
   - Everyone else earns +0.3 to +1.3.
   - Being behind the best pays when a sweep overshoots and the price comes back.
2. **Ours lose because of when they are there, not because of the replay's valuation.**
   - In the same tercile of our own signal, our fills are worth about what the real makers' fills are worth (low P: +1.41 against +1.45
     on STOP fills).
   - Our rungs stay through the high-risk moments. The real makers are not there then: they pulled or had not posted (cf. EXIT: their
     front orders are gone 79% of the time at high P).
   - Half our deep shares are filled at high P, at -1.35 / -3.64 c/sh, after the price has already run against us (-3.85 c in the
     second before our THROUGH fills, against -1.72 for theirs).
3. **So it is the timing of the pull, now for every rung.**
   - FRONT-PULL-SIZE pulled deep rungs only once they were at the best, which is too late for a sweep through several levels.
   - The test that follows: pull and block every rung when P is high.

## Limits

- Replay rule M for our fills (B1a).
- 62% of our fills (and the matching real legs) were excluded as not WS-timed or not complete. The comparison is on the same kind of
  trade, not on all of them.
- Consumed days; designed after FRONT-PULL-SIZE (exploratory).
- The P terciles are computed on the real deep fills of these days and are descriptive.

Outputs: `data/predict_fun/research/depth_timing_20261113/` (depth_timing.json / .txt, fills_*.parquet, day_*.json, logs); copies in
`reviews/depth_timing_20261113/`.
