# BEST-GAP: what causes the gap at the best price - timing, latency, size or queue place? (results, 2026-11-13)

Session ce68ebe0. USER: "The gap at the best price, is it mostly the timing issue, or latency, or size? We have been working on this,
still not sure about the cause".

Script `research/pf_best_gap.py` + `run_best_gap.sh` (a4767fc1, committed before the first run). 09:36-09:43Z, claude slot.

## Design

- **Population:** fills at the best (depth 0 before the trade) on exactly-timed trades (complete, WS-timed taker orders; ORDERS' map).
  9 consumed replay days, first 60 s. Value: sg (mid(t + 1 s) - price), the same function for real and ours.
- **Ours:** S4's rest lane, rung 0, kept through fills, guard trigger (LDG's rung 0); fills on orders >= 100 ms old (aged).
- **Levers:**
  - SIZE: 5 / 100 sh against 30.
  - LATENCY:
    - FAST: 5 ms order path and the book known at its stamp + 15.2 ms;
    - ZERO: 0 ms and known at its stamp, a bound.
    - The guard value keeps its live clock.
  - QUEUE: fill rule U, nothing ahead of us (a bound).
  - TIMING: our c/sh within P terciles re-weighted to the top 5's tercile mix.

Controls passed:
- BASE equals DEEP-PULL's LDG rung 0 fill for fill.
- The latency constants are restored after every arm.
- C1.
- The synthetic context cases.

## Answer

c/sh at 1 s, fill clock M 50 (aged fills):

| Who | c/sh | STOP (the level held): c/sh, share | THROUGH (the level was taken): c/sh, share | Fills at low / mid / high P |
|---|---|---|---|---|
| Top 5 (real) | **+0.405** | +0.53, 80% | -0.09, 20% | 70% / 24% / 7% |
| Field (real) | +0.239 | +0.36, 71% | -0.05, 29% | 72% / 19% / 9% |
| Ours, BASE (30 sh) | **+0.007** | +0.47, **37%** | -0.26, **63%** | 78% / 19% / 3% |
| Ours, 5 sh | -0.001 | +0.55, 37% | -0.33, 63% | |
| Ours, 100 sh | +0.010 | +0.48, 32% | -0.21, 68% | |
| Ours, FAST | +0.118 | +0.59, 41% | -0.20, 59% | |
| Ours, ZERO | +0.141 | +0.61, 41% | -0.19, 59% | |
| Ours, nothing ahead (U) | +0.667 | +0.75, 92% | -0.26, 8% | |

Gap G = +0.398 c/sh. What each lever closes:

| Lever | Change in our c/sh | Share of G |
|---|---|---|
| Size (best of 5 / 100 sh) | +0.003 | 1% |
| Latency, FAST (ZERO) | +0.111 (+0.133) | 28% (34%) |
| Timing (the top 5's risk mix) | -0.133 | -33% |
| **Queue place (nothing ahead, a bound)** | **+0.660** | **166%** |

Per day, queue place is the largest lever on 9 of 9 days (+0.52 to +0.85). Latency adds +0.03 to +0.30. Size ranges from -0.04 to
+0.10. Timing is negative every day. At M 0: base +0.122, gap +0.283, queue +0.660.

**R464: MAIN: QUEUE** (declared rule: the largest lever, >= 50% of G).

## What it means

1. **The gap at the best is which trades reach our order.**
   - The top makers' at-best fills are 80% trades that stop at their level. They are near the front, so the small trades fill them.
   - Ours are 37% such trades and 63% level-taking ones: we are behind, so mostly the sweeps reach us.
   - Within each kind of trade our values are close to theirs: STOP +0.47 against +0.53.
   - With nothing ahead of us, our at-best fills would be 92% STOP and earn +0.67 c/sh, more than theirs.
2. **Latency is the second lever, about 30%.** A 5 ms order path and the fastest feed lift our at-best fills from +0.007 to +0.118 c/sh.
   They lift the STOP share a little (37% -> 41%) and cut fills while a cancel is pending (4% -> 1%).
3. **Size does nothing at the best** (5, 30 and 100 sh all about 0 c/sh).
4. **Timing is not the problem at the best.** We are already at the best in lower-risk moments than they are (78% of our fills at low P
   against their 70%). Re-weighting to their mix makes ours worse.
5. **So the question is how to be nearer the front of the best level.**
   - Speed captures about a third of it.
   - The rest comes from being in the level before others: they were there first and stayed.
   - Rule M (B1a) assumes where our order is; the U bound shows the value if we were first.

## Next (proposals, the user's call)

- **FRONT-ENTRY (replay), on rung 0 with the guard pull, 30 sh:**
  - join a new best level at its birth, event-driven with the FAST path;
  - create the level by improving one tick inside a spread of 2+ ticks.
  - Measure the STOP share and value at the best on exactly-timed trades.
  - Earlier: CS found joining at births gains only with a 10-20 ms path; TB found improving puts us first but adds little at 5 sh
    with v3's gate.
- **The live probe:** our real STOP share at the best is the single number that tells us how far back our real orders are.

## Limits

- Replay rule M (B1a). U and ZERO are bounds.
- FAST / ZERO change our order path and book view, not the guard value's clock or the 50 ms decision grid.
- The levers are not additive. Consumed days.
- Exactly-timed complete trades only (about half the volume).

Outputs: `data/predict_fun/research/best_gap_20261113/` (best_gap.json / .txt, fills_*.parquet, day_*.json, logs); copies (json / txt) in
`reviews/best_gap_20261113/`.
