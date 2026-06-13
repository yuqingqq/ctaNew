# Handoff: review → evaluation
iteration: iter-022 (ALPHA track — sideways-regime XS-reversal ensemble re-rank)
fix-round: 1
verdict: **PASS** (G1 look-ahead audit PASS). The implementation REJECT is trustworthy.

## The two critical conclusions

1. **Is rel_ret_1d strictly PIT (so the −0.036 IC is real, not leakage)? → YES, PIT, IC is REAL.**
   Verified empirically against raw klines: `return_1d[t] = close[t]/close[t−288]−1` (corr=1.0,
   trailing, window `[t−288,t]` ending at decision bar t) and the forward target
   `return_pct[t] = close[t+48]/close[t]−1` (corr≈1.0, `exit_time−open_time` = exactly 48 bars,
   window `[t,t+48]` starting at t). The two windows share ONLY the boundary price `close[t]`, known
   at t. No overlap, no look-ahead. The cross-sectional demean is within-cycle (groupby open_time),
   PIT. Independently reproduced the pre-check IC = **−0.0360 (t −9.76, n 2458)** exactly — far below
   the +0.10 leak red-flag, and the re-rank LOSES gross (opposite of a leak signature).

2. **Is the ensemble sign/logic correct (so the REJECT is a real economic result, not a bug)? → YES.**
   - Sign verified empirically: long basket mean rel_ret_1d = −0.10 (recent losers), short = +0.07
     (recent winners) → longs the reversal-favored side; `−z(rel)` not doubled/flipped.
   - z_xs is per-cycle cross-sectional (computed over symbols in cycle t only).
   - **Base reproduces X117 exactly** (@4.5bps Sharpe +1.93 / maxDD −5674 / totPnL +10472); X131
     held-book engine is line-for-line the X117 engine; only the side rank key changes; bull/bear
     untouched (non-side PnL differs only by legitimate HOLD-window bleed, max 0.028).
   - **Gross-PnL collapse reproduced independently** by extrapolating net-vs-cost to cost=0:
     base gross +12272 → ens +7241 (−41%). The loss is NOT a cost artifact; pre-cost return falls.

## Why the strong orthogonal IC doesn't monetize (one-line)
The production `pred` book is already a beta-neutral XS mean-reversion basket; `−z(rel_ret_1d)` ranks
the same names from a noisier, higher-turnover angle that trades AGAINST `pred`, so the cross-sectional
reversal alpha is already absorbed by `pred` and overlaying it destroys gross return. (iter-018 mode.)

## To Evaluation
Run the gate tables as built (G2/G4/G5/G6/G7/G8). Emitted numbers are internally consistent with the
parquet artifacts. Expected verdict: **REJECT** — a clean, faithful build of a genuinely PIT,
orthogonal, era-stable XS-reversal IC that fails to monetize through the held book (gross-PnL down
41–57% on all three universes, G4 p0, G7 EXT per-episode 1/4, G6 EXT CI clears 0 NEGATIVE). A useful
clean negative; no implementation defect inflated or destroyed the result.

No fixes required.
