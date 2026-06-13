# Review — iter-022 (ALPHA track: sideways-regime XS-reversal ensemble re-rank), fix-round 1

**Verdict: PASS.** The REJECT is trustworthy. rel_ret_1d is strictly PIT (the −0.036 IC is real,
not leakage) and the ensemble sign/logic is correct (the gross-PnL collapse is a real economic
result, not a bug). Evaluation may proceed to the gate tables.

Script: `research/convexity_portable_2026-05-20/scripts/X131_xs_reversal_ensemble.py`

---

## Critical check 1 — PIT / look-ahead of rel_ret_1d (DECISIVE for IC validity): CLEAN

`rel_ret_1d[sym,t] = return_1d[sym,t] − groupby(open_time)['return_1d'].mean()` — X131.py:159-161,
verbatim the pre-check (`iter022_leadlag_transport_precheck.py:38-40`, `iter022_orth_fast.py:13`).

**Verified empirically against raw klines (`data/ml/test/parquet/klines/`):**
- `return_1d[t] == close[t]/close[t−288] − 1` to numerical precision — **corr = 1.0** vs
  `close.pct_change(288)` at open_time t (panel_hl70, BTCUSDT). It is a TRAILING 1-day (288×5m)
  return whose window is `[t−288, t]`, ENDING at the decision-bar close `close[t]`. It is NOT an
  extra-shifted variant (corr to shift(±1) drops to 0.996; to shift(−288) drops to −0.05).
- The forward target the book trades, `return_pct[t] == close[t+48]/close[t] − 1` — **corr ≈ 1.0**
  vs `close.shift(−48)/close − 1`; and in the HL70 preds file `exit_time − open_time = exactly 48
  bars` (median/mode 48). Its window is `[t, t+48]`, STARTING at `close[t]`.
- **No overlap.** The trailing-1d window `[t−288, t]` and the forward target window `[t, t+48]` share
  ONLY the boundary price `close[t]`, which is known at decision time t. No look-ahead.
- **z_xs is cross-sectional per-cycle** (X131.py:123-135 `zscore`, applied within `ge` = symbols in
  cycle t only at :241-243). No pooling across time, no rolling, no full-sample fit.
- **The demean is within-cycle PIT** — `groupby("open_time")` (X131.py:159-160) uses only same-cycle
  trailing returns, all available at-or-before t.
- **Independently reproduced the pre-check IC: HL70 IC(rel→alpha_resid) = −0.0360, t −9.76, n 2458**
  — exactly the claimed value, on the 4h-entry grid using the strictly-PIT trailing return. The IC is
  REAL. At −0.036 it is far below the +0.10 look-ahead red-flag threshold; the leak signature (high
  positive IC + monetizing) is absent — this is the opposite (the re-rank LOSES gross).

→ **rel_ret_1d is strictly PIT. The −0.036 IC is genuine, not an overlap/leakage artifact.**

## Critical check 2 — Sign / logic correctness of the ensemble (is the REJECT a bug?): CORRECT

- **Sign verified empirically.** Reconstructing one side cycle: `score = z_xs(pred) − z_xs(rel)`
  (X131.py:243), long = top-K score (X131.py:220 `make_w` L=last-K of ascending sort), short =
  bottom-K. The LONG basket has mean rel_ret_1d = **−0.10** (recent relative LOSERS), the SHORT
  basket mean rel = **+0.07** (recent relative WINNERS). Since rel has a NEGATIVE IC (reversal:
  losers bounce), longing losers is the alpha-aligned direction. The `−z(rel)` is NOT accidentally
  doubling pred's own reversal nor flipped.
- **z_xs is per-cycle cross-sectional** — confirmed (check 1).
- **Base reproduces X117 exactly.** X131 base @4.5bps from the emitted parquet:
  **Sharpe +1.93 / maxDD −5674 / totPnL +10472** = X117 to the digit. The X131 held-book engine
  (X131.py:270-289) is line-for-line the X117 engine (X117_hl70_pnl_cost.py:78-87): same regime map
  (`b30>0.10`/`<−0.10`, WIN=180), same `make_w` beta-neutral logic, same
  `active=cyc_w[max(0,t−HOLD+1):t+1]`, `net+=wt/HOLD`, `turn=Σ|net−prev|`, `pnl=cyc−turn·0.5·cost`,
  same `np.isfinite` NaN guard. Only the side rank key changes (pred → ensemble). Bull/bear weights
  are appended identically to both arms (X131.py:211, 195-198).
- **Only the side regime differs.** Non-side cycles have max |base−ens| PnL = 0.028 (raw units),
  attributable solely to legitimate held-book HOLD-window bleed from neighboring re-ranked side
  cycles — not a logic error. ens @4.5bps = +0.86 / −9818 / +5301, matching the handoff.
- **Gross PnL collapse verified independently.** net(cost) is linear in cost; extrapolating
  {1,3,4.5}bps net PnL to cost=0 recovers **base gross +12272, ens gross +7241 (−41%)** — exactly
  the handoff's numbers. The gross-PnL fall is real and correctly computed (linearity of net-vs-cost
  also confirms the `turn·0.5·cost` model). The loss is NOT eaten by cost — the pre-cost return
  itself drops.
- RNG seeded (12345). G4 placebo shuffles ONLY rel (preserving pred) within each side cycle and
  rebuilds the same ensemble + beta-neutral weights (X131.py:420-423) — the correct "does the IC
  monetize vs rank-churn" control, 200 seeds.

→ **The ensemble sign/logic is correct. The REJECT is a real economic result, not an
implementation bug.**

## Why a strong orthogonal IC doesn't monetize (one-line read)

The production `pred` book is already a beta-neutral cross-sectional mean-reversion basket; the raw
`−z(rel_ret_1d)` ordering trades the same names from a noisier, higher-turnover angle that
mis-ranks the basket vs `pred` — so the orthogonal cross-sectional reversal IC (which exists in raw
return space) is already absorbed by `pred` at the held-book layer, and overlaying it destroys
pre-cost return rather than adding it. (Same IC-transports-but-PnL-doesn't failure as iter-018.)

## Gates touched by review
- **G1 look-ahead: PASS** (both critical checks clean, empirically verified).
- G3 correctly WAIVED (equal-weight 0.5/0.5 untuned, structural — no swept parameter).
- G2/G4/G5/G6/G7/G8 are performance gates → Evaluation. The emitted numbers are internally
  consistent with the parquet (base=X117, ens matches, gross collapse reproduced).

## No fixes required.
