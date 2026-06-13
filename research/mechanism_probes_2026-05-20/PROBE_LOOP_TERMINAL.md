# Mechanism Probe Loop — Terminal Synthesis (2026-05-20)

Six probes, run iteratively without pre-formed mechanism, to characterize whether the V3.1 production strategy contains a forecastable signal worth a portable strategy plan. Verdict: **no portable mechanism survives the joint VVV-de-confound × temporal-stability gate.**

## Probe arc

| # | Question | Result |
|---|---|---|
| 1 | Can trade-level features separate extreme-winner vs extreme-loser legs OOS-symbol? | best Cohen's d 0.353 (aggr_ratio_4h); null p95 = 0.377 → **NO signal exceeds noise floor** |
| 2 | Is the convex outcome symbol-systematic? | SYS_WINNER (VVV/LINK/AAVE/PENDLE/ASTER) vs SYS_LOSER (PENGU/SEI/ICP) — mean idio_skew +0.17 vs −0.13, dom_level_vs_bk +2.86 vs −0.67 (descriptive only) |
| 3 | Does trailing-30d idio_skew predict next-leg direction OOS-symbol? | full 0.504, primed 0.497, placebo 0.500 → **REFUTED** Probe #2's predictive read |
| 4 | Does a symbol's own past 7d signed PnL predict its next leg's sign? | OOS-symbol dir acc **0.526** vs placebo 0.487; magnitude corr 0.247 — **first to exceed lifecycle ceiling 0.515** |
| 5 | Robust across windows {3, 7, 14, 30}d × primed cohort? | primed dir 0.566/0.550/0.559/0.524 — all ≥0.52, magnitude corr 0.22-0.25 stable → **looked robust** |
| 6 | Per-group, per-symbol decomposition (which symbols drive the lift)? | g3 acc 0.55 built on **VVV n=161 (40% of 405-row primed set)**; non-VVV large-n: BIO 0.45, PENGU 0.54, WIF 0.50, PENDLE 0.29, HBAR 0.38; first_half 0.495 vs second_half 0.566 |
| 6b | Survives ex-VVV de-confound? | YES — ex_vvv 7d primed 0.560 (placebo 0.510, lift +0.05), all 3/7/14d windows ≥ 0.55; ex_top3 (VVV+BIO+PENGU) 7d 0.543 |
| 6c | Temporally stable ex-VVV? | **NO** — mean half1 lift **−0.001**, half2 lift **+0.109**; 2 of 3 windows negative in half1; 3d tercile1 −0.08, tercile2 −0.15 |

## Verdict

The Probe #4 signal **survives the VVV de-confound but fails the temporal-stability de-confound**. All apparent lift is concentrated in Sep 2025 – Apr 2026 — the same regime that produced V3.1's other PnL spikes (VVV/AXS/PENDLE rotation period). First half (Jul–Sep 2025) is null-to-negative over placebo across two of three windows.

Forward-expected lift, under the conservative assumption that the second-half regime is no more likely than the first to repeat, is **at most half** of in-sample (~+0.025 over placebo, equivalent to ~0.025 directional edge per leg). At V3.1 cost levels and the matched-placebo distribution widths from MEMORY.md (typical p95 around +1.0 Sharpe), this would not survive an honest matched-basket placebo gate.

**Do not write a strategy plan for PnL-mean-reversion.** Same pattern that closed K4 (cost-aware swap), W23 (orthogonal features retrain), the adaptive-horizon Sharpe spike, and the V3.3 decay-weighted sleeve: one-regime signal that doesn't generalize.

## What the probe loop closed

In addition to the 43+ prior-session directions:

44. trade-level feature separation of convex outcomes (closed by Probe #1; best Cohen's d ≤ null p95)
45. symbol-class signature from trailing skew (closed by Probe #3; OOS-symbol acc ≈ placebo)
46. PnL mean-reversion as a portable mechanism (closed by Probe #6c; half-of-sample signal)

## What remains genuinely open

The MEMORY.md "next research directions" list still stands and is unchanged by this loop:

- (a) **Annual retrain on fresh data with fixed target_A preprocessing** — drop universe-expansion target_A clip-at-±5 hack, properly winsorize, then redo K-sweep / N-sweep on fresh sample. This is the only path that could plausibly restore portability across universe composition.
- (b) **rvol_7d / ret_3d as model features (not gates)** — cohort attribution showed Sharpe spread q4−q0 = +15.77 for btc_rvol_7d; W23 tested two related features and failed, but the strongest signal (rvol_7d itself) is untested as a model feature.
- (c) **Move beyond free-data scope** — moderate-orthogonality signals (ethbtc_change_24h Sharpe spread +8.58, xs_ret_disp_1d +7.18) sit below the Glassnode-justification threshold of >11; on-chain or paid orderbook data may carry the needed orthogonal signal.
- (d) **Operational deployment** — paper bot V3.1 wiring, HL execution, cron, kill-switch. This is engineering, not research; it validates current edge forward and is required regardless of which research path is pursued.

Free-data Binance perp 4h-horizon residual research, including the 6-probe mechanism loop, is at terminal state.
