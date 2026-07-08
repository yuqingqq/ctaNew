# Convexity v1 vs v3 — detailed flow + controlled-comparison design

> This doc is the **v1↔v3 diff + controlled comparison**. For the full canonical end-to-end v3 pipeline
> (train → predict → eligibility → regime → select → size → sleeves → accounting), see `CONVEXITY_V3_FLOW.md`.


Both strategies share the **same model** (per-symbol RidgeCV, two books: base `V0_LEAN` + long `V0_LEAN+resid_rev`,
recency-60, walk-forward). They differ **only in strategy construction + execution assumptions**. To compare them
fairly, every differing variant must be controlled — a naive "toggle truncation on v3" is NOT fair.

## v1 detailed flow (`run_convexity_v1_live.sh`, "single low-vol book + resid_rev")
1. Model: `convexity_v1_{short,long}` = V0_LEAN base + residrev long (train_twobook_models, fit_cut = panel−1d).
2. **Universe = maturity_meta (onboardDate) minus TRUNCATED top-80 high-vol** (`convexity_v1_universe.json:exclude_high_vol`,
   trailing-30d rvol_7d, frozen monthly at fit_cut). This is v1's volatility control.
3. Each 4h cycle: rank eligible names by pred → **K=3 longs (top pred) + K=3 shorts (bottom pred)**.
4. **Sizing = equal (1/K)**. **Side = beta-neutral (`SIDE_BETA_NEUT=1`)**. Hold = 6 overlapping sleeves (24h).
5. **NO gates**: no regime gate, no bull/bear overlays, no `SHORT_MIN_RET3D`, no `CONC_CAP`, no depth-cost.
6. Cost = **4.5 bps/leg flat**. Funding carry **NOT charged** (`CHARGE_FUNDING=0`).

## v3 detailed flow (`run_convexity_v3_regime_gate.sh`, "regime-gate stack")
1. Same model class (hl_lean175 base + hl_residrev_lean long).
2. Universe = panel (backtest) / maturity (live) — **NO vol-truncation**; volatility handled by *sizing*.
3. Each 4h cycle: rank → **K_SHORT=2 shorts + K_LONG=1 long** (asymmetric).
4. **Sizing = inv_sqrt_vol** (weight ∝ 1/√vol — down-weight high-vol, don't exclude). **Side NOT beta-neutral (`SIDE_BETA_NEUT=0`)**. Hold = 6 sleeves (bull hold=1).
5. **Gates ON**: `REGIME_GATE=1` (perf de-gross), `BULL_DEEP_THR=0.15` (hot-bull sit-out), `BEAR_DEPTH_RAMP=1`
   (bear depth sizing), `SHORT_MIN_RET3D=-0.20` (veto recent crashers), `CONC_CAP=0.40`+single-exempt, `STOP_SKIP_REGIMES=bear`,
   `BULL_MODE=sidealpha` + 25% BTC-long hedge, `BULL_SHORT_RANK=return_1d`.
6. Cost = **9 bps/leg + per-symbol depth-cost model**. Funding carry **charged** (`CHARGE_FUNDING=1`).

## THE VARIANTS (all differences to control)
| # | variant | v1 | v3 |
|---|---|---|---|
| 1 | K (long/short) | 3 / 3 | 1 / 2 |
| 2 | sizing | equal | inv_sqrt_vol |
| 3 | beta-neutral side | YES (=1) | NO (=0) |
| 4 | **volatility control** | **truncate top-80** | **inv_sqrt_vol sizing** |
| 5 | REGIME_GATE | off | on |
| 6 | BULL handling | none | sidealpha + BTC-hedge + DEEP_THR |
| 7 | BEAR handling | none | depth-ramp + equal + K |
| 8 | SHORT_MIN_RET3D | off | −0.20 |
| 9 | CONC_CAP | off | 0.40 + exempt |
| 10 | cost | 4.5 flat | 9 + depth-cost |
| 11 | funding charged | no | yes |
| 12 | **model feature set** | full `V0` (funding features IN), preds `hl/` | `V0_LEAN` (funding features OUT), preds `hl_lean175/` — corr 0.88, NOT identical |
| 13 | equity-DD stop in bear | on (default) | `STOP_SKIP_REGIMES=bear` (off in bear) |

Note: the "same model" phrasing is only true at the *class* level (per-symbol RidgeCV, two-book). Native v1 and v3
use different **fitted** models (different feature set, #12). The controlled comparison below neutralizes this by
using `hl_lean175` preds for ALL cells — so it isolates strategy construction, but its "v1_native" cell runs on
v3's lean preds, not v1's true funding-feature preds.

## Fair-comparison rule
- **Control execution** (cost, funding, model, universe, period) identically on both sides.
- **Vary strategy** in a ladder (one group at a time) from v1-native → v3-native, so each variant's contribution is
  attributable — especially: truncation must be paired with v1's K=3 + equal + beta-neut (NOT v3's K=1/2 + inv_sqrt_vol).

## Controlled results (matched execution: cost 9, funding on, same model/universe/period)
| config | IS Sharpe | OOS Sharpe | OOS-2023 | OOS-2024 | OOS maxDD |
|---|---|---|---|---|---|
| v3-native (K1/2, inv_sqrt_vol, gates, no-trunc) | +3.11 | −0.70 | +2,897 | −6,837 | −10,374 |
| v1-config NO truncation (K3, equal, β-neut, no gates) | +1.15 | −1.30 | −2,201 | −3,856 | −10,668 |
| v1-native (K3, equal, β-neut, TRUNCATION, no gates) | +0.38 | −0.85 | −2,201 | **−892** | **−6,335** |

Findings: (1) v3 dominates in-sample even matched-cost (3.11 vs 0.38) — construction, not cost. (2) OOS v3≈v1-native
but they protect DIFFERENT regimes: v1 truncation → 2024 (−892 vs v3 −6837) + lower maxDD; v3 gates → 2023 (+2897 vs
v1 −2201). (3) Truncation on v1's config (K3) cuts 2024 −3856→−892. => v3 gates + v1 truncation are COMPLEMENTARY;
combining (gated-v3 + truncation) was the best OOS (+0.07). Truncation belongs as a v3 ADD-ON (dispersion-gated),
tested at v3's K, not a replacement. Harness: live/controlled_v1v3.sh, build_volexclude_allow.py.
