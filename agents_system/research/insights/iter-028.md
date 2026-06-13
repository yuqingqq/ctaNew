# iter-028 — Phase-2 TARGET × HORIZON predictability heatmap — NO-CANDIDATE

The one untested Phase-2 axis (flagged by iter-027): is the cross-sectional signal
MORE predictable + transport-stable at a DIFFERENT prediction target/horizon than the
current 4h BTC-beta-residual? The 4h residual is intrinsically hard/near-noise
(iter-027: |IC|~0.005, pred sign-flips across universes). A different target/horizon
might be materially more predictable — a strategy-identity rebuild. Evaluated
TRANSPORT-FIRST (HL70 2024-12→2026-05 AND EXT 23-sym 2021-26).

## Light online rationale (target-choice)
- Short-horizon **reversal** is the dominant cross-sectional crypto effect (illiquidity-
  driven daily reversal; Dobrynskaya SSRN 3913263; FRL S1544612320303135) — known here
  as iter-022's `rel_ret_1d`. **Momentum** appears at multi-week-to-month horizons but is
  regime/era-conditional in crypto (1-sided bull vs full cycle). **Vol-scaling** the target
  is the standard SNR fix (Barroso-Santa-Clara). So the grid spans reversal (short) →
  momentum (long) predictors against raw / market-residual / vol-scaled targets at
  4h→1w horizons, plus the native 4h beta-residual anchor.

## STEP-2 — the decisive grid (mean per-cycle cross-sectional Spearman IC)
Script `iter028_target_horizon_heatmap.py`. Targets: betares (native 4h beta-resid, anchor),
raw (chained 4h fwd ret), mktres (raw − XS-mean = alt-index residual), vs (raw / trailing
vol). Horizons built by chaining non-overlapping 4h blocks. Predictors: rev_short (−trail1d),
rev_4h (−trail4h), mom_long (trail7d), mom_30d, funding_z, pred_proxy (per-sym z of −trail1d).

### Anchor — current target (betares, 4h)
| predictor | HL70 IC | EXT IC | transport |
|---|---|---|---|
| rev_short | +0.0365 | +0.0308 | same |
| rev_4h | +0.0313 | +0.0327 | **same (min\|IC\| 0.0313)** |
| mom_long | −0.0160 | −0.0227 | same |
| mom_30d | −0.0105 | −0.0175 | same |
| funding_z | +0.0005 | −0.0122 | FLIP |
| pred_proxy | +0.0326 | +0.0227 | same |

### Best TRANSPORT-STABLE predictor per cell (min|IC| = min(|HL70|,|EXT|) among sign-consistent)
| target | hor | best pred | HL70 IC | EXT IC | **min\|IC\|** |
|---|---|---|---|---|---|
| **betares (current)** | **4h** | rev_4h | +0.0313 | +0.0327 | **0.0313** |
| raw / mktres | 4h | rev_short | +0.0351 | +0.0323 | 0.0323 |
| raw / mktres | 12h | rev_short | +0.0406 | +0.0291 | 0.0291 |
| raw / mktres | 1d | rev_short | +0.0375 | +0.0275 | 0.0275 |
| raw / mktres | 3d | rev_short | +0.0118 | +0.0227 | 0.0118 |
| raw / mktres | 1w | funding_z | −0.0063 | −0.0049 | 0.0049 |
| vs | 4h | rev_4h | +0.0346 | +0.0314 | 0.0314 |
| vs | 12h | rev_short | +0.0394 | +0.0295 | 0.0295 |
| vs | 1d | rev_short | +0.0385 | +0.0281 | 0.0281 |
| vs | 3d | pred_proxy | +0.0186 | +0.0188 | 0.0186 |
| vs | 1w | mom_30d | +0.0522 | +0.0074 | 0.0074 |

### Transport-stable predictability is MONOTONICALLY DECREASING with horizon
Peak transport-stable min|IC| by horizon (best target/pred at each): **4h 0.032 → 12h 0.030
→ 1d 0.028 → 3d 0.019 → 1w 0.007.** The 4h cell is the PEAK; no cell beats it.

## Three decisive readings
1. **No (target, horizon) cell beats the current 4h beta-residual on transport-stable
   predictability.** The 4h-residual anchor (min|IC| 0.031) is at/above every other cell.
   raw/mktres/vs at 4h are the same magnitude (~0.032); 12h–1d are slightly LOWER (0.028–0.030);
   3d–1w COLLAPSE (≤0.019, many sign-flip). Lengthening or shortening the horizon does not
   raise the cross-sectional SNR — it lowers it.
2. **The big-IC cells are HL70-only and SIGN-FLIP on EXT (universe-overfit wall #2).**
   `vs/1w mom_long` HL70 **+0.0715** but EXT +0.0041; `vs/1w mom_30d` HL70 +0.0522 / EXT
   +0.0074; `raw/1w mom_long` HL70 +0.0306 / EXT −0.0093 (FLIP); `vs/3d mom_long` +0.0569 /
   +0.0006. Momentum "works" in the HL70 2025-26 bull and dies/flips full-cycle on EXT —
   exactly iter-015 (mom_180d) and iter-021 (funding). Not tradeable forward.
3. **The ONLY transport-stable predictor at EVERY horizon is short-horizon REVERSAL**
   (rev_short / rev_4h / pred_proxy, which are the same effect). This is iter-022's
   `rel_ret_1d` — already REJECTED at the construction/PnL layer. Note `raw` and `mktres`
   columns are byte-identical (Spearman rank-IC is invariant to subtracting a per-cycle
   constant) → market-residualizing the target adds nothing a rank predictor can exploit.

## STEP-3 — tradeability of the best longer cell (the cost-profile angle)
The orchestrator asked specifically: is a longer horizon TRADEABLE with a better cost
profile (lower turnover)? The best longer cell is 12h reversal (min|IC| 0.0295, near 4h).
Script `iter028_h12_tradeability.py`: rank-K=5 reversal long-short, non-overlapping entries
at horizon spacing, GROSS per-cycle spread + turnover + matched-random-timing G4 placebo.

| panel | hor | gross bps | Sharpe | turnover | G4 rank |
|---|---|---|---|---|---|
| HL70 | 4h | +0.87 | +0.17 | 0.38 | p66 |
| HL70 | 12h | −0.13 | −0.01 | 0.63 | p45 |
| HL70 | 1d | +11.73 | +0.39 | 0.84 | p84 |
| **EXT** | **4h** | **−4.66** | −1.37 | 0.33 | **p0** |
| **EXT** | **12h** | **−12.73** | −1.23 | 0.54 | **p0** |
| **EXT** | **1d** | **−17.52** | −0.84 | 0.76 | **p0** |

**Tradeability FAILS on all three pre-checks:**
- **Transport**: EXT gross spread is NEGATIVE at every horizon and gets WORSE as horizon
  lengthens (−4.7 → −12.7 → −17.5 bps). HL70 positivity does not transport.
- **G4 placebo**: HL70 p45–84 (< p95), EXT **p0** (random reversal-timing BEATS the real
  ranker). The heatmap IC is rank-info the simple top/bottom-K construction can't monetize.
- **Cost profile is the wrong way**: turnover RISES with horizon while gross goes negative,
  so the "longer horizon = lower cost" lever is moot — there is no positive gross to protect.

This is the iter-022 collapse one layer earlier: the reversal IC is real but the held-book
construction extracts no positive GROSS PnL from it on the transport panel, at any horizon.

## VERDICT — NO-CANDIDATE (strong negative finding)
No (target, horizon) cell is materially more predictable AND transport-stable AND tradeable
than the current 4h BTC-beta-residual. Concretely:
- The **4h horizon is the cross-sectional SNR peak**; predictability decreases monotonically
  as you lengthen to 12h/1d/3d/1w (and the short 1h cell is sub-grid/unavailable cleanly).
- The only **transport-stable** signal at any horizon is short-horizon **reversal** = iter-022's
  already-rejected `rel_ret_1d` (dies pre-cost / at construction, p0 on EXT here too).
- Every cell that **looks** materially better (the >0.05 IC momentum/vol-scaled long-horizon
  cells) is **HL70-only and sign-flips on EXT** — the universe-overfit wall (#1 Phase-2 killer).

**The 4h-beta-residual was already the best tractable target/horizon.** Changing the target
type (raw / alt-index-residual / vol-scaled) or the horizon (1h…1w) does not unlock more
transport-stable, tradeable cross-sectional predictability on free Binance/HL data. This closes
the last untested Phase-2 axis (different target/horizon) — joining the feature/model axis
(iter-027), construction overlays (iter-019/020), secondary-signal tilts (iter-021/022/023/025),
and reactive risk (iter-024). The codified walls now extend to the TARGET/HORIZON layer.

## What would still move the needle (out of free-data scope)
1. **PAID orthogonal LEADING data** (Coinglass liquidation flow / Glassnode on-chain) — the
   only mechanism for the DD and the only untested data axis; needs a human key.
2. **Genuinely non-cross-sectional structure** (per-symbol TS + alt-index hedge) — a fresh
   project; prior per-sym-timing (iter-004) did not monetize.

## Scripts
- `research/convexity_portable_2026-05-20/scripts/iter028_target_horizon_heatmap.py`
  (→ `results/iter028_th_grid.csv`)
- `research/convexity_portable_2026-05-20/scripts/iter028_h12_tradeability.py`

## Champion unchanged
BASELINE HL70 regime-hybrid held-book (Calmar +1.68) + iter-012 vol-norm reactive stop (k=2.0).
