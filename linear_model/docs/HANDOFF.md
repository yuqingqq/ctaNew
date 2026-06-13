# Linear Model — HANDOFF

For a future session picking up this work. **Read STATUS.md first** for current state and result ledger.

## ⚠️ FINAL VERDICT (2026-05-14, post Steps 55-57): NOT PRODUCTION-VIABLE ON HYPERLIQUID

The entire +3.11 in-sample edge is **two un-executable meme coins**: SIREN (+14,061 bps) + JELLYJELLY (+10,542 bps) = +24,603; all other 108 symbols net **−630 bps**. Both are absent from Hyperliquid (JELLYJELLY force-delisted by HL Mar-2025; SIREN never listed). Evidence triangulated:
- HL inference-restriction (trade only the 70 HL symbols): +3.11 → **−0.17**, both placebos FAIL
- HL-native retrain (train+validate from scratch on 70 HL syms): **−1.29, gross negative** — no alpha in liquid universe even with a fair model
- Drop only SIREN+JELLYJELLY: collapse (Step 57, attribution shows other 108 net −630 bps)

The strategy is a 2-position meme bet in a 110-symbol costume. All prior positive validation (placebos p100, full-PIT, causal, cost-robust) correctly measured "machinery captures SIREN/JELLYJELLY better than random" — not a transportable edge. **Not a tuning artifact; the diversified executable α does not exist.** Methodology (PIT discipline, gate-consistent + wrapper-symmetric placebos, causal aggregator, cost analytics) is sound and reusable; the strategy is dead for HL execution. Details in STATUS.md "FINAL VERDICT" + Steps 55-57.

---

## In-sample validation TL;DR (Binance 110-panel — NOT executable as-is)

- **V2 + V3.1 sleeve = Sharpe +2.19 on 51-panel** (Step 34/35), placebo-validated (P1 p99, P2 p97).
- **V2 on 110-panel FULL-PIT + CAUSAL aggregator (Steps 47-51, FINAL)**: Sharpe **+3.11** (was +3.35 lagged), CI [+1.06, +4.92] strictly positive, 7/9 folds+ (fold 3 flipped −0.49 → +0.08; fold 1 still negative at −4.39). **P1 p100 (+1.45), P2 p100 (+1.61)** — beats all 100 random matched picks with wide edges. **K-drop universe stress (Step 51, causal)**: K=10 +1.26, K=20 +0.81, K=30 +0.97, K=40 +1.02 — survives positive sign at every K but far below baseline +3.11. Fold concentration is real: folds 4+5 contribute Sharpe +7.73 and +11.23 alone; fold 1 strongly negative. **Wrapper-symmetric placebo (Step 52) PASSES at p100, edge +1.37** — the cleanest test (placebo gets IDENTICAL select_refill/picks_hist/gate/PM/sleeve; only pred differs). Reviewer's biggest concern (picks_hist asymmetry) RESOLVED in model's favor: the machinery was worth ~+0.24 of apparent edge but model retains +1.37 genuine ranking edge. Remaining open: 51-panel not rebuilt with full-PIT for parity, 100-seed placebos coarse for tight tails, paired CI vs LGBM still needed. **Entry convention** = "decide at close of bar t, enter at close[t]"; switching to open_time t would invalidate. LGBM Phase UNI-111 (−1.48) remains contextual.
- **Linear raw 4h cycle (no sleeve) Sharpe = −2.55** with correct 9 bps cost. Sleeve adds +4.74 via cost amortization. Linear NEEDS the sleeve; LGBM doesn't (LGBM raw 4h = +1.98 already covers cost).
- **conv_gate contributes only +0.34** for B_IC_signed (trail_ic per-symbol wrapper does most filtering).
- **24h target hurts** (Step 37 +1.50 vs V2 +2.19, Δ −0.69); **horizon-aligned features hurt more on top of that** (Step 38 +0.50 vs Step 37 +1.50, Δ −1.00; total vs V2 51-panel Δ −1.69). V2 4h-target features sit in a local optimum.
- **Reviewer issues still open**: picks_hist asymmetry in placebos, 100-seed thin claim, per-fold coefficient diagnostics not saved, preprocessing-vs-training-filter distribution mismatch.
- **LGBM remains production reference** (Sharpe +0.74 baseline, +2.23 with V3.1 sleeve). V2 + sleeve is now **a research-validated candidate alternative**: 110-panel full-PIT + causal +3.11 with both placebos at p100. Still NOT production-grade — outstanding gates: full-wrapper-symmetric placebo (picks_hist asymmetry), cost sweep, paired CI vs LGBM on same panel, 51-panel full-PIT parity rebuild, 1000-seed placebo tightening. Architecture decisions (no basket, no sym_id, R3_BTC + V3.1 sleeve, full-PIT shifts) are validated; signal strength of the model itself remains near-zero IC and architecture-dominated.

## Key reviewer-confirmed findings (this session)

1. **σ-idio fallback leak in `01_build_target.py`** — fixed. Was using full-panel std for HYPE/ASTER fallback, leaking OOS info. Now uses cross-symbol fold-0 median.
2. **NaN bug in `rank_transform` / `per_symbol_rank`** — fixed in Step 34. `np.searchsorted(NaN)` returned `n_train`, mapping NaN to rank +0.5 (max). 39,902 NaNs in `funding_rate_z_7d` and 14,400 in `funding_rate_1d_change` were all silently mapped to +0.5. Fix: mask finite values before searchsorted; NaN positions → 0.
3. **Rank-feature scale mismatch** — fixed in Step 34. Rank features had std 0.23–0.30 vs z-score features at 1.0 and squared at 1.4–2.1. Re-z-score rank columns after ranking using fold-0 train stats so Ridge regularization is uniform.
4. **Step 32 not auditable** — fixed in Step 34. Predictions, per-cycle PnL, LOFO CSVs now saved to `results/step34_v1_fixed/` per variant.
5. **Lift sits in wrapper, not preds** — V1 raw IC ≈ 0, but B_IC_signed Sharpe ≈ +1.21. The "model" contributes near-zero to per-cycle ranking; sign flipping via trail_ic does the work.
6. **`btc_ret_fwd` was in candidate inventory** — stripped from `feature_inventory_audit.csv`. `hour_cos`/`hour_sin` flagged as calendar artifact.

## How to continue

### 1. Read these files first

```
docs/STATUS.md                                          current state + journey
results/step34_v1_fixed/summary.csv                     V0/V1/V2 results
results/step34_v1_fixed/v{1,2}_fixed_predictions.parquet preds for placebo re-runs
results/step35_verdict.csv                              P1/P2 placebo verdict
results/step35_placebos_v{1,2}_fixed.csv                full placebo distributions
data/feature_audit.csv                                  shapes of all base features
results/feature_inventory_audit.csv                     candidate features (cleaned)
```

### 2. Re-run any step independently

```bash
cd /home/yuqing/ctaNew
python3 linear_model/scripts/01_build_target.py            # ~6s — target with σ_idio
python3 linear_model/scripts/34_v1_nan_fixed.py            # ~15 min — V0/V1/V2 with fixes
python3 linear_model/scripts/35_placebo_v1_v2_fixed.py     # ~30-45 min — P1/P2 on V1+V2
```

Step 34 saves predictions to `results/step34_v1_fixed/`. Step 35 reads them — no retraining needed for placebo work.

### Conventions (read before reproducing)

**Entry timing**: "decision at close of bar t". `return_pct = close[t+48]/close[t] − 1`, entry price = close[t] = time t+5min. Full-PIT features use through close[t−1], which is conservative under this convention (could use close[t] but don't).

**Sleeve PnL alignment**: `aggregate_hold_through` MTMs `prev_weights × alpha[t]` which has a 1-cycle lag — prev_weights earn next cycle's alpha. Inflates Sharpe by ~+0.22. Causal recompute of Step 47 = +3.13 vs reported +3.35. All sleeve-based numbers in the codebase share this convention; relative comparisons valid, absolute Sharpes discount by ~+0.2.

**Cost**: 9 bps round-trip per full-turnover cycle for K=3 long + K=3 short at weight 1/K each. (Earlier 27 bps was a triple-count bug; fixed.)

### 3. PIT discipline (don't break)

| component | discipline |
|---|---|
| β estimation | `.shift(49)` (= HORIZON+1) for strict PIT |
| σ_idio | fold-0 train rows only; cross-symbol fold-0 median fallback for HYPE/ASTER |
| Feature winsorize bounds + z-stats | fold-0 train quantiles + mean/std |
| Rank transform | fold-0 train distribution; NaN-safe (Step 34) |
| Per-symbol rank | fold-0 train per-symbol; NaN-safe (Step 34) |
| Trailing IC | trailing 90d cycles, strict `<` current |
| Ridge α selection | RidgeCV gcv on fold-train + bootstrap (seeds 42, 1337, 7, 19, 2718) |

### 4. BTC-frame discipline

Target = α_β = `return_pct − β × btc_return` (BTC-hedged residual). Features must be BTC-frame or per-symbol-absolute. **DO NOT add basket-frame features** (`xs_alpha_*`, `*_vs_bk`, `bk_*`) — they contradict the BTC-hedging stance.

R3_BTC composition (V0/V1/V2 backbone, 20 features):
```
11 frame-neutral W17:
  return_1d, atr_pct, obv_z_1d, vwap_slope_96, bars_since_high_xs_rank,
  funding_rate, funding_rate_z_7d, funding_rate_1d_change,
  corr_to_btc_1d, idio_vol_to_btc_1h, beta_to_btc_change_5d

3 R3 squared U-shape:  return_1d², corr_to_btc_1d², beta_to_btc_change_5d²

4 BTC-frame replacements for basket features:
  dom_btc_z_1d, dom_btc_change_288b, corr_to_btc_change_3d, idio_vol_to_btc_1d

2 BTC squared U-shape:  dom_btc_change_288b², corr_to_btc_change_3d²
```

V2 adds 2 short-horizon features: `return_8h_orth` (orthogonalized vs return_1d) and `vol_zscore_4h_over_7d`.

## Preprocessing discipline (the load-bearing fix)

- **Heavy-tail features** (kurt > 50) → pooled rank transform; re-z-score after ranking
  - `vwap_slope_96`, `idio_vol_to_btc_1h`, `idio_max_abs_12b`, `funding_rate`, `funding_rate_z_7d`, `funding_rate_1d_change`
- **Per-symbol biases** (funding features) → per-symbol rank pooled fold-0; re-z-score
- **Squared U-shape terms** → use standard z-scored base (rank-of-rank doesn't help)
- **Standard otherwise**: winsorize p1/p99 + z-score using fold-0 train stats
- **NaN handling**: explicit NaN → 0 (median rank); never let `np.searchsorted` see NaN

## Open paths if pursuing further

### Next research directions (V2 is promising; validate further)

1. **P1/P2 placebos on 110-universe** (~30 min) — match-universe random picks; confirm Step 41 rerun's +2.03 beats p95 placebo on expanded universe. If yes, V2 is universe-robust + signal-validated, not just an artifact.
2. **Universe-stress within 110** — drop K random from 110-panel (post-BTC-exclusion), 30 draws/K (analogous to Step 40 on 51). Measure degradation pattern. If similar to 51-panel's gentle slope, V2 has truly transferable signal.
3. **Steps 37/38 already eliminated**: 24h target + sleeve = +1.50 (worse than 4h target). 24h-aligned features hurt more (−1.00 vs Step 37). The V2 4h-tilted feature set + 4h target is a local optimum within the sleeve architecture.

### Reviewer's outstanding findings (still relevant for production-grade claims)

4. **Full-wrapper placebo** (fix picks_hist + refill asymmetry in `phase_ah_sleeve.py:216`): random pick at candidate stage, then full refill/PM/picks_hist machinery downstream. Step 35's P1/P2 placebos didn't address this — V2's +0.58 P2 edge over p95 could be partly the refill machinery (real V2 gets picks_hist feedback, placebos don't).
5. **1000-seed placebos** for tight percentile claims (Step 35 used 100; V2 at p97 has ~3% noise).
6. **Save per-fold Ridge α + coefficients + feature contribution** for linear-model interpretability.
7. **Document preprocessing-vs-training filter mismatch**: preprocessing uses fold-0 calendar rows; training filters autocorr_pctile_7d ≥ 0.5. Not leakage but distributions differ.

## What NOT to do

- **Don't add basket-frame features** to R3_BTC variants — target is BTC-hedged, basket contradicts.
- **Don't trust V1 +1.31 (Step 32 pre-fix)** — that number was NaN-bug-inflated by ~+0.10 and is methodologically stale.
- **Don't trust pre-σ-fix R3_BTC +1.92** — σ leak inflated that result; the σ fix dropped it to +0.86 (which itself fails P1/P2).
- **Don't add `btc_ret_fwd`** — literal forward return, was in inventory before cleanup.
- **Don't trust `hour_cos`/`hour_sin`** — calendar artifacts; need out-of-time block validation before use as alpha features.
- **Don't compare placebo distributions across scripts without checking gate-consistency** — Step 33 vs Step 35 P1 p95 differ by 0.34 because Step 33 gated placebos on `pred_z` while real used `pred_B` (inconsistent); Step 35 fixes this and gives wider, harder placebo.
- **Don't change `.shift(49)` to `.shift(1)`** — only worth -0.06 Sharpe and 47-bar forward leak in β.
- **Don't add sym_id one-hot** — Step 9 showed sym dummies absorb 56× more coef mass than numerics without helping.

## Honest expectation (post Step 41 partially PIT-audited rerun)

Linear model + V3.1 sleeve evidence as of Step 45 (strict-PIT placebos complete):
- V2 51-panel: +2.19, placebo-validated (P1 p99 +1.04, P2 p97 +0.58 — both pass)
- V2 110-panel STRICT-PIT (Step 44+45): **+2.11, CI [+0.14, +3.77] strictly positive, P1 p97 (+0.96) PASS, P2 p99 (+0.66) PASS** — first 110-panel result that fully placebo-validates end-to-end
- V2 110-panel non-strict (Step 41+42, DEPRECATED): +2.03, P1 pass / **P2 fail** at p92 (−0.13), CI crossed zero, K=10 univ stress mean −1.24 catastrophic
- Strict-PIT fix (beta `.shift(49)`, all rolling `.shift(1)`) BOTH lifted Sharpe (+0.08) AND transformed P2 from FAIL to PASS at p99. The leak was confounding the within-universe placebo, not just inflating Sharpe.
- Pre-fix Step 41 (+1.44) deprecated — had unshifted return_8h / vol_zscore_4h_over_7d and included BTC

Caveats that still apply on Step 44+45 strict-PIT:
- CI [+0.14, +3.77] is strictly positive but wide
- Fold 5 drives (LOFO Δ −0.45) — less concentrated than non-strict but still single-fold dependency
- Step 46 universe stress on strict-PIT: K=10 mean −1.38 (slightly WORSE than non-strict −1.24), K=20/30/40 modestly improved. **K=10 catastrophe persists**. The +2.11 is composition-fragile; depends on specific high-IC symbols in the present 110-symbol set. Real-world delistings/composition drift would materially affect performance.
- LGBM Phase UNI-111 (−1.48) is **contextual only**, not apples-to-apples (different dates, cost convention, target winsorization vs LGBM's target_A clip-at-±5)
- 51-panel V2 features have NOT been rebuilt with strict-PIT. The 51-panel +2.19 + P1/P2 pass used the same convention as 110-panel non-strict. If 51-panel is rebuilt with strict-PIT it may also shift; need consistent comparison.

**The model itself contributes little**: overall IC ≈ 0 in both panels. The lift comes from architecture (sleeve + universe + wrapper), not raw prediction quality. V2 alone at raw 4h cycle = −2.55 Sharpe (with CORRECT 9 bps cost, not the 27 bps figure that appeared in earlier drafts).

Mechanism summary:
- Linear model gross signal per 4h cycle: ~0 bps (V0 −1.5, V1 +2.6, V2 −0.9)
- 4h-cycle cost: **9 bps** at full turnover (corrected from earlier mis-stated 27 bps — each K=3 leg at weight 1/K bears 4.5/K bps; total 2 × COST_PER_LEG = 9 bps round-trip)
- Net per cycle: net-negative for linear (gross < cost)
- Sleeve overlay drops turnover ~6× and extends effective horizon to 24h, which is where linear can find slow alpha (funding regime, dominance, vol regime)

What the linear journey produced:
- **First-principles design**: clean β-residual target, frozen σ_idio with cross-symbol fold-0 median fallback, PIT-controlled core preprocessing (β shift(49), σ_idio fold-0 only, winsorize/z-score from fold-0 stats, return_8h .shift(1), vol_zscore .shift(49))
- **Five reviewer-confirmed bug fixes**: σ leak, NaN rank bug, scale mismatch, btc_ret_fwd in inventory, hour_cos calendar artifact
- **Partially PIT-audited preprocessing recipe** for any future linear/tree model: rank for heavy-tail (kurt > 50), per-sym rank for funding, re-z-score to unit variance, NaN-safe searchsorted, no basket-frame features when target is BTC-hedged. **Caveat**: other kline-derived features (`corr_to_btc_*`, `idio_vol_to_btc_*`, `dom_btc_*`, `obv_z_1d`) inherit the 51-panel rolling-value convention but are NOT separately PIT-audited under strict "position opens at open_time" interpretation.
- **Methodological discipline** for placebos: gate-consistency, full-wrapper symmetry, sufficient seed count

Linear can still be useful as:
- Interpretability baseline (Ridge coefs are readable; LGBM splits are not)
- Audit reference (PIT-controlled core preprocessing + simple model = easier to verify there's no leak in the core path)
- Slower-horizon strategy if someone retests at 24h (where linear's slow-signal capture might exceed cost)

## Memory references

- `~/.claude/projects/-home-yuqing-ctaNew/memory/project_vBTC_linear_model.md` — earlier Ridge investigation
- `~/.claude/projects/-home-yuqing-ctaNew/memory/project_vBTC_status.md` — full vBTC research history (LGBM track)
- `~/.claude/projects/-home-yuqing-ctaNew/memory/project_vBTC_ic_selector_root_cause.md` — why IC selector is noise-dominated
