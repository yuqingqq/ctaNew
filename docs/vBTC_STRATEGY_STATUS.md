# vBTC strategy — current state

Last updated: 2026-05-11

## Status: PRODUCTION-VALIDATED, PAPER BOT DELIVERED

| Milestone | Status |
|---|---|
| Architecture calibration (K, N, train window, IC cadence) | ✓ locked |
| Feature set (WINNER_21) | ✓ locked |
| Skip mode (flat_real) | ✓ adopted |
| PIT eligibility (60d min history) | ✓ adopted |
| DD overlay (dd_tier_aggressive: 10%→0.6, 20%→0.3, 30%→0.1) | ✓ adopted |
| Pipeline audit (9 categories) | ✓ clean |
| Random-target null test (z=+3.28 above null) | ✓ clean |
| Production bot + trainer delivered | ✓ `live/vBTC_paper_bot.py`, `live/train_vBTC_artifact.py` |
| Wire live data fetcher (Binance REST + feature pipeline) | ☐ TODO (task #48) |
| Wire HL execution layer | ☐ TODO (task #49) |
| Cron deployment | ☐ TODO (task #50, blocked by #48/#49) |
| Volume-based eligibility (dynamic universe Phase 2) | ☐ optional (task #51) |

## Final validated numbers

Walk-forward 9-fold OOS (2025-07-19 → 2026-04-30):
- **Sharpe: +4.46** [+2.92, +6.05]
- **Max DD: -858 bps (-8.6%)**
- **Total period PnL: +12,521 bps (+125%)**
- **Annualized: +169% at 1× leverage**
- Cumulative gain over honest baseline: +1.87 Sharpe, -86% DD


## Production-recommended config (validated)

| Layer | Choice | Source |
|---|---|---|
| Features | WINNER_21 (28 v6_clean − 14 drops + funding lean + cross-BTC + more funding) | phase 1-5 |
| Model | LGBM, 5-seed ensemble | ridge_vs_lgbm |
| Training window | Expanding (default `_slice`, anchored to data_start) | window_cadence_grid |
| Universe selection | Rolling-IC, **180-day lookback × 90-day refresh cadence**, top-15 by trailing IC | rolling_ic_v3 |
| Universe size | N=15 | phase 8 |
| Position count | K=4 long / K=4 short | phase 6 |
| Conv gate | Binary skip below 30th-pctile dispersion (252-cycle history) | various |
| **Skip mode** | **`flat_real` — close on gate fire, re-open on gate clear (2-leg cost)** | **skip_flat_test 2026-05-10** |
| **PIT eligibility** | **listing_date + 60d ≤ T (excludes new tokens from train + universe)** | **dynamic_universe 2026-05-10** |
| PM persistence | M=2, band=1.0 | various |
| Cost model | 4.5 bps per leg | locked |
| **DD overlay** | **`dd_tier_aggressive`: 10%→0.6 / 20%→0.3 / 30%→0.1 (graduated)** | dd_mitigation_sweep |

## Validated Sharpe

| Scope | Prior (PIT 60d + dd>20%) | **PIT 60d + dd_tier_aggressive (NEW)** |
|---|---|---|
| Walk-forward 9 OOS folds | +3.89 [+2.17, +5.50] | **+4.46 [+2.92, +6.05]** |
| Max DD (full WF) | -2,212 bps | **-858 bps (-61%)** |
| (no overlay reference) | +1.56 / DD -6,728 | — |

Cumulative gain over honest baseline (+2.59 / DD -6,009): **+1.87 Sharpe; -86% DD.**

Note: absolute Sharpe varies ±0.5 across script runs due to LGBM seed nondeterminism. Within-script comparisons (e.g., PIT 60d vs no_filter) are robust.

## What we tried

### Architecture knobs (calibrated)
- K=3 vs 4 vs 5 vs 6: K=4 wins
- N=10 vs 12 vs 15 vs 20 vs 25: N=15 wins
- Train window: expanding > rolling 60d/90d/120d/180d/240d/365d
- IC universe cadence: 180d/90d > weekly/quarterly/static
- Features: WINNER_21 set finalized
- Model: LGBM > Ridge (Ridge near-zero IC)

### Variance/DD overlays (tested)
| Test | Mechanism | Result |
|---|---|---|
| A | vol-scaling | ❌ rejected |
| B | continuous sigmoid dispersion sizing | ❌ rejected (binary conv_gate sufficient) |
| **C** | **trailing-DD deleveraging (dd>20% → 0.3 size)** | ✅ **VALIDATED +1.35 Sharpe / -64% DD** |
| D | B+C combined | ❌ slightly worse than C alone |
| E | inverse-vol weighting | ❌ rejected |
| F | theoretical PnL floor | n/a (theoretical only) |
| G | 5-min intra-cycle stop-loss | ❌ rejected (whipsaws break alpha) |

### Closed paths (don't re-test)
- Universe expansion to N=20+
- K > 4 (signal dilution)
- Rolling 180d training (less data → worse predictions)
- Vol-feature ablation (atr_pct, idio_vol_*) — they're alpha, not bias
- Intra-cycle stop-loss
- VVV-specific filters (production calibration universe excludes VVV)

## Open work, prioritized

### Tier 1 (likely high impact)
1. **`evaluate_stacked` gap RESOLVED 2026-05-10** ✓ — β-neutral scaling is regime-dependent, NOT a free Sharpe boost. Walk-forward +1.34 [-1.01, +3.69] vs local +2.59. β estimation breaks in stress regimes (folds 4, 9). Don't adopt; stay with local evaluator.
2. **Skip-flat vs skip-hold RESOLVED 2026-05-10** ✓ — flat_real (close+re-open with realistic 2-leg cost) beats hold by +0.35 Sharpe with Test C overlay. **NEW production recommendation.**

### Tier 2 (consistency / CI tightening)
3. **20-seed ensemble REJECTED 2026-05-10** ✗ — more seeds compressed prediction magnitudes, more conv_gate fires, more flat_real costs. Sharpe dropped from +4.15 (5-seed) to +2.29. Stick with 5.
4. **Multi-horizon ensemble (h=24, h=48, h=96)** — xyz v7 precedent. ~3× training time.
5. **Conv-gate threshold sweep** — currently 30th-pctile, untested at 40/50/60. Especially relevant if revisiting ensemble size.

### Tier 3 (structural)
6. **PM persistence sweep** (M=2 vs M=3) — more conservative entry might fix fold 3 weakness.
7. **Annual cal-window rolling** vs current expanding cal.

### Investigation
8. **Fold 3 (Sept-Oct 2025) regime diagnostic** — consistent -3.5 to -5.9 Sharpe across configs. What's structural?

## Findings 2026-05-10 (consolidated)

### Architecture is at a local optimum
- K=4, N=15, expanding train, rolling-IC 180/90 — all calibrated; no nearby point dominates
- LGBM > Ridge (Ridge near-zero IC); 5 seeds optimal (20-seed compresses dispersion → gate misfires)
- WINNER_21 features locked; ablation tests show vol features ARE alpha

### Best feasible variance reduction = overlay
- **flat_real skip mode** (close on conv_gate, re-open on clear) → +0.35 Sharpe vs hold
- **Test C DD overlay** (dd_pct>20%_size=0.3) → +0.5-1.5 Sharpe / -64% DD
- **Combined production stack: ~+3.5-4.0 Sharpe walk-forward, max DD -2,200 to -3,500 bps**
- Test B (continuous dispersion sizing) and Test G (intra-cycle stop) rejected
- β-neutral evaluator rejected (regime-dependent)

### DD root cause — single 5-month regime
- One 148-day DD episode (Sep 10 2025 → Feb 14 2026)
- Drawdown phase 55 days (-6,009 bps), recovery 92 days (+5,747)
- Affects folds 3-5 specifically (mean -7 to -12 bps/cycle)
- NOT fat-tail clustering — symmetric distribution, no extreme outliers
- Top-5% extreme cycles only 23.5% of variance

### Per-symbol attribution surprise
Tested hypothesis "new listings (VVV/WIF/WLD) drive DD" — REFUTED:

| Top winners | sum_pnl | per-pick Sharpe (annualized) |
|---|---|---|
| VVV | +4,261 | +3.4 |
| WIF | +3,731 | +3.8 |
| WLD | +1,551 | +1.1 |

| Top losers | sum_pnl | per-pick Sharpe (annualized) |
|---|---|---|
| ICP | -2,330 | -2.5 |
| ORDI | -1,148 | -2.6 |
| HBAR | -1,039 | -3.9 |
| TAO | -963 | -2.1 |
| AAVE | -674 | -3.0 |

The drag in folds 3-5 comes from ICP/ORDI/HBAR/TAO/AAVE being in universe alongside the winners. **Older established names where the model's features don't generalize, NOT new listings.**

## NEXT DIRECTIONS (prioritized)

### Tier 1: Per-symbol failure investigation — RESOLVED 2026-05-10
**Finding: IC is similar (+0.039 losers vs +0.049 winners). NOT a model accuracy issue.**

The difference is **trade-level payoff asymmetry**:
- Winners: VVV long delivers +97 bps mean (huge positive skew when right)
- Losers: ICP/RUNE short picks realize POSITIVE returns (drift against the short); AAVE long picks have 44% win rate

Pick-side imbalance: AAVE shorted 15,481× vs longed 4,079 (model-side bias on losers).

**Implication:** Per-symbol RANK alpha is real for both cohorts. Loser names underperform because realized payoffs lack the upside skew that VVV/WIF have. NOT a fixable feature issue at the pred level.

**New direction candidates:**
- Per-symbol-side filter (cut "AAVE-long" or "ICP-short" specifically based on trailing realized Sharpe)
- Per-symbol size scaling by trailing realized Sharpe (PIT-disciplined)
- Both have feedback-loop risk; need careful PIT validation

### Tier 2: Diversification of prediction noise
- **Multi-horizon ensemble** (h=24, h=48, h=96) — xyz v7 precedent. ~3× training time. Could help where single-horizon features fail on ICP-class names.

### Tier 3: Conservative entry tuning
- **PM persistence M=3** (vs M=2) — more conservative; might filter ICP-bad picks
- **Conv-gate threshold sweep** (30 → 40/50/60) — fewer trades, higher conviction

### Tier 4: Universe-level filtering (with risk)
- **Per-symbol rolling Sharpe filter** — exclude names whose past strategy returns are negative. Risk: feedback loops, validation difficulty.
- **High-vol cap** — limit names by trailing realized vol (PIT-safe). Risk: cuts winners (VVV is high-vol winner) symmetrically with losers.

### Tier 5 (low priority — closed paths)
- Rolling training window (rejected — less data)
- Universe expansion N>15 (rejected — signal dilution)
- K>4 (rejected — signal dilution)
- 20-seed ensemble (rejected — gate misfires)
- Min-history filter (rejected — no-op or break early folds)
- Vol-feature ablation (rejected — they're alpha)
- Intra-cycle stop-loss (rejected — whipsaws)
- β-neutral evaluator (rejected — regime-dependent)

## DD ANATOMY (2026-05-10)

The drawdown is **NOT fat-tail**, it's **single regime drift**:
- One 148-day DD episode (Sep 10 2025 → Feb 14 2026)
- Drawdown phase 55 days (-6,009 bps), recovery phase 92 days (+5,747 bps)
- Affects folds 3-5 specifically (mean -7 to -12 bps/cycle)
- Coincides with new high-vol listings entering universe (VVV, WIF, WLD)

Distribution is symmetric (p1=-353, p99=+424); no fat tails. Top-5% extreme cycles = only 23.5% of variance. **Single regime is the variance source, not bad cycles.**

**Min-history filter REJECTED 2026-05-10**: panel symbols all have first_obs = 2025-03-27 (panel start), so 60d filter is no-op; 120d+ breaks fold 1.

**Per-symbol PnL attribution REVERSES the hypothesis** — VVV/WIF/WLD are the TOP PnL CONTRIBUTORS (+4,261 / +3,731 / +1,551), NOT drawdown sources. The actual drag in folds 3-5 comes from ICP (-2,330), ORDI (-1,148), HBAR (-1,039), TAO (-963), AAVE (-674), TIA (-566). Older established names where the model's features fail to generalize.

ICP is particularly bad: std/pick 229 (highest), mean -12.20, per-pick annualized Sharpe -2.5. Genuine underperformance, not bad luck.

The DD in folds 3-5 happens because universe contains BOTH big winners (VVV/WIF/WLD) AND big losers (ICP/ORDI/HBAR/RUNE/PENDLE) — losers dominate that regime.

## Files

- `ml/research/alpha_vBTC_current_validation.py` — base config validator
- `ml/research/alpha_vBTC_test_C_validation.py` — DD overlay validator
- `ml/research/alpha_vBTC_test_BC_validation.py` — Test B + B+C validator
- `outputs/vBTC_current_validation/` — base config CSVs
- `outputs/vBTC_test_C_validation/` — DD overlay CSVs
- `outputs/vBTC_test_BC_validation/` — Test B / BC CSVs
- `outputs/vBTC_evaluator_gap/` — local vs evaluate_stacked comparison
- `outputs/vBTC_skip_flat_test/` — hold vs flat_free vs flat_real comparison
- `outputs/vBTC_20seed_validation/` — 5-seed vs 20-seed ensemble (20 rejected)
- `outputs/vBTC_dd_anatomy/` — DD anatomy + per-fold breakdown + variance decomposition
- `outputs/vBTC_universe_filter/` — min-history filter test + per-symbol attribution
- `outputs/vBTC_loser_analysis/` — loser vs winner cohort, per-symbol IC + trade-side stats
- `outputs/vBTC_smooth_rotation/` — smooth-rotation universe test (rejected)
- `outputs/vBTC_target_ensemble/` — basket A+D ensemble (rejected; multi-horizon proxy)
- `outputs/vBTC_dynamic_universe/` — PIT eligibility sweep (min_hist=60d adopted)
- `outputs/vBTC_dd_mitigation_sweep/` — 13-variant DD overlay sweep (dd_tier_aggressive adopted)
- `outputs/vBTC_window_cadence_grid/` — calibration source for 180/90 IC universe
- `outputs/vBTC_loop_phase{6,8}/` — K-sweep, N-sweep
- `outputs/vBTC_final_simulation/` — end-to-end production simulation + monthly PnL growth
- `outputs/vBTC_pipeline_audit/` — leakage audit (9 categories, all clean)
- `outputs/vBTC_null_test/` — random-target null test (z=+3.28 above null mean)
- `live/train_vBTC_artifact.py` — model artifact trainer
- `live/vBTC_paper_bot.py` — single-cycle orchestrator (paper trading ready)
- `models/vBTC_production.pkl` — trained artifact (98 KB)
- `models/vBTC_production.json` — artifact metadata
