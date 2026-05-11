# vBTC drawdown root cause analysis

Date: 2026-05-10
Strategy: alpha_vBTC (cross-sectional residual on Binance USDM, 25-name universe)

## TL;DR

Drawdowns are a **fat-tail problem**, not a chronic-underperformance problem.

- Bottom 1% of cycles (9 of 900) cause **29%** of all losses
- Bottom 5% of cycles (45 of 900) cause **69%** of all losses
- **VVVUSDT alone appears in 57% of the worst-30 cycles** — confidence-wrong predictions during regime shifts

Best **feasible** variance reducer: Test C (trailing-DD deleveraging at 20% drawdown → size=0.5).
Yields +4.84 Sharpe (vs baseline +4.27), 26% DD reduction, validated on 9-fold walk-forward.

## Root cause structure

### Concentration in worst cycles

| Slice | n cycles | Total adverse PnL | % of total losses |
|---|---|---|---|
| Bottom 1% | 9 | -8,165 bps | 29% |
| Bottom 5% | 45 | -19,539 bps | 69% |
| Bottom 10% | 90 | -28,498 bps | 100% (all neg) |

### Symbol concentration in worst-30 cycles

| Symbol | Occurrences | Mean adverse move |
|---|---|---|
| **VVVUSDT** | **17** (57%) | -51 bps avg, but extremes to -1,237 |
| ORDIUSDT | 3 | -943 bps avg |
| GMXUSDT | 2 | +336 bps adverse (short squeeze) |
| NEARUSDT | 2 | +390 bps adverse |
| INJUSDT | 1 | -541 bps |

### Failure-mode pattern: model overconfidence on VVV in regime shifts

Examples (worst-30 cycle log):

| Cycle | VVV pred | VVV realized | Direction |
|---|---|---|---|
| 2026-01-10 08:00 | +0.0555 (very high long signal) | -202 bps | wrong |
| 2026-03-02 20:00 | +0.0147 (long) | **-1,237 bps** | wrong (catastrophic) |
| 2026-02-14 04:00 | +0.0126 (long) | -616 bps | wrong |
| 2026-01-23 04:00 | -0.0209 (strong short) | +249 bps | wrong (squeeze) |
| 2026-01-06 16:00 | +0.0117 (long) | -481 bps | wrong |

Pattern: VVV's high-magnitude predictions tend to be confidence-wrong during periods when basket dispersion is HIGH (>0.7 percentile), not low. So `conv_gate` doesn't filter them — the model is "confident-wrong" precisely when the gate says go.

Mean dispersion percentile of worst-30: **0.63** (vs all-cycle 0.70). Only slightly below average; the mechanism is regime-specific not signal-strength-specific.

## Mitigation tests (this report)

Tests B–F iterate on variance-reduction mechanisms. Stats use 5-fold production (folds 5–9) with 10 LGBM seeds.

| Mechanism | Sharpe | std_bps | max_DD | Feasibility |
|---|---|---|---|---|
| **Baseline** | +4.27 | 261 | -3,763 | already production |
| Test B: continuous-sigmoid dispersion sizing | +4.34 | 191 (-27%) | -2,718 (-28%) | ✓ feasible |
| Test C: trailing-DD deleveraging (dd>20% → 0.5) | **+4.84** | 213 | -2,789 (-26%) | ✓ feasible (recommended) |
| Test D: B+C combined | +4.71 | 155 (-41%) | -1,389 (-63%) | ✓ feasible (but WF -0.33 cost) |
| Test E: inverse-vol weighting | +3.46 | 276 | -3,725 | ✓ feasible, **doesn't help** |
| Test F: per-cycle floor at -300 bps | +6.07 | 240 | -3,277 | THEORETICAL only — see Test G |
| Test G: realistic intra-cycle stop -300 | +3.77 | 280 | -4,305 | ✗ rejected (whipsaws hurt) |

### Why inverse-vol weighting (Test E) failed

Weighting by 1/σ_i caps VVV's allocation, but VVV is a **net winner** — in cycles where the model gets VVV right, the win is proportionally large. Inverse-vol cuts those wins symmetrically with the losses, so net Sharpe drops (-0.31).

VVV's blow-ups are not driven by allocation; they're driven by **directional prediction failure**. Inverse-vol doesn't address direction.

### Why Test F (theoretical floor) overstates the gain

Capping per-cycle PnL at -300 bps yields +6.07 Sharpe **if** we could exit instantly at the floor. But this assumes cycles whose ending PnL is below -300 are the same as cycles that touch -300 mid-flight.

**Test G (real 5-min intra-cycle stop) refutes this.**

| Threshold | Test F (theoretical) | Test G (realistic) | Verdict |
|---|---|---|---|
| -100 | +9.76 / DD -950 | +2.29 / DD -4,758 | **destroys Sharpe** |
| -300 | +6.07 / DD -3,277 | +3.77 / DD -4,305 | **flat Sh, worse DD** |
| -500 | +5.32 / DD -3,489 | +3.40 / DD -4,778 | -1.9 Sharpe |

**Why Test G fails:** cycles that touch -300 mid-cycle often recover by end-of-cycle. Stop-loss locks in those would-have-recovered losses (whipsaw). The cross-sectional alpha implicitly bets on spread reversion — a hard stop cuts that reversion.

DD gets *worse* under stops because whipsaw losses chain into longer drawdown sequences.

**Conclusion: intra-cycle stop-loss is a dead path. Don't invest in the live-execution infra.**

## Recommendation

**Production**: deploy Test C (trailing-DD deleveraging) — it's already validated:
- +0.57 Sharpe (4.27 → 4.84)
- 26% DD reduction
- 9-of-9-fold walk-forward consistency
- No new infrastructure

**Closed paths** (ruled out by this analysis):
- Inverse-vol weighting (Test E): cuts wins symmetrically, -0.31 Sharpe
- Intra-cycle stop-loss (Test G): whipsaws on mean-reverting alpha, makes DD worse

## Files

- `ml/research/alpha_vBTC_dd_root_cause.py` — cycle-level diagnostic, worst-30 attribution
- `ml/research/alpha_vBTC_test_E_vol_weighted.py` — inverse-vol weighting (failed)
- `ml/research/alpha_vBTC_test_F_stop_loss.py` — per-cycle floor (theoretical upper bound)
- `ml/research/alpha_vBTC_test_G_intracycle_stop.py` — 5-min intra-cycle stop (realistic, rejected)
- `outputs/vBTC_dd_root_cause/` — cycle_logs.csv, worst30_cycles.csv, extreme_adverse_moves.csv
- `outputs/vBTC_test_E_vol_weighted/test_E_results.csv`
- `outputs/vBTC_test_F_stop_loss/test_F_results.csv`
- `outputs/vBTC_test_G_intracycle/test_G_results.csv`
