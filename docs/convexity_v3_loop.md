# Convexity v3 optimization loop (2026-06-06+) — monthly-PIT universe, honest gates

**Baseline:** monthly-PIT universe, v2 stack (equal-wt, K=3, bear K=2, inv-vol, stop-off) = **Sharpe +3.892 / totPnL +16,222 / maxDD -2,793** (n=1463, Oct04→6/4).
**Gates:** beat baseline AND per-fold ≥6/9 AND matched-placebo p95 AND (tuned params) nested-OOS. Reject fold-concentrated/mirage wins.
**Hard rule:** monthly-PIT universe always (frozen universe understates by ~1.6 Sharpe — see a0ca282).

## Ledger

### Iter 1 (2026-06-06) — global hold sweep [env-only, monthly-PIT]
| HOLD | hold | Sharpe | totPnL | maxDD |
|---|---|---|---|---|
| 3 | 12h | +3.014 | +15244 | -3592 |
| 4 | 16h | +3.578 | +17024 | -3242 |
| **6** | **24h** | **+3.893** | +16222 | **-2793** |
| 9 | 36h | +3.337 | +12171 | -2469 |
**Insight:** 24h is the GLOBAL optimum on both Sharpe and maxDD; shorter holds hurt both, longer hurts Sharpe.
Confirms the production hold. Prior now AGAINST P2 "bear=12h" (12h -0.88 globally) — but bear is ~1/3 of cycles,
so per-regime isolation still needed before closing P2. No global lift. KEEP HOLD=6.

### Iter 2 (2026-06-06) — global hysteresis sweep [env-only, monthly-PIT]
| N | Sharpe | totPnL | maxDD |
|---|---|---|---|
| 2 | +3.757 | +15721 | -2793 |
| **3** | **+3.892** | +16222 | **-2793** |
| 4 | +3.921 | +16216 | -2872 |
| 5 | +3.926 | +16174 | -2925 |
**Insight:** N=3 near-optimal; N=4/5 +0.03 Sharpe but WORSE maxDD + flat PnL = within noise. No lift. KEEP N=3.
Two cheap structural levers (hold, hysteresis) both confirm production at local optimum → "v3 exhausted" prior holding.
