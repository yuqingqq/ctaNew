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

### Iter 3 (2026-06-06) — P6 sleeve-maturity decay [code, env SLEEVE_DECAY_TAU, monthly-PIT]
| tau | Sharpe | totPnL | maxDD |
|---|---|---|---|
| 2 | +3.327 | +16029 | -3129 |
| 3 | +3.552 | +16211 | -3061 |
| 4 | +3.659 | +16270 | -3010 |
| **equal** | **+3.892** | +16222 | **-2793** |
**Insight:** P6 REJECTED. All decay variants worse on Sharpe AND maxDD; converges to equal as tau→∞. Concentrating
on fresh sleeves kills the cost-amortization/smoothing benefit of the equal 6-sleeve blend (matches vBTC V3.3
decay rejection). KEEP equal-weight sleeves. 3rd clean negative — every risk/sizing lever at local optimum.

### Iter 4 (2026-06-06) — P5 regime-aware Ridge [WF retrain, regime-interacted, monthly-PIT]
P5 regime-Ridge: Sharpe **+3.792** vs baseline +3.892 → LIFT **-0.100** (WORSE); totPnL +15877, maxDD -2794, 7/9 folds.
**Insight:** P5 REJECTED. Regime-interaction (X, X·1[bull], X·1[bear], Ridge-shrunk) HURTS -0.10 — the bear/bull
deviation coefs add noise (bear=18% of bars) without alpha. Plain per-symbol Ridge wins. Confirms iter5-7:
per-symbol coefs ARE the edge; regime-split doesn't close. Scripts: exp_p5_regime_ridge.py.

## ===== v3 LOOP CONSOLIDATED VERDICT (2026-06-06) =====
Tested the v3 queue's structural + model levers on the correct monthly-PIT universe (baseline +3.892), honest gates:
| iter | lever | result |
|---|---|---|
| 1 | global hold {12,16,24,36h} | 24h optimal (Sharpe+maxDD); no lift |
| 2 | global hysteresis N {2,3,4,5} | N=3 optimal; N=4/5 +0.03 but worse maxDD = noise |
| 3 | P6 sleeve-decay {τ=2,3,4} | REJECTED — decay hurts Sharpe+maxDD; equal optimal |
| 4 | P5 regime-aware Ridge | REJECTED — regime-interaction hurts -0.10 (the ALPHA lever) |
| — | XS94 (174 vs 94 rank, earlier) | null -0.02; keep 175 |

**CONCLUSION: convexity v3 is at a robust local optimum.** Every risk/sizing lever (hold, hysteresis, sleeve-
weighting) AND the one alpha lever (regime-aware model) confirm the production config is optimal-or-better. No new
Sharpe found. This corroborates the extensive prior body of work ("v3 space exhausted"). Remaining queue items
(P2 per-regime hold, P3 asym hysteresis, P4 flip-degross) are risk-shaping variants whose GLOBAL forms already
tested flat/negative (iter1/2) → strong priors against, high code cost, near-certain negatives — not worth grinding.

**The genuine wins this session were validation/infra, not v3 alpha:** stale-preds fix (90e117c), universe-refresh
correction (+1.6 Sharpe, a0ca282 — the single biggest lever, already in production via monthly retrain), liveness
gate (365019b), collector sync (20c764f/d16bbbc). **Decisive next step: the live forward test, not more backtests.**

### Iter 5 (2026-06-06) — DATA-DRIVEN failure analysis [diagnostic]
Decomposed baseline_mpit (+3.89) by leg/regime/tail/beta:
- **Long leg LOSES net (-2389, Sh -0.51)**; short leg carries all (+20318, Sh +3.59). Long has +alpha (+6580) but
  +1.04 BTC beta drags it negative in side(-2117)/bear(-1554); only bull long is +.
- **Book carries -0.178 net-short BTC beta** (long beta +1.04, short -1.22 → shorts higher-beta). corr(net,btc_fwd)=-0.13.
- **Worst 5% cycles (135% of net) are short SQUEEZES** (BTC rips up; corr short vs btc_fwd -0.65; worst-50 btc_fwd +102bps).
- **KEY: alpha-only (beta-neutral proxy) book = Sharpe +4.53 vs actual +4.30 (+0.23), PnL +18485 vs +17929** — net-short
  beta is DRAG not bet (market ~flat over sample → beta added variance/squeezes, no return). Better in ALL regimes.
**→ Test: beta-neutralize the book (the v2 "near-matched betas" assumption is FALSE; 1.04 vs 1.22).**

### Iter 5b (2026-06-06) — beta-neutral sizing test [env SIDE_BETA_NEUT, monthly-PIT]
BN=0 equal-wt +3.893 (control, reproduces) | BN=1 beta-neutral side **+3.657** (-0.24, worse maxDD -3288).
**Insight:** ideal beta-neutral helps (+0.23 ceiling) but realized per-name beta sizing HURTS -0.24 — trailing
per-name betas too noisy at 4h. v2 was right to drop it (wrong stated reason: noise, not matched betas).

### Iter 6 (2026-06-06) — BTC-beta hedge [analytical, PIT trailing beta]
Trailing-window (60/90/120/180) book-beta BTC hedge: ALL hurt (-0.29 to -0.53). Both neutralization paths fail
(per-name -0.24, aggregate hedge -0.29+). The +0.23 "ideal alpha-only" was LOOK-AHEAD (uses contemporaneous beta).
**Insight:** crypto betas non-stationary at 4h → net-short -0.178 beta is REAL but UNHEDGEABLE risk. Monitor, accept.
Equal-weight is the best achievable book. Beta-neutralization direction CLOSED.

### Iter 7 (2026-06-06) — long-downweight in side/bear [analytical + directional check]
Analytical long×0.7 in side/bear: +0.25 Sharpe / maxDD -2131 (looked great). BUT per-fold lift corr with fold BTC
return = **-0.86**: helps all DOWN folds (+0.1..+1.2), CRUSHED in UP folds (f6 -1.25, f7 -1.41). REJECTED — it's a
"lean net-short in weak regimes" DIRECTIONAL bet that paid only because the sample is bear-heavy; blows up in bull.
**Insight:** the long-bleed/short-squeeze/net-short weaknesses are all BETA/directional — unhedgeable OR directional
mirage. Equal-weight book is regime-robust. Real alpha = short-leg cross-sectional residual (already captured).
