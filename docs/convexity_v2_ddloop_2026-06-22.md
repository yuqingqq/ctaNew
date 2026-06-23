# Convexity v2 — drawdown investigation + mitigation (validated)

Trigger: the live forward-test report attributed the −15%/3-day drawdown (06-18→06-22) to illiquid
micro-cap short squeezes and proposed liquidity-floor / concentration-cap / asymmetric-short fixes.
**Every number below is from a replay/test run on this box; claims without a test are not included.**
Data basis: frozen 5.29 per-symbol RidgeCV models (`convexity_v1_{short,long}_model.pkl`, fit_cut 2026-05-29),
real Binance June funding (FAPI export `data/funding_export/`, unpacked to cache), Vision klines to 06-21,
`maturity_meta` universe. Replay window Oct-04-2025 → Jun-21-2026.

## 1. Backtest vs live reconciliation (measured)
- **Preds bit-exact**: my regenerated preds vs the live golden preds (`docs/xref/...0529_0604`): max│Δ│ **0.0000** over 3,854 rows.
- **Realized returns match**: golden vs panel `return_pct`, corr **0.9999** (only diffs on the 06-08 restart-gap bar).
- **Replaying the live golden preds** through this engine (exact base config) reproduces live regime 99.7%,
  short-picks 98.4%, long-picks 91.8%.
- **The cumulative-OOS gap (live +17.80% vs replay +6.98%) is config-vintage, not data/model**: `LONG_MAX_RET3D=0.20`
  was enabled in the live config on **2026-06-08** (commit f51b77f; absent in parent). Replay with gate OFF matches
  live pre-06-08 long-picks 90%; gate ON matches post-06-08 98% — neither single config matches both.
- **The earlier "backtest≠live" gap was funding**: before the real-funding export, 0/94 symbols had real June
  funding (forward-filled from 05-31); substituting real funding moved peak +17%→+28.6% (live ~+29%) and
  ME-shorted 2×→10× (live 12×).
- **Regime detection is PIT** (verified): `btc_ret_30d` reconstructed from raw klines using only data ≤ bar t
  matches the bot's regime input (best match = `close[t-1]/close[t-181]`, i.e. one bar lagged); hysteresis is
  causal (N=3 consecutive past cycles to switch in, instant exit).

## 2. Drawdown attribution (measured, from live cycles 06-18→06-21)
- Total −1389 bps, 100% bear regime, 97% of loss is alpha (long_alpha −846 + short_alpha −506 of −1389).
- **Long leg is the larger, broad loser**: long_alpha −846, long-pick hit-rate 28% (vs 59% pre-drawdown).
- **Short leg is a concentrated tail**: short_alpha −506, but median short alpha **+8 bps** (53% hit) — net
  driven by 2–3 names (worst-3 short picks = 81% of the short loss).
- **Per-regime alpha (full history, live + backtest)**: bear long_alpha −4.1/−4.6, short_alpha +10.0/+8.8;
  bull long +7.0/−2.7, short +17.0/+10.1; side long +13.1/+0.2, short +13.3/+5.0. Long leg negative in both
  trending regimes; short leg positive in all regimes.

## 3. Squeeze not forecastable ex-ante (measured, bear shorts, worst-5% = squeeze)
- Squeeze-discrimination AUC (bear-only): funding_rate **0.506**, funding_z_7d **0.495**, ret_3d **0.589**.
- Shorting ripped names is net-positive: bear shorts with ret_3d>30% mean **+5.52%** (77% win); a
  `SHORT_MAX_RET3D` gate drops names whose mean (+5.52%) exceeds the kept set (+0.44%).
- Crowded shorts (funding_z<−1) mean **+1.77%** (best bucket). → no tested feature separates squeezes.

## 4. Mitigations tested (real-funding replay; base = production config, Sharpe +1.97 / maxDD −2741 / June −24.6%)
| mitigation | Sharpe | maxDD | June DD | verdict |
|---|---|---|---|---|
| liquidity floor $15/25/40M | −0.19/−0.05/−0.42 | worse | worse | rejected |
| concentration cap (global & bear-only) | 1.49–1.94 | ~−2600 | ~−23% | rejected (costs Sharpe, no tail help) |
| short_fund_floor=0 | +0.82 | −2741 | −24.6 | rejected (removes edge) |
| short_max_ret3d=0.30 | +0.97 | −2578 | −22.7 | rejected (removes edge) |
| symmetric bear de-gross 0.3 | +2.43 | −2197 | −8.0 | tail cut real; Sharpe gain bootstrap-NS → risk-reduction only |
| bear long-cut → net-short (BLM=0) | +2.87 | −2309 | −21.1 | +0.36 of gain is net-short beta (\|net\| 0.28) |
| **bear long-cut + BTC hedge** | **+2.51** | **−2201** | **−18.4** | **adopted (see §5)** |
| bull long-cut | +1.97 (unch) | −2741 | — | no effect |

## 5. Recommended fix: `BEAR_HEDGE_BTC=1` — validation results
Construction: in bear, drop alt longs, short alts (the +alpha leg), add a BTC long sized to neutralize the
short-basket beta. vs production base:
- **Per-regime**: bear Sharpe +0.60→+1.18, bear maxDD −2741→−2033; side +2.33→+2.52; bull unchanged.
- **Matched placebo**: model bear-long basket realized 24h-alpha **−35.6 bps** vs random eligible longs **+13.8 bps**
  (5 seeds) — the model's bear longs underperform random.
- **Split-half**: H1 (Oct-Feb) +1.33→+1.78, H2 (Mar-Jun) +2.77→+3.59.
- **Per-month**: ≥ base in 6/8 months.
- **Beta-neutral**: avg │net│ 0.09 (vs 0.28 for the un-hedged net-short cut).
- **Alpha vs beta (measured)**: BTC-hedge +2.51 vs un-hedged net-short +2.87 vs base +1.97 → +0.54 from the
  beta-neutral cut, +0.36 additional from the net-short beta.
- **No-op verified**: `BEAR_HEDGE_BTC=0` reproduces production cycles, max│Δpnl_bps│ = 0.0.
- `LONG_MAX_RET3D=0.20` on top: Sharpe 2.51 vs 2.56 off (within noise), maxDD −2201 vs −2361 → kept for the
  shallower maxDD.

### Caveats (measured limits, not adopted as positive claims)
- Bootstrap dSharpe +0.54, **CI90 [−0.87,+1.83], P=0.76 — not significant** on 244 days.
- Per-cycle CVaR5 unchanged (−258→−257): the fix reduces the bear drawdown, not the per-cycle squeeze tail.
- Sample contains 0 bear cycles with btc_ret_30d > −5% (no near-reversal bear in-sample).
- In-sample only; forward test is the arbiter.

## Recommended config (env-gated, byte-identical to production when off)
`BEAR_MODE=equal BEAR_K=2 SIZING_MODE=inv_vol LONG_MAX_RET3D=0.20 BEAR_HEDGE_BTC=1`
