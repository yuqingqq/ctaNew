# Diff Protocol — backtest-server reproduction vs the live convexity_v2 baseline

**Scope: through 2026-06-21.** Purpose: confirm the backtest server reproduces the live
forward-test *modeled* track **before** trusting any experiment delta (liquidity floor,
concentration cap, asymmetric short). If the baseline doesn't reproduce, the experiment
results are meaningless — a delta could be the change *or* an environment mismatch.

## 0. The baseline
- `data/live_export/convexity_v2_cycles_through_2026-06-21.csv` — 573 cycles, 03-03 → 06-21 20:00.
- **OOS-live window = 144 cycles since 5.29**, cumulative **+17.801%** (this is the number to match).
- It is the live system's own `cycles.csv` (modeled, perfect-fill net of a cost assumption).

## 1. Reproduction config — must match EXACTLY
| item | value |
|---|---|
| model | `live/models/convexity_v1_short_model.pkl` — **sha1 `dc86ff04c2ce9785`**, fit_cut 2026-05-29, 174 per-sym models, full_V0 (17 feats). **Tracked in repo — use this file, do not retrain.** |
| env | `COST_BPS_LEG=4.5  STRAT_K=3  SIDE_MODE=default  XS_LEAN=1  CONVEXITY_PIT_DVOL=1` |
| strategy | `BEAR_MODE=equal  STOP_SKIP_REGIMES=bear  SIDE_BETA_NEUT=0  BEAR_K=2  SIZING_MODE=inv_vol  LONG_MAX_RET3D=0.20` |
| structure | HOLD=6 overlapping 4h sleeves · K=2/side · regime hysteresis N=3 (`effective_regime_series`) |
| replay | `python -m live.convexity_paper_bot --replay-all` → writes `STATE/cycles.csv` |

The 17 V0 features: `return_1d, atr_pct, obv_z_1d, vwap_slope_96, bars_since_high,
bars_since_high_xs_rank, autocorr_pctile_7d, corr_to_btc_1d, beta_to_btc_change_5d,
idio_vol_to_btc_1h, idio_vol_to_btc_1d, funding_rate, funding_rate_z_7d,
funding_rate_1d_change, rvol_7d, ret_3d, btc_rvol_7d`.

## 2. Data dependencies (the reproduction chain)
```
klines + funding  ──►  panel (V0 features)  ──►  preds (base+long)  ──►  replay  ──►  cycles.csv
```
All four are **gitignored & geo-block-affected** — transport what the server lacks:
- **funding** — ✅ already pushed (`data/funding_export/`; run `scripts/unpack_funding_export.py`).
- **klines** — needed to build the panel. The server is geo-blocked; if its June klines are
  missing/Vision-sourced they may differ from the live collector's WS klines (backfilled gaps).
  **#1 divergence source.** Ask to transport the OOS-window klines if Stage A fails.
- **panel** (`outputs/vBTC_features/panel_expanded_v0.parquet`, 13M) — needed for the **liquidity
  test's ADV/volume feature**. Transport if the server can't rebuild it from klines.
- **preds** (`live/state/convexity_v2/base.parquet` + `long.parquet`, ~4M) — the direct input to
  `select_legs`/replay. **Transporting these lets you reproduce the baseline + run the concentration
  test without klines/panel.** (Liquidity test still needs the panel for ADV.)

> Replaying **from transported preds** is the fastest way to validate the replay logic + run the
> concentration experiment. Building **from klines** is required only to also reproduce the preds
> (Stage B) and to run the liquidity experiment.

### Provided in this repo (`data/live_export/`)
- **`maturity_meta.parquet`** — the `onboardDate` eligibility grid. **Required** (the replay reads it;
  can't be regenerated — geo-blocked Binance call). Point the run at it:
  `export CONVEXITY_UNIVERSE_META=$PWD/data/live_export/maturity_meta.parquet`.
- **`base_live_golden.parquet`, `long_live_golden.parquet`** — the live preds (`fold=-1`, pure
  frozen-model). Use as the **Stage-B golden reference**: diff your regenerated `base/long.parquet`
  against these (`|Δpred| < 1e-5`); or replay straight from them as a deterministic golden run.
- `convexity_v2_cycles_through_2026-06-21.csv` (the baseline) + `../funding_export/` (funding).

**Models** are tracked in `live/models/` (short + long) — just `git pull`. Still build the **panel**
from your klines+funding (or request the 13M `panel_expanded_v0.parquet` to skip the rebuild and diff
Stage A directly). With those, the chain is fully deterministic — nothing else is needed.

## 3. Staged diff — run in order, stop at the first stage that fails
**A — klines / panel inputs.** Diff V0 features per (symbol, open_time) vs the live panel (if
transported); at minimum verify per-symbol `close` matches the live klines over the OOS window
(all price features cascade from close). `bars_since_high_xs_rank` must be ranked over the **full
174+BTC cohort each bar** — a partial cohort drifts it for *every* symbol. `funding_rate` must
match after unpack.
**B — predictions.** Diff `pred` (and `pred_long`) per (symbol, open_time). If the panel matched,
LGBM is deterministic → `|Δpred| < 1e-5`. A larger Δ means the panel diverged → back to A.
**C — decisions.** Per cycle: `regime` (exact), `top_k_long`/`bot_k_short` (exact **set**),
`net_after`/`gross_after_stop` (1e-4). regime mismatch → check hysteresis N=3 + `btc_ret_30d`.
**D — P&L (headline).** Per cycle: `pnl_bps`, `gross_pnl_bps`, and the leg attribution. Then the
cumulative-OOS match. Use the helper:
```bash
python scripts/diff_cycles.py --candidate <your_run>/cycles.csv
# PASS = all fields within tolerance AND |Δ cumulative-OOS| < 0.1%
```

## 4. Tolerances
| field | tol | type |
|---|---|---|
| close price | exact | same kline |
| V0 features | 1e-4 | float (standardized) |
| bars_since_high_xs_rank | exact | rank |
| funding_rate | exact | same source |
| pred / pred_long | 1e-5 | deterministic LGBM |
| regime, stop_engaged | exact | categorical |
| top_k_long / bot_k_short | exact set | categorical |
| net_after, gross_after_stop, turnover | 1e-4 | float |
| pnl_bps, gross_pnl_bps | 0.5 bps | float |
| long/short_ret/alpha_bps | 1.0 bps | float |
| cumulative OOS | 0.1% | float |

## 5. Known caveats — do NOT flag these as failures
- **27 NaN-alpha cycles** (`2026-06-05 12:00 → 2026-06-09 20:00`): baseline has NaN
  `long/short_(ret|alpha)_bps` (settle-path logging gap, fixed 4d55ecc). **`pnl_bps` is valid** there;
  the diff tool auto-skips the NaN cells.
- **18 funding peers partial-June** (through 06-06): `SPX, STABLE, STBL, STRK, TIA, TON, TURBO,
  USUAL, VIRTUAL, VVV, WIF, WLD, WLFI, W, XPL, ZEC, ZEN, ZK` — scored but **never traded (0 picks)**,
  so their forward-filled June funding cannot change any pick or pnl_bps.
- **06-08 collector restart-gap** (the 16:35 bar) was FAPI-backfilled in the live klines; if the
  server's klines differ there, expect a ≤1-cycle blip around 06-08 only.
- **The real-fill track (+6.32% real PnL) is NOT reproducible offline** — it needs live Hyperliquid
  fills. Only the **modeled `cycles.csv`** is reproducible. Don't diff the real-fill ledger.

## 6. Once the baseline PASSES
Change `select_legs` for the experiment, re-replay, and diff the experiment run **against this same
validated baseline** — only then are the deltas (Sharpe, maxDD, the 06-18→06-21 drawdown) trustworthy.
