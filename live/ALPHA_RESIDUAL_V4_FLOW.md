# Alpha Residual v4 Flow

Local record of the current live/replay v4 residual-target system. This is the
`live/` convexity v4 thread, not the older `ml/research/alpha_v4_xs.py`
LightGBM cross-sectional research baseline.

## Scope

The current v4 residual system is:

- Model: per-symbol `sklearn.linear_model.RidgeCV`.
- Label: cross-sectional z-score of `alpha_vs_btc_realized`, clipped to
  `[-10, 10]`.
- Cadence: 4h prediction grid.
- Execution harness: `live/convexity_paper_bot.py`.
- Prediction generators:
  - `live/gen_oos_v4.py` for held-out OOS books.
  - `live/gen_residual_target.py` for row-matched return-target vs
    residual-target A/B books.

The historical `ml/research/alpha_v4_xs.py` uses a 5-seed LightGBM ensemble on
a 25-symbol basket residual target. Keep it as historical context only.

## Data Inputs

- Main panel: `outputs/vBTC_features/panel_expanded_v0.parquet`.
- Core panel columns used by v4 generators:
  `symbol`, `open_time`, `exit_time`, `return_pct`,
  `alpha_vs_btc_realized`, plus feature columns.
- `alpha_vs_btc_realized` is the beta-removed forward residual that v4 trains
  on and the replay uses as `alpha_A`.
- Kline data for internal BTC regime/momentum/beta calculations:
  `data/ml/test/parquet/klines/{symbol}/5m/*.parquet`.
- Optional depth cost input:
  `live/state/v3loop/persym_cost_cal.csv`.

## Feature Sets

Source definitions are imported from `live/train_twobook_models.py`, which in
turn loads `research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py`.

`V0` has 17 columns:

```text
return_1d
atr_pct
obv_z_1d
vwap_slope_96
bars_since_high
bars_since_high_xs_rank
autocorr_pctile_7d
corr_to_btc_1d
beta_to_btc_change_5d
idio_vol_to_btc_1h
idio_vol_to_btc_1d
funding_rate
funding_rate_z_7d
funding_rate_1d_change
rvol_7d
ret_3d
btc_rvol_7d
```

`V0_LEAN` has 14 columns: `V0` minus the three funding columns.

Residual-reversal features used by the long/rerank book:

```text
resid_rev_2 = -sum(previous 2 bars of alpha_vs_btc_realized)
resid_rev_3 = -sum(previous 3 bars of alpha_vs_btc_realized)
```

Both are point-in-time because the generator uses `shift(1)` before the rolling
sum. Missing warmup values are filled with `0.0`.

Current feature verdict from the July 2026 investigation:

- Feature additions such as Alpha191 survivors failed net strategy validation.
- In-sample feature prune candidates did not survive OOS.
- Regime-specific feature pruning was tested and rejected.
- The feature layer is considered closed; construction/gating is where the
  remaining edge lives.

## Model Training

Both v4 generators train one independent Ridge model per symbol per fold.

Training protocol:

1. Restrict panel to the 4h grid: `open_time.hour % 4 == 0` and minute `0`.
2. Compute target per 4h bar:
   `xs_z = (alpha_vs_btc_realized - cross_sectional_mean) / cross_sectional_std`,
   clipped to `[-10, 10]`.
3. For each monthly fold, train on rows whose `exit_time < fold_start - 1d`.
4. Skip symbols with fewer than 300 training rows.
5. Preprocess per symbol:
   - heavy-tail features get fold-local rank normalization;
   - other features are winsorized at p1/p99 and z-scored.
6. Fit `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])`.
7. Apply exponential recency weights:
   `weight = exp(-(train_end - open_time) / 60 days)`.

Primary prediction books:

- Base/short book: `V0_LEAN`, residual target.
- Long/rerank book: `V0_LEAN + resid_rev_2 + resid_rev_3`, residual target.

Common outputs:

- OOS base: `live/state/convexity/hl_v4base_oos/v0full_hl60.parquet`.
- OOS long: `live/state/convexity/hl_v4long_oos/v0full_hl60.parquet`.
- Recent A/B books from `gen_residual_target.py`:
  `hl_tgt_{ret,res}_{base,long}/v0full_hl60.parquet`.

Each prediction parquet carries:
`symbol`, `open_time`, `alpha_A`, `return_pct`, `exit_time`, `pred`, and usually
`fold`.

## Replay And Construction

`live/convexity_paper_bot.py` converts prediction ranks into a sleeve portfolio.

1. Load base predictions from `CONVEXITY_PREDS_PATH`.
2. If `CONVEXITY_PREDS_LONG` is set, merge its `pred` as `pred_long`.
3. Compute internal point-in-time features:
   - `mom30`: 30-day 4h momentum, shifted by one bar.
   - trailing beta to BTC, shifted by one bar.
   - `btc_ret_30d` for regime classification.
4. Filter the tradable universe each cycle:
   - minimum maturity, default `180` days;
   - hygiene exclusions;
   - PIT trailing 30d dollar-volume floor;
   - liveness gate for halted/delisted names.
5. Classify regime:
   - bull if BTC 30d return `> +0.10`;
   - bear if `< -0.10`;
   - side otherwise.
   Hysteresis requires `REGIME_HYSTERESIS_N=3` consecutive raw cycles before
   switching into bull or bear.
6. Call `select_legs()`:
   - side: mean-reversion on prediction ranks; long highest `pred_long` if
     available, short lowest base `pred`; beta-neutral if `SIDE_BETA_NEUT=1`.
   - bear: default `BEAR_MODE=flat`; with `BEAR_MODE=equal`, trade equal-weight
     L/S on the same mean-reversion ranks.
   - bull: default `BULL_MODE=mom`, rank by `mom30` momentum rather than model
     prediction. `BULL_MODE=sidealpha` forces prediction-based mean reversion.
7. Aggregate 6 overlapping 4h sleeves by default (`STRAT_HOLD=6`, 24h effective
   hold). `STRAT_HOLD_BULL` can shorten bull aggregation.
8. Apply overlays in replay order:
   concentration cap, vol-normalized drawdown stop, `REGIME_GATE`, and optional
   vol targeting.
9. Mark PnL with `return_pct`; residual alpha attribution uses `alpha_A`.

Replay writes local state under `CONVEXITY_STATE`, including:
`cycles.csv`, `regime.csv`, `equity.csv`, `sleeves.csv`,
`predictions.parquet`, `universe.csv`, `positions.json`,
`edge_hist.json`, and `replay_summary.json`.

## Costs And Funding

`cost_of()` has two branches:

- With `DEPTH_COST_CSV`: cost is
  `sum(abs(delta_weight) * (per_symbol_depth_cost + FEE_BPS_FILL))`.
  `FEE_BPS_FILL` defaults to `4.5` bps per fill.
- Without `DEPTH_COST_CSV`: flat cost fraction is `turnover * 0.5 * COST`,
  where `COST = COST_BPS_LEG * 1e-4` if set, otherwise `4.5e-4`.

Funding is optional via `CHARGE_FUNDING=1` and uses contemporaneous funding as
realized carry, not as a predictive input. The default cycle fraction is
`FUND_CYCLE_FRAC=0.5` for a 4h bar.


## Value Audit

Checked against code on 2026-07-07.

| Value | Code source | Status |
|---|---|---|
| Model class = per-symbol `RidgeCV` | `live/gen_oos_v4.py`, `live/gen_residual_target.py` | matched |
| Ridge alpha grid = `[0.01, 0.1, 1.0, 10.0, 100.0]` | `X6_controlled_matrix.py::RIDGE_ALPHAS` | matched |
| Generator embargo = `1d` | `gen_oos_v4.py::EMB`, `gen_residual_target.py::EMB` | matched |
| Recency half-life divisor = `60.0` days | `gen_oos_v4.py::HL`, `gen_residual_target.py::HL` | matched |
| Minimum rows per symbol = `300` | generator `if len(g) < 300: continue` | matched |
| Residual target clip = `[-10, 10]` | generator target construction | matched |
| `V0` = 17 cols, `V0_LEAN` = 14 cols | `train_twobook_models.py`, `X6_controlled_matrix.py` | matched |
| `resid_rev_2/3` use `shift(1).rolling(...)` | `gen_oos_v4.py`, `gen_residual_target.py` | matched |
| Replay code default `STRAT_K=5` | `convexity_paper_bot.py::K` | matched; differs from vanilla-v4 override |
| Replay code default `STRAT_HOLD=6` | `convexity_paper_bot.py::HOLD` | matched |
| Replay code default `BULL_MODE=mom` | `convexity_paper_bot.py::BULL_MODE` | corrected here; old flow doc wrote `default`, which behaves like momentum but is not the literal default |
| Replay code default `BEAR_MODE=flat` | `convexity_paper_bot.py::BEAR_MODE` | matched |
| Replay code default `SIDE_BETA_NEUT=1` | `convexity_paper_bot.py::SIDE_BETA_NEUT` | matched |
| Regime thresholds `+0.10/-0.10` | `REGIME_BULL_THR`, `REGIME_BEAR_THR` | matched |
| Regime hysteresis `3` | `REGIME_HYSTERESIS_N` | matched |
| Momentum window `180` 4h bars | `MOM_WINDOW` | matched |
| Maturity floor `180d` | `CONVEXITY_MIN_HISTORY_DAYS` | matched |
| Liquidity floor `$3,000,000` trailing daily dollar volume | `LIQ_FLOOR_DOLLAR_VOL_30D` | matched |
| Liveness gate default on, `7d`, max flat fraction `0.85` | `LIVENESS_GATE`, `LIVENESS_WIN_DAYS`, `LIVENESS_MAX_ZERO_FRAC` | matched |
| Depth-cost fee add-on `4.5` bps/fill | `FEE_BPS_FILL` | matched |
| Funding cycle fraction `0.5` | `FUND_CYCLE_FRAC` | matched |
| Regime-gate defaults `W=180`, `floor=0.0`, `K=2`, `minhist=60`, `binary`, `full` | `REGIME_GATE_*` | matched |
| Frozen A/B stack values | `live/run_v4_ab.sh` | matched |

## Configs In Play

Vanilla v4 explicit replay config, normalized against code from
`live/CONVEXITY_V4_FLOW.md`. These are experiment overrides, not all code
defaults:

```bash
STRAT_K=2
STRAT_K_LONG=1
# derived if STRAT_K_SHORT is unset: STRAT_K_SHORT=2
BEAR_MODE=flat
SIZING_MODE=inv_sqrt_vol
REGIME_GATE=0
BEAR_DEPTH_RAMP=0
BULL_MODE=mom
CONC_CAP=0.99
STOP_SKIP_REGIMES=side,bear,bull
SHORT_MIN_RET3D=-999
```

Optimized v4 under the July 2026 tuning ledger:

```bash
# vanilla plus:
BEAR_MODE=equal
REGIME_GATE=1
```

Frozen v3 stack used for target A/B in `live/run_v4_ab.sh`:

```bash
COST_BPS_LEG=9
CHARGE_FUNDING=1
CONVEXITY_PIT_DVOL=1
XS_LEAN=1
CONVEXITY_UNIVERSE_META=outputs/vBTC_features/panel_expanded_v0.parquet
DEPTH_COST_CSV=live/state/v3loop/persym_cost_cal.csv
DEPTH_COST_TIER=cost_10k
CONVEXITY_DVOL_CACHE_PKL=live/state/v3loop/ddloop/_dvol_cache.pkl
STRAT_K=2
STRAT_K_LONG=1
BEAR_K=2
SIDE_MODE=default
SIZING_MODE=inv_sqrt_vol
SIDE_BETA_NEUT=0
BEAR_MODE=equal
STOP_SKIP_REGIMES=bear
LONG_MAX_RET3D=999
SHORT_MIN_RET3D=-0.20
BEAR_DEPTH_RAMP=1
BEAR_DEPTH_D0=0.10
BEAR_DEPTH_D1=0.30
CONC_CAP=0.40
CONC_CAP_SINGLE_EXEMPT=1
REGIME_GATE=1
REGIME_GATE_W=180
REGIME_GATE_FLOOR=0.0
REGIME_GATE_K=2
REGIME_GATE_MINHIST=60
REGIME_GATE_MODE=binary
REGIME_GATE_UNIV=full
BULL_MODE=sidealpha
BULL_GROSS_MULT=1
BULL_LONG_MULT=0.25
BULL_LONG_INSTRUMENT=btc
BTC_HEDGE_COST_BPS=2
BULL_K=2
STRAT_HOLD_BULL=1
BULL_SHORT_RANK=return_1d
BULL_DEEP_THR=0.15
# per target run:
CONVEXITY_PREDS_PATH=live/state/convexity/hl_tgt_{ret,res}_base/v0full_hl60.parquet
CONVEXITY_PREDS_LONG=live/state/convexity/hl_tgt_{ret,res}_long/v0full_hl60.parquet
CONVEXITY_STATE=live/state/longtail/v4_ab/{ret,res}
```

## Current Read

- The residual-target change is structurally cleaner but not a clear Sharpe
  upgrade versus the return target. The full-stack verdict is net-neutral.
- v4 raw, gates-off is regime-fragile: recent in-sample looked strong, but
  backward OOS was negative.
- The model is a candidate selector, not a regime timer. It finds reversion
  candidates; whether they revert is regime-dependent.
- The robust construction levers are coarse regime routing, `REGIME_GATE`, and
  farming bear with `BEAR_MODE=equal`.
- Feature-layer tuning is closed unless new orthogonal data is added.
- K shape from diagnostics: long edge is concentrated at rank 1; short edge is
  a broader rank band. `K_long=1`, `K_short=3` is the current practical pick.

## Reproduction Commands

Generate OOS residual-target books:

```bash
python3 live/gen_oos_v4.py 2023-01-01 2025-10-01
```

Generate row-matched return-target vs residual-target recent books:

```bash
python3 live/gen_residual_target.py
```

Run the frozen stack A/B from the generated books:

```bash
bash live/run_v4_ab.sh
```

Run a direct replay with an explicit state directory:

```bash
CONVEXITY_PREDS_PATH=live/state/convexity/hl_tgt_res_base/v0full_hl60.parquet \
CONVEXITY_PREDS_LONG=live/state/convexity/hl_tgt_res_long/v0full_hl60.parquet \
CONVEXITY_STATE=live/state/longtail/v4_manual_check \
python3 -m live.convexity_paper_bot --replay-all
```

## Historical v4 Research Baseline

`ml/research/alpha_v4_xs.py` is different:

- Model: 5-seed LightGBM regression ensemble.
- Horizon: 48 5m bars.
- Features: `features_ml.cross_sectional.XS_FEATURE_COLS`, 17 columns.
- Target: beta-adjusted residual versus an equal-weight basket.
- Evaluation: cross-sectional top/bottom quintile portfolio.

Do not mix its model/config claims with the current live v4 residual-target
Ridge system.
