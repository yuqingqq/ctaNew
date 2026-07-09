# Alpha-residual system parameter review

Date: 2026-07-09

Scope: the active v4 forward path
`run_convexity_v4_live.sh -> incremental_xs_feats.py -> incremental_panel.py ->
predict_v4_incremental.py -> convexity_paper_bot.py`.

This is an inventory, not a recommendation to sweep the parameters. The system has
many environment-gated research controls, but only a smaller set is active. A parameter
is classified as:

- **Validated/frozen**: supported by the program's dual-window or holdout process.
- **Forward candidate**: intentionally live for prospective evaluation, not adopted.
- **Inherited/heuristic**: active, but not independently established as optimal.
- **Operational**: affects parity, freshness, or availability rather than alpha.
- **Off/rejected**: code remains available, but the active launch path disables it.

## Executive findings

1. The bot exposes 147 environment controls. The v4 strategy is not only residual Ridge plus four KEEPSET4 levers; counting
   inherited defaults, it has roughly 30 active choices that can change positions,
   gross, costs, or eligibility.
2. The launch script does not fully specify the strategy. Important active values are
   inherited from `convexity_paper_bot.py`: regime thresholds and hysteresis, sleeve
   hold, stop floor/window/heal/timeout, deep-bull K/gross, funding fraction, universe
   thresholds, and liveness thresholds.
3. Parameters are coupled. In particular:
   - `STRAT_HOLD` controls both sleeve persistence and the regime-gate outcome delay.
   - `GLOBAL_GROSS_MULT=0.5` multiplies every other gross rule. The deep-bull overlay's
     nominal gross 0.5 therefore becomes 0.25 before stop/regime-gate effects.
   - Regime changes control new sleeves; old sleeves remain until the six-sleeve held
     book rolls off. `BULL_GROSS_MULT=0` is not an instantaneous liquidation.
   - Any sizing or selection change alters turnover, costs, equity, stop activation,
     and later regime-gate paths.
4. `COST_BPS_LEG=9` is shadowed in v4 because `DEPTH_COST_CSV` activates the
   per-symbol cost branch. The active cost controls are the selected depth tier,
   `FEE_BPS_FILL`, realized turnover, and funding.
5. `CONC_CAP=0.99` with `CONC_CAP_SINGLE_EXEMPT=1` is effectively inert in the active
   concentrated book. It is configured but should not be described as a material lever.
6. Most feature-window, model-class, K, dynamic sizing, and stage-2 variants are already
   closed or rejected. They should not become a new unconstrained parameter sweep.

## Parallel v3 versus v4 is not a controlled model comparison

Both launchers are active research paths, but they change many modules at once:

| Control | v4 live | v3 live |
|---|---|---|
| Model target | residual XS z-score | raw-return XS z-score |
| Global gross cap | 0.5 | 1.0 implicit |
| Bear K | inherited 1L/2S | 2L/2S |
| Bear depth ramp | off | on, 0.10 to 0.30 |
| Side beta neutral | on implicit | off |
| Concentration cap | 0.99, effectively inert | 0.40 |
| Short crash filter | off | `ret_3d >= -0.20` |
| Mild-bull construction | no new sleeve | side-alpha, BTC long, return-1d shorts |
| Bull hold | 6 cycles | 1 cycle |
| Deep bull | long-only forward candidate | flat |

Therefore the live v3-v4 PnL difference cannot be attributed to the residual target.
A model comparison requires a matched construction/gross arm with only the prediction
artifact changed. The current loops are valid as two strategy forward ledgers, but not
as a causal model A/B.


## Module 1 - Data, cadence, and parity

| Parameter | Active value | Status | Influence |
|---|---:|---|---|
| Source bar | 5 minutes | inherited/frozen | Base time grid for every feature |
| Decision cadence | 4 hours | validated/frozen | Six decisions/day |
| `XS_LEAN` | 1 | operational | Lean and full feature engines should be identical |
| XS input window | 45 days + 5 files | operational | Incremental feature warmup |
| `XS_RECOMPUTE_TAIL_DAYS` | 14 | operational | Repairs corrected/backfilled klines |
| Panel warmup | 14 days | operational | Supports bounded feature lookbacks |
| `--rebuild-days` | 10 | operational | Propagates late-bar repairs into the panel |
| `PREDICT_RECOMPUTE_DAYS` | 10 | operational | Rewrites only the recent OOS prediction tail |
| BTC completeness lookback/minimum | 8 days / 90% | operational guard | Prevents sparse-beta corruption |
| Thin cross-section delay | 48 hours | operational guard | Defers skewed cross-sectional ranks |

These values should be changed only for parity or data-quality reasons. Treating them
as alpha knobs risks making live and research features inconsistent.

## Module 2 - Feature definitions

The base and short book use 14 V0_LEAN features. The long book adds two residual-
reversal features.

| Feature/family | Effective window or definition | Status |
|---|---|---|
| `return_1d` | 288 x 5m = 1 day | tested family; incumbent frozen |
| `atr_pct` | EWM span 14 x 5m | inherited/heuristic |
| `vwap_slope_96` | 8h VWAP, compared with 25m lag | inherited/heuristic |
| `obv_z_1d` | OBV signal standardized over 288 bars, min 72 | audited/frozen |
| `bars_since_high` | bars since rolling 1d high | audited/frozen |
| `bars_since_high_xs_rank` | same-cycle cross-sectional percentile | audited/frozen |
| `autocorr_pctile_7d` | lag-1 autocorr over about 3h; percentile over 7d | inherited/heuristic |
| `corr_to_btc_1d` | 288 bars, min 72, shifted one bar | tested family; incumbent frozen |
| `beta_to_btc_change_5d` | 1d beta difference over 5d | beta family tested; frozen |
| `idio_vol_to_btc_1h` | 12 bars, min 6, shifted | active sizing input and feature |
| `idio_vol_to_btc_1d` | 288 bars, min 72, shifted | frozen |
| `rvol_7d` | 2016 bars, min 288, shifted | frozen |
| `ret_3d` | 864 bars, shifted | tested family; frozen |
| `btc_rvol_7d` | BTC 2016 bars, min 288, shifted | frozen |
| `resid_rev_2/3` | negative sum of prior 2/3 residual labels | long-book only; frozen |

The feature-window and window-by-horizon programs found no promotable replacement
among their preregistered cells. That closes those tested representatives, not every
possible window. Any future feature change needs a new information source or a specific
mechanism; another broad window sweep is not justified.

## Module 3 - Label and beta estimator

| Parameter | Active value | Status | Influence |
|---|---:|---|---|
| Prediction target horizon | 48 x 5m = 4h | validated/frozen | Stage-1 score horizon |
| Residual beta window | 1 day, min 6h | beta family tested; frozen | Removes BTC component |
| Beta timing | shifted one 5m bar | required PIT | Prevents current-bar leakage |
| Target | same-cycle XS z-score of residual return | v4 forward candidate | Changes model ordering |
| Target clip | +/-10 | inherited/heuristic | Limits extreme target leverage |
| Purge/embargo | `exit_time < fit_cut`, plus 1 day | required PIT | Training eligibility |

The model predicts a 4h target while the portfolio retains six overlapping sleeves
for 24h. This is deliberate in the current design. Sleeve-aligned h12/h72 tests did
not produce a promotable alternative.

## Module 4 - Model training and inference

| Parameter | Active value | Status | Influence |
|---|---:|---|---|
| Model class | separate RidgeCV per symbol | validated incumbent | Stage-1 score |
| Ridge alpha grid | 0.01, 0.1, 1, 10, 100 | inherited/frozen | Regularization |
| Current selected alphas | 100 for 172-173/175 symbols | diagnostic | Model is heavily regularized |
| Sample decay time constant | 60 days e-folding (about 42d half-life) | inherited/heuristic | Regime adaptation |
| Minimum rows/symbol | 300 | inherited/heuristic | Model/universe coverage |
| Standard feature clipping | training q1/q99 | inherited/heuristic | Outlier handling |
| Heavy-tail transform | empirical rank from training set | frozen preprocessing | Scale robustness |
| Retrain schedule | described as monthly, not launch-enforced | operational policy | Model age/drift |
| Current artifact fit cut | 2026-06-29 | current state | Defines OOS boundary |
| Seed requirement | mandatory unless explicit override | required PIT | Preserves forward ledger |

The alpha grid is boundary-saturated: 172-173 of 175 current models select the
maximum alpha (100). This means RidgeCV has not demonstrated an interior regularization
optimum. If stage-1 tuning is ever reopened, use one preregistered extended-grid
sensitivity and judge top-K spread, not training MSE or rank-IC alone.

Pooled Ridge improves average ordering but has not improved production tips; pooled
LGBM and label winsorization failed the tip endpoint. Model-class and label tuning at
fixed features should remain closed.

## Module 5 - Universe and eligibility

| Parameter | Active value | Status | Influence |
|---|---:|---|---|
| `CONVEXITY_MIN_HISTORY_DAYS` | 180 | inherited/heuristic | Excludes young listings |
| `LIQ_FLOOR_DOLLAR_VOL_30D` | $3m/day | execution-motivated | Eligibility and selection pool |
| `CONVEXITY_PIT_DVOL` | 1 | validated correctness | Avoids end-of-sample liquidity leak |
| Dedup correlation threshold | 0.90 | inherited/hard-coded | Removes near-duplicate names |
| Hygiene exclusions | static set | structural | Removes stables/wrapped/non-crypto |
| Liveness gate | on | validated safety | Removes halted/delisted names |
| Liveness window/zero fraction | 7d / 85% | inherited/heuristic | Speed of dead-name removal |
| Universe meta path | maturity metadata | required live input | Controls historical eligibility |

Universe thresholds can materially alter cross-sectional opportunity and cost. They
should be evaluated as universe-policy changes, not tuned jointly with model or regime
parameters.

## Module 6 - Portfolio construction and sleeve logic

| Parameter | Active value | Status | Influence |
|---|---:|---|---|
| `STRAT_K_LONG` | 1 | validated incumbent | Concentration and long-tail capture |
| `STRAT_K_SHORT` | 2 (inherits `STRAT_K`) | validated incumbent | Short alpha breadth |
| `STRAT_HOLD` | 6 cycles = 24h | inherited baseline | Sleeve persistence and turnover |
| `SLEEVE_DECAY_TAU` | 0, equal sleeve weights | tested/rejected alternatives | Age weighting |
| Side construction | default | validated incumbent | Uses model long/short books |
| `SIDE_BETA_NEUT` | 1, implicit default | active but not explicit | Side leg gross allocation |
| Portfolio beta estimator | 30d (180 x 4h), min 7d, shifted | inherited/heuristic | Beta-neutral leg scaling; distinct from the tested label-beta family |
| Bear construction | equal | validated KEEPSET4 alpha lever | Trades model L/S in bear |
| `BEAR_K` | 0 | active inheritance | Bear uses 1 long / 2 shorts |
| Bear raw gross | 1.0, implicit | active inheritance | Before global/stop/gate multipliers |
| Sizing | inverse square-root idio vol | validated keep | Within-leg weights |
| Sizing feature | `idio_vol_to_btc_1h` | implicit default | Risk concentration |
| `CONC_CAP` | 0.99 | effectively inert | Only extreme concentration |
| Single-name cap exemption | on | correctness | Prevents hidden directional tilt |

Top-3 long reduces variance/CVaR but failed the preregistered jackpot bar. It remains
a risk-objective near-miss, not an alpha promotion. Wider K and learned stage-2
selection are rejected under current costs and data.

## Module 7 - BTC regime and regime-specific construction

| Parameter | Active value | Status | Influence |
|---|---:|---|---|
| Regime signal | BTC trailing 30d return | inherited baseline | Bull/side/bear state |
| Bull threshold | +10% | inherited/heuristic | State occupancy |
| Bear threshold | -10% | inherited/heuristic | State occupancy |
| Entry hysteresis | 3 x 4h bars | historically motivated | Transition churn |
| Exit to side | immediate | structural rule | Transition asymmetry |
| Mild-bull gross | 0 | validated KEEPSET4 risk lever | No new mild-bull sleeve |
| Deep-bull threshold | +15% BTC 30d return | forward candidate | Activates long-only overlay |
| Deep-bull mode | top return-1d longs | forward candidate | Candidate exposure/ranking |
| Deep-bull K | 2, implicit | forward candidate | Concentration |
| Deep-bull raw gross | 0.5, implicit | forward candidate | Before global/stop/gate multipliers |
| Bull hold | 6, inherits global hold | inherited | Deep-bull sleeve persistence |

The deep-bull ranking increment is unproven; only long-alt exposure has historical
support. The forward placebo ledger is the verdict-bearing test. Do not tune threshold,
K, and gross simultaneously.

## Module 8 - Adaptive risk overlays

| Parameter | Active value | Status | Influence |
|---|---:|---|---|
| Regime gate | on | validated KEEPSET4 | Whole-book de-gross |
| Gate lookback | 180 cycles = 30d | validated representative | Adaptation speed |
| Gate floor | 0 | validated | Flat when trailing edge <= 0 |
| Gate thermometer K | 2 per side | validated | Edge estimate |
| Gate minimum history | 60 cycles = 10d | inherited in validated cell | Warmup |
| Gate mode/universe | binary/full | validated | Decision law |
| Gate skip regimes | none, implicit | active inheritance | Gate also applies in bear |
| Stop | on outside bear | validated KEEPSET4 | Drawdown control |
| Stop threshold | 2 sigma | validated | Engagement sensitivity |
| Stop floor | 0.40, implicit | inherited in validated cell | Gross while engaged |
| Stop sigma window | 180 cycles, implicit | inherited in validated cell | Vol/DD normalization |
| Stop warmup/heal/timeout | 60 / 0.5 / 90 | implicit hard-coded | State duration |
| Stop skip regimes | bear | validated | Preserves bear reversion |
| Global gross multiplier | 0.5 | holdout-mandated | Scales every position |

The regime gate and stop are path-dependent. Variant estimation should use matched
book-level score/spread endpoints before a full replay; otherwise small prediction
changes can bifurcate risk paths and manufacture apparent performance differences.

## Module 9 - Costs, funding, and execution

| Parameter | Active value | Status | Influence |
|---|---:|---|---|
| Per-symbol depth file | enabled | required net accounting | Slippage by name/size |
| Depth tier | `cost_10k` | AUM assumption | Capacity/slippage |
| Depth calibration coverage | 152/175 model symbols; 23 use median fallback | current-state risk | New or missing names lose symbol-specific cost |
| Active cost distribution | median 9.98 bps slippage + 4.5 bps fee per fill; p90 slippage 16.90 | current calibration | Cost dominates thin alpha |
| Taker fee | 4.5 bps/fill | current assumption | Direct net return |
| `COST_BPS_LEG` | 9, shadowed | inactive in depth branch | No current effect |
| Funding charge | on | validated accounting | Carry |
| Funding cycle fraction | 0.5 | calendar assumption | Converts 8h rate to 4h |
| Turnover | realized `sum(abs(delta weight))` | structural | Fee/slippage driver |

This module is the most credible performance optimization path. Cost parameters must
be calibrated from actual fills and account fee tier, not optimized against PnL.

## Module 10 - Off, rejected, or research-only controls

The following families remain in `convexity_paper_bot.py` but are disabled in v4.
They should not be interpreted as open tuning dimensions:

- Side-flat and regime-band skips.
- Adaptive bull ramp and bull short entry confirmation.
- Bear depth ramp, bear mid-band gross, bear long cuts, and BTC bear hedge.
- Auto regime sizer and volatility target.
- Alternate side modes, defensive long rerankers, and BTC-hedged side modes.
- Long idio/rvol/residual/return/funding filters.
- Short return, funding, correlation, taper, and conviction filters.
- Absolute/relative BTC-correlation universe filters.
- Prediction floors, dispersion gates, dynamic allowlists, and entry-hour gates.
- Bull short feature reranking and short pick hysteresis.
- Tail-score gate, random/placebo controls, and alternate sleeve decay.

Most were rejected, null, era-fragile, or exist only as experimental/placebo
instruments. Reopening one requires a new mechanism, new data, or a materially changed
cost structure.

## What is genuinely open

| Candidate | Test purpose | Priority |
|---|---|---|
| Regime lookback, +/- thresholds, hysteresis | Robustness map only; verify the incumbent sits on a stable plateau rather than select the best historical cell | medium |
| Portfolio beta window/shrinkage | Reduce realized beta error and gross distortion; this is distinct from the rejected label-beta test | medium |
| Ridge alpha-grid extension | Resolve the current maximum-alpha boundary saturation; top-K spread is the endpoint | low-medium |
| Training decay constant | Check adaptation versus sample-size loss with one fixed challenger, not a broad sweep | low-medium |
| Six-sleeve hold under current K=1/2 | Risk/turnover sensitivity; old K=5 evidence favored keeping six, so low expected alpha lift | low |
| Stop floor/window/heal/timeout | Risk-budget calibration only, never an alpha/Sharpe search | low |
| Deep-bull threshold/K/gross | Already frozen as one forward candidate; do not retune on spent history | forward only |
| Gross cap release | Governed by the preregistered bear-forward condition | forward only |

A regime sensitivity should report state occupancy, transition count, turnover,
per-regime net PnL, and worst-period behavior. It should not promote whichever
threshold has the highest full-sample Sharpe.

## Parameter governance recommendations

1. Create one checked-in v4 manifest containing every active value, including defaults.
   The launch script should load it and the bot should log the resolved manifest hash.
2. Validate mode enums and reject unknown values. For example, `BULL_MODE=default`
   currently falls through to momentum behavior even though `default` is not a documented
   mode.
3. Mark shadowed and inert settings at startup (`COST_BPS_LEG` under depth costs,
   `CONC_CAP=0.99`, disabled-family subparameters).
4. Separate parameters into:
   - estimator/model parameters,
   - construction parameters,
   - path-dependent risk parameters,
   - accounting/capacity assumptions,
   - operational parity controls.
5. Never sweep across modules in one experiment. Use preregistered, one-mechanism cells
   with matched populations and dual-era endpoints.
6. Prioritize forward confirmation, execution calibration, and new positioning data.
   Do not spend more search budget on the closed window/model/K grids.

## Recommended tuning order

1. **Accounting calibration, not PnL tuning:** fee tier, depth tier, funding fraction,
   and realized fill/turnover reconciliation.
2. **Forward-only release condition:** retain 0.5 gross until the preregistered bear
   confirmation passes; do not optimize the release threshold retrospectively.
3. **Deep-bull candidate:** keep threshold/K/gross fixed and judge against the frozen
   random-alt forward placebo.
4. **Universe policy:** test maturity/liquidity/liveness only if coverage or fill data
   demonstrates a concrete eligibility problem.
5. **New signal data:** positioning/liquidation depth. Existing price-window and
   stage-2 parameter families remain closed.
