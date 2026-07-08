# Convexity v3 — canonical end-to-end flow (STABLE)

**This document describes the one stable v3**: the `convexity-v3-regime-gate` branch config, frozen in
`run_convexity_v3_regime_gate.sh` (backtest) and `run_convexity_v3_live.sh` (live). These two drivers are
byte-identical in strategy env and are the single source of truth. Numbers reported for v3 (in-sample
Sharpe +2.23, maxDD −4276) come from exactly this recipe.

> The `docs/convexity_v3_loop.md` optimization loop (long-winner gate `LONG_MAX_RET3D=0.20`, entry-hour gate)
> is a **separate experimental lineage and is NOT part of stable v3**. Stable v3 keeps `LONG_MAX_RET3D=999`
> (long-winner gate OFF). Ignore the loop config when reasoning about the deployed strategy. Scripts that still
> carry the loop gates (`ddloop_run.sh`, `convexity_v1_cycle_once.sh`) are experimental, not the deploy target.

Both v1 and v3 share the same model *class* (per-symbol RidgeCV, two books) but **not the same fitted model**:
v3 uses the **`V0_LEAN` feature set (funding features DROPPED)** — native preds `hl_lean175/` + deployed artifact
`train_v3_artifact.py`. v1 native preds (`hl/`) use the **full `V0` set (funding features INCLUDED)**. The two pred
sets correlate 0.88 (mean|diff| 0.035) — a real model-input difference, not just execution. They also differ in
strategy construction + execution — see `V1_vs_V3_FLOW.md` for the variant diff. This doc is the full pipeline.

---

## Stage 0 — Data
- **Panel**: `outputs/vBTC_features/panel_expanded_v0.parquet` — 4h bars, ~175 symbols, features + `funding_rate`
  + `alpha_vs_btc_realized` (the beta-neutral residual return that is the label).
- **Live incremental**: `incremental_xs_feats.py` → `incremental_panel.py` append the latest bars each cycle;
  `ingest_funding_fapi.py` refreshes funding.

## Stage 1 — Model training (`train_v3_artifact.py`)
- Two books, each a **per-symbol RidgeCV** on standardized features, recency-weighted (HL=60d):
  - **base** = `V0_LEAN` feature set → `convexity_v3_base_model.pkl`
  - **residrev (long)** = `V0_LEAN + resid_rev` → `convexity_v3_residrev_model.pkl`
- **`fit_cut = latest panel bar − 1d`** (point-in-time; no look-ahead).
- Monthly walk-forward: retrain on the training box, commit the two `.pkl`, live box `git pull`s them.

## Stage 2 — Prediction (`predict_v3_incremental.py`)
- Loads both artifacts, predicts the trailing window → writes `base.parquet` (short/base book) and
  `long.parquet` (long book) with columns `symbol, open_time, pred`.
- First run seeds 300d of history (`PREDICT_SEED_DAYS=300`); afterwards appends incrementally.
- Backtest uses the pre-computed frozen pred files
  (`hl_lean175/v0full_hl60.parquet`, `hl_residrev_lean/v0full_hl60.parquet`).

## Stage 3 — Universe eligibility (`eligible_universe_at`, per cycle, PIT)
From the full ~175-symbol model universe, each 4h cycle drops names failing **any** gate (order as in `eligible_universe_at`):
1. **Hygiene-exclude** — static set of stablecoins (`HYGIENE_EXCLUDE` = USDC/BUSD/TUSD/DAI/FDUSD/…). Reason `hygiene`.
2. **Maturity** — listed ≥ **180 days** (`CONVEXITY_MIN_HISTORY_DAYS=180`, from onboardDate/first bar). Reason `maturity_<180`.
3. **Liquidity** — trailing-30d dollar volume ≥ **$3M/day** (`LIQ_FLOOR_DOLLAR_VOL_30D=3_000_000`, PIT via `CONVEXITY_PIT_DVOL=1`). Reason `liquidity_<`.
4. **Liveness** — drop delisted/halted: > **85%** flat-return days over trailing **7d** (`LIVENESS_MAX_ZERO_FRAC=0.85`, `LIVENESS_WIN_DAYS=7`). Caught e.g. VINEUSDT.
- (Optional, env-gated, OFF in stable v3: `SYM_ALLOWLIST`, `CONVEXITY_DYNAMIC_ALLOWLIST_PATH`.)
- Result: **~145 eligible names/cycle** (137–155; the tradeable pool). **Stable v3 does NOT vol-truncate** — volatility is
  handled by sizing (Stage 6), not exclusion. (v1's top-80 truncation is the key thing v3 dropped.)
- **NOT wired: correlation dedup.** `DEDUP_CORR_THRESHOLD=0.90` is a dead constant (spec'd in iter-036, never
  implemented in `eligible_universe_at`). Consequence: correlated/duplicate names are NOT dropped, and `CONC_CAP`
  is per-name not per-cluster — so factor/correlated-name concentration is uncontrolled. Known gap, not active.

## Stage 4 — Regime classification
- **Static label** (`regime_for_cycle`): `btc_ret_30d` = BTC 180-bar (30d) return.
  `> +0.10` → **bull**; `< −0.10` → **bear**; else **side**. Smoothed by hysteresis.
- **Performance regime gate** (the operative classifier, `REGIME_GATE=1`): a *thermometer* on the strategy's OWN
  trailing realized cross-sectional L/S edge over `REGIME_GATE_W=180` cycles (top/bottom `K=2`, full universe,
  min-hist 60). **binary** mode: if trailing edge ≤ 0 (momentum regime) → **de-gross the whole book to the floor
  (0.0)**; if > 0 → full gross. `VOL_TARGET_CAP=1.0` (de-gross only, never lever up). This — not the static bull/
  side/bear label — is what protects the book in momentum regimes.

## Stage 5 — Leg selection + regime overlays (`select_legs`)
Rank eligible names by `pred`; pick legs per the regime, applying gates:
- **K**: `STRAT_K=2` shorts, `STRAT_K_LONG=1` long (asymmetric — side long alpha lives entirely in the
  top-conviction pick; the 2nd long is −EV). `SIDE_BETA_NEUT=0` (not beta-neutralized).
- **SHORT_MIN_RET3D = −0.20**: veto shorting names already down > 20% over 3d (they squeeze/bounce; −57bp cohort).
- **BULL** (`BULL_MODE=sidealpha`): short-alt book + **25% BTC-long ballast** (`BULL_LONG_MULT=0.25`,
  `BULL_LONG_INSTRUMENT=btc`, hedge cost 2bps) to cap net-short; bull short ranked by `return_1d`
  (`BULL_SHORT_RANK=return_1d`); bull sleeves hold **1 bar** (`STRAT_HOLD_BULL=1`, front-loaded edge).
  - **BULL_DEEP_THR = 0.15**: open **no NEW** bull sleeve when `btc_ret_30d ≥ +15%` (hot melt-up squeezes the short
    leg); active sleeves roll off via hold, not liquidated. *The one OOS-validated longtail lever.*
- **BEAR** (`BEAR_MODE=equal`, `BEAR_K=2`): **depth-ramp** (`BEAR_DEPTH_RAMP=1`, `D0=0.10 D1=0.30`) — bear gross
  scales continuously with drawdown depth (0 at −10%, full at −30%); short only works in deep capitulation, is
  anti-alpha in the shallow grind. `STOP_SKIP_REGIMES=bear`.

## Stage 6 — Sizing (`SIZING_MODE=inv_sqrt_vol`)
- Weight ∝ **1/√vol** per name, normalized so basket gross is unchanged (`raw[s]=1/√vol[s]`, then rescaled).
  Missing/zero vol → median. Down-weights high-vol names instead of excluding them.
- **CONC_CAP = 0.40** with **`CONC_CAP_SINGLE_EXEMPT=1`**: cap any single name at 40% of a side's gross via
  water-fill — but exempt a side that has collapsed to ONE name (with K_LONG=1 the long side often is one name;
  capping it would throw away long weight and create hidden net-short). This exemption is a correctness fix.

## Stage 7 — Sleeves / hold
- **HOLD = 6** overlapping sleeves → 24h effective hold (6 × 4h). A new sleeve opens each cycle; the oldest
  rolls off. Book exposure = aggregate of active sleeves (`aggregate_active_sleeves`).
- **Bull hold = 1** (only the newest bull sleeve carries; front-loaded edge).

## Stage 8 — Accounting (per cycle)
`equity ← equity × (1 + gross_pnl − cost_unit − fund_unit)`
- **gross_pnl**: signed net weight × realized forward residual per leg.
- **cost_unit** (`cost_of`): turnover × cost. `COST_BPS_LEG=9` floor **+ per-symbol depth-cost model**
  (`DEPTH_COST_CSV=persym_cost_cal.csv`, tier `cost_10k`) — thin names charged more. 2× v1's flat 4.5.
- **fund_unit** (`CHARGE_FUNDING=1`): `FUND_CYCLE_FRAC(0.5) × Σ net_weight × fund_pit`, where
  `fund_pit = funding_rate.shift(2)` (8h settled, PIT). Longs pay positive funding, shorts receive. BTC-hedge leg
  charged BTC perp funding via `_btc_funding_at()` (from `data/ml/cache/funding_BTCUSDT.parquet`, ~0.21bps/8h).
  - **Coverage caveat**: panel funding is 100% in-sample (2025+) but only ~53–60% in OOS 2023–24; missing bars
    charged 0, so OOS funding drag is slightly under-counted (small, given ~0.6–0.9bps/8h magnitudes).

---

## Frozen config (single source of truth)
| group | setting |
|---|---|
| execution | `COST_BPS_LEG=9` + depth-cost (`cost_10k`), `CHARGE_FUNDING=1`, `CONVEXITY_PIT_DVOL=1`, `XS_LEAN=1` |
| K / sizing | `STRAT_K=2`, `STRAT_K_LONG=1`, `BEAR_K=2`, `SIZING_MODE=inv_sqrt_vol`, `SIDE_BETA_NEUT=0` |
| conc cap | `CONC_CAP=0.40`, `CONC_CAP_SINGLE_EXEMPT=1` |
| short gate | `SHORT_MIN_RET3D=-0.20`, `LONG_MAX_RET3D=999` (long-winner gate OFF) |
| regime gate | `REGIME_GATE=1 W=180 K=2 FLOOR=0.0 MINHIST=60 MODE=binary UNIV=full` |
| bull | `BULL_MODE=sidealpha GROSS_MULT=1 LONG_MULT=0.25 LONG_INSTRUMENT=btc K=2 HOLD=1 SHORT_RANK=return_1d DEEP_THR=0.15`, `BTC_HEDGE_COST_BPS=2` |
| bear | `BEAR_MODE=equal`, `BEAR_DEPTH_RAMP=1 D0=0.10 D1=0.30`, `STOP_SKIP_REGIMES=bear` |
| hold | `STRAT_HOLD=6` (24h), `STRAT_HOLD_BULL=1` |
| universe | hygiene-exclude (stables), maturity≥180d, liquidity≥$3M/30d, liveness (>85%flat/7d = dead); **no vol-truncation; dedup NOT wired** |

## Reproduce / run
- **Backtest**: `bash live/run_convexity_v3_regime_gate.sh <STATE_DIR>` (uses frozen preds; `--replay-all`).
- **Live**: `tmux new -d -s cvx3 'bash live/run_convexity_v3_live.sh'` (refresh → predict → `--cycle` loop).
- **Parity gate**: `bash live/parity_v3.sh` (replay-all vs replay-from+cycle; 0/359 decision mismatch validated).
- **Deploy**: see `V3_LIVE_DEPLOY.md`. Retrain monthly on the training box (`train_v3_artifact.py`) + push; live box pulls.

## Config validation — leave-one-out ablation (2026-07-02, in-sample, matched pred set)
Baseline full v3 = Sharpe +3.109 / totPnL +22968 / maxDD -4276. Revert each new config to v1/default; ΔSh>0 = validated.
VALIDATED (removing hurts Sharpe): bull sidealpha stack +0.734, bear equal+ramp +0.698, REGIME_GATE +0.391,
K_LONG=1 +0.379, stop-off-bear +0.325, BEAR_DEPTH_RAMP +0.282, BULL_DEEP_THR +0.280, inv_sqrt_vol +0.090.
NEUTRAL in-sample (value is OOS/tail): SHORT_MIN -0.20 (Δ0.00; crash protection, tail-only), beta-neut OFF (+0.03).
Long-winner divergence RESOLVED: adding it ON = -0.08 Sh on stable stack → keeping OFF is correct.
Feature diffs validated separately (funding-out, truncation-out). Execution diffs (cost 9, funding) = realism, not tuned.

## CONC_CAP threshold sweep (2026-07-02) — tail-limiter, real return cost
cap 0(off): Sh+3.359 PnL+27961 maxDD-4672 CVaR5-503 worstCyc-2139 | cap 0.40(prod): Sh+3.109 PnL+22968 maxDD-4276
CVaR5-453 worstCyc-1888 | 0.50: Sh+3.050 maxDD-5341 | 0.60: Sh+3.134 maxDD-4754 | 0.80: Sh+3.324 maxDD-4672.
FINDINGS: (1) cap=0.40 gives BEST tail on every metric (maxDD/CVaR5/worst-cycle) — confirmed tail-limiter. (2) Costs
the MOST return (lowest PnL, -0.25 Sh vs off). (3) Non-monotonic: 0.50 worst-of-both, 0.60/0.80 recover Sharpe but
lose tail protection. No free-lunch middle → effectively binary. DECISION: keep 0.40 for live forward-test (tail
protection > 0.25 in-sample Sh when validating execution transport). Harness: live/validate_v3_loo.sh + frozen driver sweep.
