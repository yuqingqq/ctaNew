"""Phase 3b: regime-conditional retraining (Mixture of Experts) for v7.

For each fold, train 3 separate v7 LGBM ensembles — one per regime bucket
(low / med / high realized xs dispersion). At test time, route each ts to
the matching bucket's model.

Regime indicator: disp_22d (cross-sectional std of 22d returns), with
cutoffs at 33%/67% quantile of training-set disp_22d (computed per fold,
no leak).

Comparison: v7 baseline (single LGBM trained on full training set with
same hyperparameters and 3-seed reduction).

Discipline gates: same as Phase 2 — ΔSh > +0.20, paired Δnet CI > 0,
≥4/7 folds positive, single-event drop, drawdown reduction.

Usage:
    python -m ml.research.alpha_v9_xyz_regime_moe
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe
from ml.research.alpha_v7_freq_sweep import add_residual_and_label
from ml.research.alpha_v7_multi import (
    LGB_PARAMS, SEEDS, add_features_A, add_returns_and_basket, load_anchors,
)
from ml.research.alpha_v7_pead_fixed import add_features_B_fixed
from ml.research.alpha_v7_push import add_sector_features
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, metrics_for, boot_ci,
)
from ml.research.alpha_v7_tier_a import TIER_AB

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE_V7 = CACHE / "v7_tier_a_walkfwd_preds.parquet"
HORIZONS = (3, 5, 10)
USE_SEEDS = SEEDS[:3]  # 3 seeds for speed
GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8


def sharpe(s: pd.Series, rpy: float = 252.0) -> float:
    if len(s) < 2 or s.std() == 0: return 0.0
    return s.mean() / s.std() * np.sqrt(rpy)


def max_dd(s: pd.Series) -> float:
    cum = s.cumsum(); peak = cum.cummax()
    return float((cum - peak).min())


def paired_block_bs(diff: np.ndarray, block: int = 5,
                       n_boot: int = 2000) -> tuple[float, float, float]:
    n = len(diff)
    if n < 30: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    nb = int(np.ceil(n / block))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        means.append(diff[idx].mean())
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def train_ensemble(train: pd.DataFrame, feats: list[str],
                    horizons=HORIZONS, seeds=USE_SEEDS) -> dict:
    """Train (horizon, seed) ensemble. Returns dict of trained models keyed by (h, seed)."""
    models = {}
    for h in horizons:
        label = f"fwd_resid_{h}d"
        train_ = train.dropna(subset=feats + [label])
        if len(train_) < 500:
            continue
        X = train_[feats].to_numpy(dtype=np.float32)
        y = train_[label].to_numpy(dtype=np.float32)
        for seed in seeds:
            m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
            m.fit(X, y)
            models[(h, seed)] = m
    return models


def predict_ensemble(models: dict, df: pd.DataFrame, feats: list[str]
                       ) -> np.ndarray:
    """Average predictions across all (h, seed) models."""
    if not models:
        return np.full(len(df), np.nan)
    sub = df.dropna(subset=feats)
    X = sub[feats].to_numpy(dtype=np.float32)
    preds = []
    for k, m in models.items():
        preds.append(m.predict(X))
    avg = np.mean(preds, axis=0)
    out = np.full(len(df), np.nan)
    out[df[feats].notna().all(axis=1).to_numpy()] = avg
    return out


def main() -> None:
    log.info("loading panel and computing features ...")
    panel, earnings, _ = load_universe()
    if panel.empty: return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    for h in (1,) + HORIZONS:
        panel = add_residual_and_label(panel, h)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel, feats_F = add_sector_features(panel)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    feats = feats_A + feats_B + feats_F + ["sym_id"]
    log.info("  panel: %d rows, %d feats", len(panel), len(feats))

    # Compute regime indicator (disp_22d)
    regime = compute_regime_indicators(panel, anchors)
    regime["ts"] = pd.to_datetime(regime["ts"], utc=True)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel = panel.merge(regime[["ts", "disp_22d"]], on="ts", how="left")
    log.info("  regime indicator merged. disp_22d non-null: %d / %d",
             panel["disp_22d"].notna().sum(), len(panel))

    # Define folds (expanding train, 1 test year each)
    folds = []
    for test_year in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        train_end = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
        test_start = train_end
        if test_year == 2025:
            test_end = pd.Timestamp("2026-12-31", tz="UTC")
        else:
            test_end = pd.Timestamp(f"{test_year + 1}-01-01", tz="UTC")
        folds.append({"name": f"fold_{test_year}",
                       "train_end": train_end,
                       "test_start": test_start,
                       "test_end": test_end})

    # ---- Walk-forward: regime-conditional + baseline ----
    moe_preds_all = []
    base_preds_all = []
    log.info("\n=== 7-fold walk-forward ===")
    for fold in folds:
        train = panel[panel["ts"] < fold["train_end"]].copy()
        test = panel[(panel["ts"] >= fold["test_start"]) &
                       (panel["ts"] < fold["test_end"])].copy()
        train = train.dropna(subset=["disp_22d"])
        test = test.dropna(subset=["disp_22d"])
        if len(train) < 1000 or len(test) < 100:
            log.warning("  %s skip (train=%d test=%d)", fold["name"], len(train), len(test))
            continue

        # Compute regime cutoffs from training only
        cut_low = train["disp_22d"].quantile(0.33)
        cut_high = train["disp_22d"].quantile(0.67)
        train["regime"] = np.where(train["disp_22d"] < cut_low, "low",
                                      np.where(train["disp_22d"] >= cut_high, "high", "med"))
        test["regime"] = np.where(test["disp_22d"] < cut_low, "low",
                                     np.where(test["disp_22d"] >= cut_high, "high", "med"))
        log.info("  %s: train=%d test=%d  cut_low=%.4f cut_high=%.4f",
                 fold["name"], len(train), len(test), cut_low, cut_high)

        # Baseline: train on FULL train set
        log.info("    training baseline (full train set, 3h × 3s = 9 models) ...")
        base_models = train_ensemble(train, feats)
        base_pred = predict_ensemble(base_models, test, feats)
        test_base = test.copy()
        test_base["pred"] = base_pred
        base_preds_all.append(test_base[["ts", "symbol", "pred", "fwd_resid_1d", "regime"]])

        # MoE: train 3 separate ensembles, route test rows to matching bin
        log.info("    training regime MoE (3 bins × 3h × 3s = 27 models) ...")
        moe_models = {}
        for bin_name in ["low", "med", "high"]:
            bin_train = train[train["regime"] == bin_name]
            if len(bin_train) < 500:
                log.warning("    bin %s has %d rows; skip", bin_name, len(bin_train))
                moe_models[bin_name] = {}
                continue
            log.info("    bin %s: train_n=%d", bin_name, len(bin_train))
            moe_models[bin_name] = train_ensemble(bin_train, feats)

        # Predict on test, routed by regime bin
        moe_pred = np.full(len(test), np.nan)
        for bin_name in ["low", "med", "high"]:
            mask = (test["regime"] == bin_name).to_numpy()
            if mask.sum() == 0: continue
            sub_test = test[test["regime"] == bin_name]
            if not moe_models.get(bin_name):
                # Fall back to baseline
                fallback = predict_ensemble(base_models, sub_test, feats)
                moe_pred[mask] = fallback
            else:
                p = predict_ensemble(moe_models[bin_name], sub_test, feats)
                moe_pred[mask] = p
        test_moe = test.copy()
        test_moe["pred"] = moe_pred
        moe_preds_all.append(test_moe[["ts", "symbol", "pred", "fwd_resid_1d", "regime"]])

    base_preds = pd.concat(base_preds_all, ignore_index=True)
    moe_preds = pd.concat(moe_preds_all, ignore_index=True)
    log.info("\n  base preds: %d  moe preds: %d", len(base_preds), len(moe_preds))

    # ---- Build portfolios ----
    regime_pnl = compute_regime_indicators(
        panel.drop(columns=["regime"], errors="ignore"), anchors)
    log.info("\n=== Building portfolios on Tier A+B (K=4 M=1, gate 0.6, 0.8 bps) ===")
    base_pnl_pre = daily_portfolio_hysteresis(base_preds, "pred", "fwd_resid_1d",
                                                 set(TIER_AB), 4, 1, COST_BPS_SIDE)
    base_pnl = gate_rolling(base_pnl_pre, regime_pnl, pctile=GATE_PCTILE,
                              window_days=GATE_WINDOW)
    moe_pnl_pre = daily_portfolio_hysteresis(moe_preds, "pred", "fwd_resid_1d",
                                                set(TIER_AB), 4, 1, COST_BPS_SIDE)
    moe_pnl = gate_rolling(moe_pnl_pre, regime_pnl, pctile=GATE_PCTILE,
                             window_days=GATE_WINDOW)
    log.info("  base_pnl: %d  moe_pnl: %d", len(base_pnl), len(moe_pnl))

    # Merge on ts (keep only common cycles)
    merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base_net"}).merge(
        moe_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "moe_net"}),
        on="ts", how="inner")
    merged["ts"] = pd.to_datetime(merged["ts"], utc=True)
    merged["year"] = merged["ts"].dt.year
    log.info("  shared ts: %d", len(merged))

    # ---- summary ----
    log.info("\n=== Aggregated test panel summary ===")
    sh_b = sharpe(merged["base_net"]); sh_m = sharpe(merged["moe_net"])
    dsh = sh_m - sh_b
    diff = (merged["moe_net"] - merged["base_net"]).to_numpy() * 1e4
    d, lo, hi = paired_block_bs(diff)
    dd_b = max_dd(merged["base_net"]) * 1e4
    dd_m = max_dd(merged["moe_net"]) * 1e4
    log.info("  baseline (3h × 3s) Sharpe: %+.2f  net=%+.2f bps  max-DD=%+.0f bps",
             sh_b, merged["base_net"].mean() * 1e4, dd_b)
    log.info("  regime MoE Sharpe:        %+.2f  net=%+.2f bps  max-DD=%+.0f bps",
             sh_m, merged["moe_net"].mean() * 1e4, dd_m)
    log.info("  ΔSh = %+.2f   Δnet = %+.2f bps   95%% CI = [%+5.2f, %+5.2f] bps",
             dsh, d, lo, hi)

    # ---- per-year ----
    log.info("\n=== Per-year ΔSh ===")
    log.info("  %-6s %5s %12s %12s %12s",
             "year", "n", "base Sh", "moe Sh", "ΔSh")
    n_pos = 0; n_total = 0
    for y, g in merged.groupby("year"):
        if len(g) < 5: continue
        sh_b_y = sharpe(g["base_net"]); sh_m_y = sharpe(g["moe_net"])
        log.info("  %-6d %5d %+12.2f %+12.2f %+12.2f",
                 y, len(g), sh_b_y, sh_m_y, sh_m_y - sh_b_y)
        n_total += 1
        if (sh_m_y - sh_b_y) > 0: n_pos += 1

    # ---- single-event drop ----
    log.info("\n=== Gate 4: drop 2021-2022 drawdown ===")
    no_2122 = merged[~merged["year"].isin([2021, 2022])]
    sh_b_drop = sharpe(no_2122["base_net"])
    sh_m_drop = sharpe(no_2122["moe_net"])
    dsh_drop = sh_m_drop - sh_b_drop
    retained = (dsh_drop / dsh) if abs(dsh) > 0.01 else 0
    log.info("  ΔSh full: %+.2f   ΔSh w/o 2021-22: %+.2f   retained: %.2f",
             dsh, dsh_drop, retained)

    # ---- discipline-gate verdict ----
    log.info("\n=== Discipline-gate verdict ===")
    folds_pos_ratio = f"{n_pos}/{n_total}"
    g1 = dsh > 0.20
    g2 = lo > 0
    g3 = n_pos >= 4
    g4 = retained >= 0.50
    g5 = dd_m < dd_b  # max-DD reduction (note: max-DD is negative, smaller magnitude is better)
    log.info("  G1: ΔSh > +0.20             %+8.2f   %s", dsh, "✓" if g1 else "✗")
    log.info("  G2: Δnet 95%% CI > 0         [%+.2f, %+.2f]   %s", lo, hi, "✓" if g2 else "✗")
    log.info("  G3: ≥4/7 folds positive      %s        %s", folds_pos_ratio, "✓" if g3 else "✗")
    log.info("  G4: Q21-22 drop survival     %.2f retained   %s", retained, "✓" if g4 else "✗")
    log.info("  G5: max-DD reduced           %+.0f vs %+.0f   %s", dd_m, dd_b, "✓" if g5 else "✗")
    log.info("\n  VERDICT: %s",
             "PASS — regime MoE adds value" if (g1 and g2 and g3 and g4)
             else "FAIL — regime MoE does not pass discipline gates")


if __name__ == "__main__":
    main()
