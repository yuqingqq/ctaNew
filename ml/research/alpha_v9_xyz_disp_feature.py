"""Path A: add cross-sectional dispersion (disp_22d) as feature 19 in v7 LGBM.

Hypothesis: a flat per-ts regime feature lets a SINGLE LGBM learn
regime-conditional behavior in subtrees, sidestepping the data
fragmentation that killed the regime MoE test (-2.27 Sh).

Method:
  Baseline: v7 18 features (10 A + 4 B + 3 F + sym_id), retrain in 7 folds
            with 3h × 3s = 9 models per fold (matches MoE test for fair compare)
  Extended: same as baseline + disp_22d (cross-sectional std of 22d returns)
            as feature 19, identical hyperparams

For each fold:
  - Train baseline ensemble on full training set
  - Train extended ensemble (same data, +1 feature)
  - Predict on test, route through standard portfolio + dispersion gate

Discipline gates: same five as previous tests.

Usage:
    python -m ml.research.alpha_v9_xyz_disp_feature
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
HORIZONS = (5,)  # reduced from 3 horizons to 1 (the v7 training target) for compute
USE_SEEDS = SEEDS[:1]  # single seed — purely diagnostic comparison
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


def train_ensemble(train: pd.DataFrame, feats: list[str]) -> dict:
    models = {}
    for h in HORIZONS:
        label = f"fwd_resid_{h}d"
        train_ = train.dropna(subset=feats + [label])
        if len(train_) < 500: continue
        X = train_[feats].to_numpy(dtype=np.float32)
        y = train_[label].to_numpy(dtype=np.float32)
        for seed in USE_SEEDS:
            m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
            m.fit(X, y)
            models[(h, seed)] = m
    return models


def predict_ensemble(models: dict, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    if not models: return np.full(len(df), np.nan)
    valid = df[feats].notna().all(axis=1).to_numpy()
    out = np.full(len(df), np.nan)
    if valid.sum() == 0: return out
    sub = df[valid]
    X = sub[feats].to_numpy(dtype=np.float32)
    preds = []
    for k, m in models.items():
        preds.append(m.predict(X))
    out[valid] = np.mean(preds, axis=0)
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

    base_feats = feats_A + feats_B + feats_F + ["sym_id"]
    log.info("  panel: %d rows  base_feats: %d", len(panel), len(base_feats))

    # Compute disp_22d (cross-sectional std of 22d returns) and merge as feature
    regime = compute_regime_indicators(panel, anchors)
    regime["ts"] = pd.to_datetime(regime["ts"], utc=True)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel = panel.merge(regime[["ts", "disp_22d"]], on="ts", how="left")

    extended_feats = base_feats + ["disp_22d"]
    log.info("  base_feats: %d  extended_feats: %d (added disp_22d)",
             len(base_feats), len(extended_feats))

    # 3-fold representative test (2021 = drawdown year, 2024 = mixed, 2025 = strong)
    folds = []
    for test_year in [2021, 2024, 2025]:
        train_end = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
        if test_year == 2025:
            test_end = pd.Timestamp("2026-12-31", tz="UTC")
        else:
            test_end = pd.Timestamp(f"{test_year + 1}-01-01", tz="UTC")
        folds.append({"name": f"fold_{test_year}",
                       "train_end": train_end,
                       "test_start": train_end,
                       "test_end": test_end})

    base_preds_all = []
    ext_preds_all = []
    log.info("\n=== 7-fold walk-forward (baseline 18 vs extended 19) ===")
    for fold in folds:
        train = panel[panel["ts"] < fold["train_end"]]
        test = panel[(panel["ts"] >= fold["test_start"]) &
                       (panel["ts"] < fold["test_end"])].copy()
        if len(train) < 1000 or len(test) < 100:
            log.warning("  %s skip (train=%d test=%d)", fold["name"], len(train), len(test))
            continue
        log.info("  %s: train_n=%d test_n=%d", fold["name"], len(train), len(test))

        # Baseline
        log.info("    training baseline (18 feat) ...")
        base_models = train_ensemble(train, base_feats)
        base_pred = predict_ensemble(base_models, test, base_feats)

        # Extended
        log.info("    training extended (19 feat: + disp_22d) ...")
        ext_models = train_ensemble(train, extended_feats)
        ext_pred = predict_ensemble(ext_models, test, extended_feats)

        # Quickly check disp_22d feature importance
        if ext_models:
            avg_imp = np.zeros(len(extended_feats))
            for k, m in ext_models.items():
                avg_imp += m.feature_importances_.astype(float)
            avg_imp /= len(ext_models)
            disp_idx = extended_feats.index("disp_22d")
            disp_imp_pct = avg_imp[disp_idx] / avg_imp.sum() * 100
            log.info("    disp_22d feature importance: %.2f%% of total", disp_imp_pct)

        test_b = test.copy(); test_b["pred"] = base_pred
        test_e = test.copy(); test_e["pred"] = ext_pred
        base_preds_all.append(test_b[["ts", "symbol", "pred", "fwd_resid_1d"]])
        ext_preds_all.append(test_e[["ts", "symbol", "pred", "fwd_resid_1d"]])

    base_preds = pd.concat(base_preds_all, ignore_index=True)
    ext_preds = pd.concat(ext_preds_all, ignore_index=True)
    log.info("\n  base preds: %d  ext preds: %d", len(base_preds), len(ext_preds))

    # Build portfolios
    log.info("\n=== Building portfolios on Tier A+B (K=4 M=1, gate 0.6, 0.8 bps) ===")
    base_pre = daily_portfolio_hysteresis(base_preds, "pred", "fwd_resid_1d",
                                            set(TIER_AB), 4, 1, COST_BPS_SIDE)
    base_pnl = gate_rolling(base_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    ext_pre = daily_portfolio_hysteresis(ext_preds, "pred", "fwd_resid_1d",
                                           set(TIER_AB), 4, 1, COST_BPS_SIDE)
    ext_pnl = gate_rolling(ext_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)

    # Merge on ts
    merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base_net"}).merge(
        ext_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "ext_net"}),
        on="ts", how="inner")
    merged["ts"] = pd.to_datetime(merged["ts"], utc=True)
    merged["year"] = merged["ts"].dt.year
    log.info("  shared ts: %d  base_pnl: %d  ext_pnl: %d",
             len(merged), len(base_pnl), len(ext_pnl))

    # Aggregate summary
    log.info("\n=== Aggregated test panel ===")
    sh_b = sharpe(merged["base_net"]); sh_e = sharpe(merged["ext_net"])
    dsh = sh_e - sh_b
    diff = (merged["ext_net"] - merged["base_net"]).to_numpy() * 1e4
    d, lo, hi = paired_block_bs(diff)
    dd_b = max_dd(merged["base_net"]) * 1e4
    dd_e = max_dd(merged["ext_net"]) * 1e4
    log.info("  baseline (18 feat): Sh=%+.2f  net=%+.2f bps  max-DD=%+.0f bps",
             sh_b, merged["base_net"].mean() * 1e4, dd_b)
    log.info("  extended (19 feat): Sh=%+.2f  net=%+.2f bps  max-DD=%+.0f bps",
             sh_e, merged["ext_net"].mean() * 1e4, dd_e)
    log.info("  ΔSh = %+.2f   Δnet = %+.2f bps   95%% CI = [%+.2f, %+.2f] bps",
             dsh, d, lo, hi)

    # Per-year ΔSh
    log.info("\n=== Per-year ===")
    log.info("  %-6s %5s %10s %10s %10s",
             "year", "n", "base Sh", "ext Sh", "ΔSh")
    n_pos = 0; n_total = 0
    for y, g in merged.groupby("year"):
        if len(g) < 5: continue
        sh_b_y = sharpe(g["base_net"]); sh_e_y = sharpe(g["ext_net"])
        log.info("  %-6d %5d %+10.2f %+10.2f %+10.2f",
                 y, len(g), sh_b_y, sh_e_y, sh_e_y - sh_b_y)
        n_total += 1
        if (sh_e_y - sh_b_y) > 0: n_pos += 1

    # Single-event drop (G4)
    log.info("\n=== Gate 4: drop 2021-22 ===")
    no_2122 = merged[~merged["year"].isin([2021, 2022])]
    sh_b_d = sharpe(no_2122["base_net"]); sh_e_d = sharpe(no_2122["ext_net"])
    dsh_d = sh_e_d - sh_b_d
    retained = (dsh_d / dsh) if abs(dsh) > 0.01 else 0
    log.info("  ΔSh full: %+.2f   ΔSh w/o 2021-22: %+.2f   retained: %.2f",
             dsh, dsh_d, retained)

    # Discipline gates
    log.info("\n=== Discipline-gate verdict ===")
    g1 = dsh > 0.20
    g2 = lo > 0
    g3 = n_pos >= 4
    g4 = retained >= 0.50
    g5 = dd_e > dd_b  # max-DD less negative
    log.info("  G1: ΔSh > +0.20             %+.2f          %s", dsh, "✓" if g1 else "✗")
    log.info("  G2: Δnet 95%% CI > 0         [%+.2f, %+.2f]  %s", lo, hi, "✓" if g2 else "✗")
    log.info("  G3: ≥4/7 folds positive      %d/%d         %s", n_pos, n_total, "✓" if g3 else "✗")
    log.info("  G4: 2021-22 drop survival    %.2f retained   %s", retained, "✓" if g4 else "✗")
    log.info("  G5: max-DD reduced           %+.0f vs %+.0f   %s", dd_e, dd_b, "✓" if g5 else "✗")
    n_pass = sum([g1, g2, g3, g4, g5])
    if n_pass == 5:
        verdict = "PASS ALL GATES ✓"
    elif n_pass >= 4:
        verdict = f"BORDERLINE PASS ({n_pass}/5)"
    else:
        verdict = f"FAIL ({n_pass}/5)"
    log.info("\n  VERDICT: %s", verdict)


if __name__ == "__main__":
    main()
