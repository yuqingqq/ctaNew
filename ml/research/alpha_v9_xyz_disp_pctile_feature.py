"""Path A': add disp_22d as ROLLING PERCENTILE RANK (regime-stationary).

Path A failed with raw disp_22d (drifting distribution: 67th-pctile rose
+16% over 6y → train-test mismatch).

Path A' uses trailing-252d percentile rank instead — bounded [0,1],
regime-stationary, same shape as the gate's own threshold logic.

Three variants compared, all on 7-fold walk-forward:
  baseline      = v7 18 features
  Path A'-1     = baseline + disp_22d_pctile (19 features) [absolute REPLACED]
  Path A'-2     = baseline + disp_22d_pctile + disp_22d (20 features) [both]

Also tracks LGBM feature importance to confirm the model uses the new feature.

Usage:
    python -m ml.research.alpha_v9_xyz_disp_pctile_feature
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
from ml.research.alpha_v7_daily_optimized import daily_portfolio_hysteresis
from ml.research.alpha_v7_tier_a import TIER_AB

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
HORIZONS = (5,)
USE_SEEDS = SEEDS[:3]  # 3 seeds for tighter CI; compute is cheap
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


def feat_importance_pct(models: dict, feats: list[str], target: str) -> float:
    """Average importance % for `target` across ensemble."""
    if not models or target not in feats: return 0.0
    idx = feats.index(target)
    avg = np.zeros(len(feats))
    for k, m in models.items():
        avg += m.feature_importances_.astype(float)
    return avg[idx] / avg.sum() * 100


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

    # Compute regime + dispersion percentile (PIT-correct: trailing 252d, shift 1)
    regime = compute_regime_indicators(panel, anchors)
    regime["ts"] = pd.to_datetime(regime["ts"], utc=True)
    regime = regime.sort_values("ts").reset_index(drop=True)

    log.info("  computing disp_22d_pctile_252d (rolling pctile rank, shifted 1)...")
    # Vectorized rolling rank: where does today's value sit in past 252 days?
    def rolling_pctile_rank(s, window=252, min_periods=60):
        ranks = s.rolling(window, min_periods=min_periods).apply(
            lambda x: (x.iloc[-1] >= x).sum() / len(x), raw=False)
        return ranks.shift(1)

    regime["disp_22d_pctile"] = rolling_pctile_rank(regime["disp_22d"])
    log.info("  pctile non-null: %d / %d  range: [%.3f, %.3f]",
             regime["disp_22d_pctile"].notna().sum(), len(regime),
             regime["disp_22d_pctile"].min(), regime["disp_22d_pctile"].max())

    # Merge to panel
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel = panel.merge(regime[["ts", "disp_22d", "disp_22d_pctile"]],
                          on="ts", how="left")

    feats_v1 = base_feats + ["disp_22d_pctile"]
    feats_v2 = base_feats + ["disp_22d_pctile", "disp_22d"]
    log.info("  base: %d  v1 (replace abs→pct): %d  v2 (both): %d",
             len(base_feats), len(feats_v1), len(feats_v2))

    # 7-fold expanding-train walk-forward
    folds = []
    for test_year in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
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
    v1_preds_all = []
    v2_preds_all = []
    log.info("\n=== 7-fold walk-forward (baseline 18 vs v1 19 [pct] vs v2 20 [both]) ===")
    log.info("  %-10s %8s %8s %8s %8s",
             "fold", "imp pct", "v2 imp pct", "v2 imp abs", "test_n")
    for fold in folds:
        train = panel[panel["ts"] < fold["train_end"]]
        test = panel[(panel["ts"] >= fold["test_start"]) &
                       (panel["ts"] < fold["test_end"])].copy()
        if len(train) < 1000 or len(test) < 100: continue

        base_models = train_ensemble(train, base_feats)
        v1_models = train_ensemble(train, feats_v1)
        v2_models = train_ensemble(train, feats_v2)

        base_pred = predict_ensemble(base_models, test, base_feats)
        v1_pred = predict_ensemble(v1_models, test, feats_v1)
        v2_pred = predict_ensemble(v2_models, test, feats_v2)

        # Importance diagnostics
        v1_imp = feat_importance_pct(v1_models, feats_v1, "disp_22d_pctile")
        v2_imp_pct = feat_importance_pct(v2_models, feats_v2, "disp_22d_pctile")
        v2_imp_abs = feat_importance_pct(v2_models, feats_v2, "disp_22d")
        log.info("  %-10s %7.2f%% %7.2f%% %7.2f%% %8d",
                 fold["name"], v1_imp, v2_imp_pct, v2_imp_abs, len(test))

        for arr, store in [(base_pred, base_preds_all),
                              (v1_pred, v1_preds_all),
                              (v2_pred, v2_preds_all)]:
            test_x = test.copy(); test_x["pred"] = arr
            store.append(test_x[["ts", "symbol", "pred", "fwd_resid_1d"]])

    base_preds = pd.concat(base_preds_all, ignore_index=True)
    v1_preds = pd.concat(v1_preds_all, ignore_index=True)
    v2_preds = pd.concat(v2_preds_all, ignore_index=True)

    # Build portfolios
    log.info("\n=== Building portfolios ===")
    pnls = {}
    for label, preds in [("baseline (18)", base_preds),
                            ("v1: pct-only (19)", v1_preds),
                            ("v2: both (20)", v2_preds)]:
        pre = daily_portfolio_hysteresis(preds, "pred", "fwd_resid_1d",
                                            set(TIER_AB), 4, 1, COST_BPS_SIDE)
        pnl = gate_rolling(pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
        pnls[label] = pnl

    # Aggregate
    log.info("\n=== Aggregated test panel summary ===")
    base_pnl = pnls["baseline (18)"]
    base_pnl["ts"] = pd.to_datetime(base_pnl["ts"], utc=True)
    log.info("  %-22s %5s %10s %14s %14s %14s",
             "config", "n", "Sharpe", "net bps/cyc", "max-DD bps", "ΔSh vs base")
    base_sh = sharpe(base_pnl["net_alpha"])
    for label, pnl in pnls.items():
        pnl["ts"] = pd.to_datetime(pnl["ts"], utc=True)
        # Merge on shared ts
        merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base"}).merge(
            pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "test"}),
            on="ts", how="inner")
        sh_t = sharpe(merged["test"])
        net_t = merged["test"].mean() * 1e4
        dd_t = max_dd(merged["test"]) * 1e4
        log.info("  %-22s %5d %+10.2f %+14.2f %+14.0f %+14.2f",
                 label, len(merged), sh_t, net_t, dd_t, sh_t - sharpe(merged["base"]))

    # Paired Δnet CI for v1 and v2
    log.info("\n=== Paired Δnet bps vs baseline (block-bootstrap) ===")
    for label in ["v1: pct-only (19)", "v2: both (20)"]:
        merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base"}).merge(
            pnls[label][["ts", "net_alpha"]].rename(columns={"net_alpha": "test"}),
            on="ts", how="inner")
        diff = (merged["test"] - merged["base"]).to_numpy() * 1e4
        d, lo, hi = paired_block_bs(diff)
        log.info("  %-22s  Δnet=%+7.2f bps  CI=[%+5.2f, %+5.2f]", label, d, lo, hi)

    # Per-year breakdown
    log.info("\n=== Per-year ΔSh ===")
    log.info("  %-6s %5s %10s %12s %12s",
             "year", "n", "base Sh", "v1 ΔSh", "v2 ΔSh")
    for label in ["v1: pct-only (19)", "v2: both (20)"]:
        if label == "v1: pct-only (19)":
            merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base"}).merge(
                pnls["v1: pct-only (19)"][["ts", "net_alpha"]].rename(columns={"net_alpha": "v1"}),
                on="ts").merge(
                pnls["v2: both (20)"][["ts", "net_alpha"]].rename(columns={"net_alpha": "v2"}),
                on="ts")
            merged["year"] = merged["ts"].dt.year
            for y, g in merged.groupby("year"):
                if len(g) < 5: continue
                sh_b = sharpe(g["base"]); sh_v1 = sharpe(g["v1"]); sh_v2 = sharpe(g["v2"])
                log.info("  %-6d %5d %+10.2f %+12.2f %+12.2f",
                         y, len(g), sh_b, sh_v1 - sh_b, sh_v2 - sh_b)
            break

    # Discipline-gate verdict
    log.info("\n=== Discipline-gate verdict ===")
    for label in ["v1: pct-only (19)", "v2: both (20)"]:
        merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base"}).merge(
            pnls[label][["ts", "net_alpha"]].rename(columns={"net_alpha": "test"}),
            on="ts", how="inner")
        merged["year"] = merged["ts"].dt.year
        sh_b = sharpe(merged["base"]); sh_t = sharpe(merged["test"])
        dsh = sh_t - sh_b
        diff = (merged["test"] - merged["base"]).to_numpy() * 1e4
        d, lo, hi = paired_block_bs(diff)
        n_pos = sum(1 for y, g in merged.groupby("year")
                       if len(g) >= 5 and sharpe(g["test"]) > sharpe(g["base"]))
        n_total = sum(1 for y, g in merged.groupby("year") if len(g) >= 5)
        no_2122 = merged[~merged["year"].isin([2021, 2022])]
        sh_b_d = sharpe(no_2122["base"]); sh_t_d = sharpe(no_2122["test"])
        retained = ((sh_t_d - sh_b_d) / dsh) if abs(dsh) > 0.01 else 0
        dd_b = max_dd(merged["base"]); dd_t = max_dd(merged["test"])
        g1 = dsh > 0.20
        g2 = lo > 0
        g3 = n_pos / n_total >= 0.5
        g4 = retained >= 0.5
        g5 = dd_t > dd_b
        n_pass = sum([g1, g2, g3, g4, g5])
        log.info("  %-22s ΔSh=%+.2f folds=%d/%d retain=%.2f g1=%s g2=%s g3=%s g4=%s g5=%s  → %d/5",
                 label, dsh, n_pos, n_total, retained,
                 "✓" if g1 else "✗", "✓" if g2 else "✗",
                 "✓" if g3 else "✗", "✓" if g4 else "✗", "✓" if g5 else "✗",
                 n_pass)


if __name__ == "__main__":
    main()
