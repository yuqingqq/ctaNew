"""Path B: per-name feature normalization variants for v7.

Tests four normalized variants of the A features (10 features) vs the
baseline raw form. B (PEAD) and F (sector) features unchanged.

Variants:
  baseline        = 18 raw v7 features
  xs_rank_repl    = replace A with cross-sectional rank (per-ts, [0,1])
  ts_pct_repl     = replace A with trailing-252d percentile (per-name)
  xs_rank_add     = baseline + xs-rank A (28 features)
  ts_pct_add      = baseline + ts-pct A (28 features)

Each transform applied to all 10 A features. B/F/sym_id unchanged.

7-fold expanding-train walk-forward. 1 horizon × 3 seeds.

Discipline gates: same five.

Usage:
    python -m ml.research.alpha_v9_xyz_feature_norm
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
USE_SEEDS = SEEDS[:3]  # 3 seeds — confirmation run for xs_rank_add
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


def add_xs_rank(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Cross-sectional rank within each ts, normalized to [0,1]."""
    log.info("  computing xs-rank for %d features ...", len(feats))
    for f in feats:
        panel[f + "_xs_rank"] = panel.groupby("ts")[f].rank(pct=True)
    return panel


def add_ts_pctile(panel: pd.DataFrame, feats: list[str],
                    window: int = 252) -> pd.DataFrame:
    """Per-name trailing-window percentile rank, shifted by 1 (PIT)."""
    log.info("  computing ts-pctile for %d features (window=%d, this may take ~30s) ...",
             len(feats), window)
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    for f in feats:
        panel[f + "_ts_pct"] = (
            panel.groupby("symbol", group_keys=False)[f].apply(
                lambda s: s.rolling(window, min_periods=60).apply(
                    lambda x: (x.iloc[-1] >= x).sum() / len(x), raw=False))
            .shift(1)  # PIT correctness within sym
        )
    return panel


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
    log.info("  base_feats: %d", len(base_feats))

    # TS-percentile only on the 3 vol features (fastest meaningful subset)
    panel = add_xs_rank(panel, feats_A)
    feats_A_xs = [f + "_xs_rank" for f in feats_A]
    vol_feats = ["A_vol_22d", "A_vol_60d", "A_idio_vol_22d"]
    panel = add_ts_pctile(panel, vol_feats)
    feats_vol_ts = [f + "_ts_pct" for f in vol_feats]

    variants = {
        "baseline (raw)": base_feats,
        "xs_rank_add": base_feats + feats_A_xs,
        "ts_pct_add (vol only)": base_feats + feats_vol_ts,
        "xs+ts combo": base_feats + feats_A_xs + feats_vol_ts,
    }
    log.info("  variants: %s",
             ", ".join(f"{k}={len(v)}" for k, v in variants.items()))

    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)

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

    preds_all = {k: [] for k in variants}
    log.info("\n=== 7-fold walk-forward ===")
    for fold in folds:
        train = panel[panel["ts"] < fold["train_end"]]
        test = panel[(panel["ts"] >= fold["test_start"]) &
                       (panel["ts"] < fold["test_end"])].copy()
        if len(train) < 1000 or len(test) < 100: continue
        log.info("  %s: train_n=%d test_n=%d", fold["name"], len(train), len(test))
        for var_name, feats in variants.items():
            models = train_ensemble(train, feats)
            preds = predict_ensemble(models, test, feats)
            test_v = test.copy(); test_v["pred"] = preds
            preds_all[var_name].append(test_v[["ts", "symbol", "pred", "fwd_resid_1d"]])

    # Build portfolios
    log.info("\n=== Building portfolios ===")
    regime = compute_regime_indicators(panel.drop(columns=[
        c for c in panel.columns if c.endswith("_xs_rank") or c.endswith("_ts_pct")
    ], errors="ignore"), anchors)
    pnls = {}
    for var_name in variants:
        preds = pd.concat(preds_all[var_name], ignore_index=True)
        pre = daily_portfolio_hysteresis(preds, "pred", "fwd_resid_1d",
                                            set(TIER_AB), 4, 1, COST_BPS_SIDE)
        pnl = gate_rolling(pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
        pnl["ts"] = pd.to_datetime(pnl["ts"], utc=True)
        pnls[var_name] = pnl

    # Aggregate summary
    log.info("\n=== Aggregated summary (Tier A+B, K=4 M=1, gate, 0.8 bps) ===")
    log.info("  %-22s %5s %10s %14s %14s %12s",
             "config", "n", "Sharpe", "net bps/cyc", "max-DD bps", "ΔSh vs base")
    base_pnl = pnls["baseline (raw)"]
    base_sh = sharpe(base_pnl["net_alpha"])
    for var_name, pnl in pnls.items():
        merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base"}).merge(
            pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "test"}),
            on="ts", how="inner")
        sh_t = sharpe(merged["test"])
        net_t = merged["test"].mean() * 1e4
        dd_t = max_dd(merged["test"]) * 1e4
        log.info("  %-22s %5d %+10.2f %+14.2f %+14.0f %+12.2f",
                 var_name, len(merged), sh_t, net_t, dd_t, sh_t - sharpe(merged["base"]))

    # Paired Δnet CIs
    log.info("\n=== Paired Δnet block-bootstrap (vs baseline) ===")
    for var_name in variants:
        if var_name == "baseline (raw)": continue
        merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base"}).merge(
            pnls[var_name][["ts", "net_alpha"]].rename(columns={"net_alpha": "test"}),
            on="ts", how="inner")
        diff = (merged["test"] - merged["base"]).to_numpy() * 1e4
        d, lo, hi = paired_block_bs(diff)
        log.info("  %-22s  Δnet=%+7.2f bps  CI=[%+5.2f, %+5.2f]",
                 var_name, d, lo, hi)

    # Per-year ΔSh
    log.info("\n=== Per-year ΔSh ===")
    yrs_data = {}
    for var_name, pnl in pnls.items():
        merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base"}).merge(
            pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": var_name}),
            on="ts", how="inner")
        merged["year"] = merged["ts"].dt.year
        yrs_data[var_name] = merged
    base_merged = yrs_data["baseline (raw)"]
    var_keys = [k for k in variants if k != "baseline (raw)"]
    header = f"  {'year':>6} {'n':>5} {'base Sh':>10}  " + "  ".join(f"{k[:14]:>14}" for k in var_keys)
    log.info(header)
    for y, g_base in base_merged.groupby("year"):
        if len(g_base) < 5: continue
        sh_b = sharpe(g_base["base"])
        deltas = []
        for var in var_keys:
            g = yrs_data[var][yrs_data[var]["year"] == y]
            sh_v = sharpe(g[var])
            deltas.append(sh_v - sh_b)
        delta_str = "  ".join(f"{d:+14.2f}" for d in deltas)
        log.info(f"  {y:>6} {len(g_base):>5} {sh_b:+10.2f}  {delta_str}")

    # Discipline-gate verdict
    log.info("\n=== Discipline-gate verdict ===")
    log.info("  %-22s %8s %12s %18s %8s %s",
             "variant", "ΔSh", "folds +", "Δnet 95% CI", "Q21-22", "verdict")
    for var_name in variants:
        if var_name == "baseline (raw)": continue
        merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base"}).merge(
            pnls[var_name][["ts", "net_alpha"]].rename(columns={"net_alpha": "test"}),
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
        verdict = "PASS" if n_pass == 5 else (
            f"borderline {n_pass}/5" if n_pass >= 3 else f"fail {n_pass}/5")
        log.info("  %-22s %+8.2f %5d/%d  [%+5.2f, %+5.2f] %.2f  %s",
                 var_name, dsh, n_pos, n_total, lo, hi, retained, verdict)


if __name__ == "__main__":
    main()
