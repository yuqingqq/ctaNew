"""Validation: ABF_xs_add at full v7 ensemble (3 horizons × 5 seeds).

Earlier 3-seed × 1-horizon test gave ΔSh +0.49 with wide CI. This runs
at the full v7 ensemble configuration to validate the point estimate.

Variants:
  baseline (raw 18 feat)
  ABF_xs_add (35 feat = 18 raw + 17 xs-rank versions of A+B+F)

7-fold walk-forward, 3 horizons × 5 seeds = 15 models per ensemble per fold.

Discipline gates: same five.

Usage:
    python -m ml.research.alpha_v9_xyz_ABF_validate
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
HORIZONS = (3, 5, 10)  # full v7 spec
USE_SEEDS = SEEDS  # all 5 seeds
GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8


def sharpe(s, rpy=252.0):
    if len(s) < 2 or s.std() == 0: return 0.0
    return s.mean() / s.std() * np.sqrt(rpy)


def max_dd(s):
    cum = s.cumsum(); peak = cum.cummax()
    return float((cum - peak).min())


def paired_block_bs(diff, block=5, n_boot=2000):
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


def add_xs_rank(panel, feats):
    for f in feats:
        panel[f + "_xs_rank"] = panel.groupby("ts")[f].rank(pct=True)
    return panel


def train_ensemble(train, feats):
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


def predict_ensemble(models, df, feats):
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


def main():
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
    log.info("  base: %d  A=%d B=%d F=%d", len(base_feats), len(feats_A), len(feats_B), len(feats_F))

    panel = add_xs_rank(panel, feats_A + feats_B + feats_F)
    feats_A_xs = [f + "_xs_rank" for f in feats_A]
    feats_B_xs = [f + "_xs_rank" for f in feats_B]
    feats_F_xs = [f + "_xs_rank" for f in feats_F]
    abf_feats = base_feats + feats_A_xs + feats_B_xs + feats_F_xs

    log.info("  baseline=%d  ABF_xs_add=%d", len(base_feats), len(abf_feats))
    log.info("  ensemble: %d horizons × %d seeds = %d models per fold",
             len(HORIZONS), len(USE_SEEDS), len(HORIZONS) * len(USE_SEEDS))

    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)

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

    base_preds_all = []; abf_preds_all = []
    log.info("\n=== 7-fold walk-forward (full v7 ensemble) ===")
    for fold in folds:
        train = panel[panel["ts"] < fold["train_end"]]
        test = panel[(panel["ts"] >= fold["test_start"]) &
                       (panel["ts"] < fold["test_end"])].copy()
        if len(train) < 1000 or len(test) < 100: continue
        log.info("  %s: train_n=%d test_n=%d", fold["name"], len(train), len(test))

        log.info("    training baseline (18 feat × 15 models) ...")
        base_models = train_ensemble(train, base_feats)
        base_pred = predict_ensemble(base_models, test, base_feats)

        log.info("    training ABF_xs_add (35 feat × 15 models) ...")
        abf_models = train_ensemble(train, abf_feats)
        abf_pred = predict_ensemble(abf_models, test, abf_feats)

        # ABF feature importance check
        if abf_models:
            avg_imp = np.zeros(len(abf_feats))
            for k, m in abf_models.items():
                avg_imp += m.feature_importances_.astype(float)
            avg_imp /= len(abf_models)
            xs_idx = [abf_feats.index(c) for c in abf_feats if c.endswith("_xs_rank")]
            xs_imp_pct = avg_imp[xs_idx].sum() / avg_imp.sum() * 100
            log.info("    xs-rank features total importance: %.1f%% of model gain",
                     xs_imp_pct)

        for arr, store in [(base_pred, base_preds_all), (abf_pred, abf_preds_all)]:
            test_x = test.copy(); test_x["pred"] = arr
            store.append(test_x[["ts", "symbol", "pred", "fwd_resid_1d"]])

    base_preds = pd.concat(base_preds_all, ignore_index=True)
    abf_preds = pd.concat(abf_preds_all, ignore_index=True)

    log.info("\n=== Building portfolios ===")
    regime = compute_regime_indicators(panel.drop(columns=[
        c for c in panel.columns if c.endswith("_xs_rank")
    ], errors="ignore"), anchors)
    base_pre = daily_portfolio_hysteresis(base_preds, "pred", "fwd_resid_1d",
                                            set(TIER_AB), 4, 1, COST_BPS_SIDE)
    base_pnl = gate_rolling(base_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    abf_pre = daily_portfolio_hysteresis(abf_preds, "pred", "fwd_resid_1d",
                                            set(TIER_AB), 4, 1, COST_BPS_SIDE)
    abf_pnl = gate_rolling(abf_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    base_pnl["ts"] = pd.to_datetime(base_pnl["ts"], utc=True)
    abf_pnl["ts"] = pd.to_datetime(abf_pnl["ts"], utc=True)

    merged = base_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "base"}).merge(
        abf_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "abf"}),
        on="ts", how="inner")
    merged["year"] = merged["ts"].dt.year
    log.info("  shared ts: %d", len(merged))

    # Aggregate
    log.info("\n=== Aggregated 7-fold OOS (full ensemble validation) ===")
    sh_b = sharpe(merged["base"]); sh_a = sharpe(merged["abf"])
    dsh = sh_a - sh_b
    diff = (merged["abf"] - merged["base"]).to_numpy() * 1e4
    d, lo, hi = paired_block_bs(diff)
    dd_b = max_dd(merged["base"]) * 1e4
    dd_a = max_dd(merged["abf"]) * 1e4
    log.info("  baseline (18 raw):    Sh=%+.2f  net=%+.2f bps  max-DD=%+.0f bps",
             sh_b, merged["base"].mean() * 1e4, dd_b)
    log.info("  ABF_xs_add (35 feat): Sh=%+.2f  net=%+.2f bps  max-DD=%+.0f bps",
             sh_a, merged["abf"].mean() * 1e4, dd_a)
    log.info("  ΔSh = %+.2f   Δnet = %+.2f bps   95%% CI = [%+.2f, %+.2f] bps",
             dsh, d, lo, hi)

    # Per-year
    log.info("\n=== Per-year ΔSh ===")
    log.info("  %-6s %5s %10s %10s %10s",
             "year", "n", "base Sh", "abf Sh", "ΔSh")
    n_pos = 0; n_total = 0
    for y, g in merged.groupby("year"):
        if len(g) < 5: continue
        sh_b_y = sharpe(g["base"]); sh_a_y = sharpe(g["abf"])
        log.info("  %-6d %5d %+10.2f %+10.2f %+10.2f",
                 y, len(g), sh_b_y, sh_a_y, sh_a_y - sh_b_y)
        n_total += 1
        if (sh_a_y - sh_b_y) > 0: n_pos += 1

    # Single-event drop
    log.info("\n=== Gate 4: drop 2021-22 ===")
    no_2122 = merged[~merged["year"].isin([2021, 2022])]
    sh_b_d = sharpe(no_2122["base"]); sh_a_d = sharpe(no_2122["abf"])
    dsh_d = sh_a_d - sh_b_d
    retained = (dsh_d / dsh) if abs(dsh) > 0.01 else 0
    log.info("  ΔSh full: %+.2f   ΔSh w/o 2021-22: %+.2f   retained: %.2f",
             dsh, dsh_d, retained)

    log.info("\n=== Discipline-gate verdict ===")
    g1 = dsh > 0.20
    g2 = lo > 0
    g3 = n_pos / n_total >= 0.5
    g4 = retained >= 0.5
    g5 = dd_a > dd_b
    n_pass = sum([g1, g2, g3, g4, g5])
    log.info("  G1 ΔSh > +0.20             %+.2f          %s", dsh, "✓" if g1 else "✗")
    log.info("  G2 Δnet 95%% CI > 0         [%+.2f, %+.2f]  %s", lo, hi, "✓" if g2 else "✗")
    log.info("  G3 ≥4/7 folds positive      %d/%d         %s", n_pos, n_total, "✓" if g3 else "✗")
    log.info("  G4 2021-22 drop survival    %.2f retained   %s", retained, "✓" if g4 else "✗")
    log.info("  G5 max-DD reduced           %+.0f vs %+.0f   %s", dd_a, dd_b, "✓" if g5 else "✗")
    if n_pass == 5:
        verdict = "PASS ALL GATES ✓"
    elif n_pass >= 4:
        verdict = f"BORDERLINE PASS ({n_pass}/5)"
    elif n_pass >= 3:
        verdict = f"MARGINAL ({n_pass}/5)"
    else:
        verdict = f"FAIL ({n_pass}/5)"
    log.info("\n  VERDICT: %s", verdict)


if __name__ == "__main__":
    main()
