"""Multi-horizon ensemble for crypto v6_clean.

Hypothesis (from xyz work): training models on different forward horizons
and averaging predictions reduces single-model noise and lifts Sharpe.
xyz result: 3-horizon ensemble lifted Sharpe by ~+0.30 over single-horizon.

Crypto v6_clean test:
  - Universe: ORIG25 (don't change — exhaustively validated)
  - Features: v6_clean 28-col set (don't change)
  - Horizons: h=48, h=96, h=144, h=288 (4 different forward windows)
  - Each horizon: 5-seed LGBM ensemble = 20 models total
  - Average predictions per (sym, ts) across all 20
  - Portfolio: top-K=7, h=48 cadence, turnover-aware cost
  - Compare to single-horizon h=48 K=7 baseline (Sharpe +3.63)

Note: predictions across horizons have different scales (h=288 returns
are ~2.5× std of h=48 returns). We z-score predictions per-bar across
symbols before averaging to balance horizon contributions.
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    assemble_universe, build_basket, build_kline_features, list_universe,
    make_xs_alpha_labels,
)
import gc
from ml.research.alpha_v4_xs import (
    _train, _stack_xs_panel, _walk_forward_splits, _slice,
    portfolio_pnl_turnover_aware, block_bootstrap_ci,
    NAKED_COST_BPS_PER_LEG,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENSEMBLE_SEEDS = (42, 7, 123, 99, 314)
HORIZONS = (48, 96, 144, 288)   # multi-horizon set
PNL_HORIZON = 48                # use h=48 cadence for portfolio P&L
TOP_FRAC_K7 = 7 / 25            # 7 long, 7 short out of 25
REGIME_CUTOFF = 0.50
COST_BPS_PER_LEG = 4.5          # HL VIP-0 taker (production reality)
N_FOLDS = 9                     # multi-OOS-style (was 5 in basic walk-forward)


def build_panel_with_multi_horizon_labels(symbols: list[str]) -> pd.DataFrame:
    """Mirror `_build_full_panel` from train_v6_clean_artifact, but compute
    labels at multiple forward horizons and store as separate columns."""
    log.info("loading per-symbol kline features for %d symbols...", len(symbols))
    feats_by_sym = {}
    for s in symbols:
        f = build_kline_features(s)
        if not f.empty:
            feats_by_sym[s] = f
    closes = pd.DataFrame({s: f["close"] for s, f in feats_by_sym.items()}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(sorted(feats_by_sym.keys()))}

    log.info("enriching with basket features...")
    enriched = {}
    for s, f in feats_by_sym.items():
        f = f.reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        enriched[s] = f

    # Build labels at multiple horizons
    log.info("computing labels at horizons %s...", HORIZONS)
    labels_per_horizon = {h: make_xs_alpha_labels(enriched, basket_close, h)
                           for h in HORIZONS}

    # Stack panel with all needed columns + multi-horizon labels
    rank_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                       + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                       + src_cols) - set(rank_cols))

    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df_base = f[avail].copy()
        # Join labels from each horizon
        for h in HORIZONS:
            lab_h = labels_per_horizon[h][s]
            df_base = df_base.join(lab_h.rename(columns={
                "demeaned_target": f"demeaned_target_h{h}",
                "return_pct": f"return_pct_h{h}",
                "basket_fwd": f"basket_fwd_h{h}",
                "alpha_realized": f"alpha_realized_h{h}",
                "exit_time": f"exit_time_h{h}",
            }), how="inner")
        df_base["symbol"] = s
        df_base = df_base.reset_index().rename(columns={"index": "open_time"})
        for c in df_base.select_dtypes("float64").columns:
            df_base[c] = df_base[c].astype("float32")
        frames.append(df_base)
    del enriched, feats_by_sym, labels_per_horizon
    gc.collect()
    panel = pd.concat(frames, ignore_index=True, sort=False)
    del frames
    gc.collect()

    log.info("adding xs_rank features...")
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    panel[rank_cols] = panel[rank_cols].astype("float32")
    panel = panel.dropna(subset=rank_cols + ["autocorr_pctile_7d"])

    # Set primary P&L horizon columns
    panel["exit_time"] = panel[f"exit_time_h{PNL_HORIZON}"]
    panel["return_pct"] = panel[f"return_pct_h{PNL_HORIZON}"]
    panel["alpha_realized"] = panel[f"alpha_realized_h{PNL_HORIZON}"]
    panel["basket_fwd"] = panel[f"basket_fwd_h{PNL_HORIZON}"]

    # Drop rows missing any horizon's target
    target_cols = [f"demeaned_target_h{h}" for h in HORIZONS]
    panel = panel.dropna(subset=target_cols + XS_FEATURE_COLS_V6_CLEAN)
    log.info("panel: %d rows × %d cols, %d unique syms, %d horizon labels",
             len(panel), panel.shape[1],
             panel["symbol"].nunique(), len(HORIZONS))
    return panel


def train_horizon_ensemble(train: pd.DataFrame, cal: pd.DataFrame,
                            features: list[str], target_col: str,
                            seeds: tuple = ENSEMBLE_SEEDS) -> list:
    """Train K-seed ensemble on a given target column. Returns list of models."""
    train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    if len(train_f) < 1000 or len(cal_f) < 100:
        return []
    Xt = train_f[features].values.astype(np.float32)
    yt = train_f[target_col].values.astype(np.float32)
    Xc = cal_f[features].values.astype(np.float32)
    yc = cal_f[target_col].values.astype(np.float32)
    models = []
    for seed in seeds:
        m = _train(Xt, yt, Xc, yc, seed=seed)
        models.append(m)
    return models


def predict_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    """Average predictions across model list."""
    if not models:
        return np.zeros(len(X))
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0)


def zscore_per_bar(panel: pd.DataFrame, pred_col: str) -> pd.Series:
    """Z-score predictions across symbols within each open_time bar."""
    def _z(s: pd.Series) -> pd.Series:
        std = s.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / max(std, 1e-8)
    return panel.groupby("open_time")[pred_col].transform(_z)


def main() -> None:
    symbols = list_universe(min_days=200)
    if os.environ.get("UNIVERSE", "ORIG25") == "ORIG25":
        from live.train_v6_clean_artifact import NEW_SYMBOLS
        symbols = [s for s in symbols if s not in NEW_SYMBOLS]
    log.info("universe: %d symbols", len(symbols))

    panel = build_panel_with_multi_horizon_labels(symbols)
    folds = _walk_forward_splits(panel, n_folds=N_FOLDS,
                                  train_days=120, cal_days=20,
                                  test_days=30, embargo_days=2)
    log.info("walk-forward folds: %d", len(folds))

    cycle_records = {"single_h48": [], "multi_zscore": [], "multi_raw": []}

    for fold in folds:
        train, cal, test = _slice(panel, fold)
        if len(train) < 1000 or len(test) < 100:
            log.warning("fold %d: skip (train=%d test=%d)",
                        fold["fid"], len(train), len(test))
            continue
        log.info("fold %d: train=%d cal=%d test=%d (%s → %s)",
                 fold["fid"], len(train), len(cal), len(test),
                 fold["test_start"].date(), fold["test_end"].date())

        # Train ensembles for each horizon
        all_preds_by_horizon = {}
        for h in HORIZONS:
            models = train_horizon_ensemble(
                train, cal, XS_FEATURE_COLS_V6_CLEAN,
                target_col=f"demeaned_target_h{h}"
            )
            if not models:
                log.warning("  h=%d: training failed", h)
                continue
            X_test = test[XS_FEATURE_COLS_V6_CLEAN].values.astype(np.float32)
            preds = predict_ensemble(models, X_test)
            all_preds_by_horizon[h] = preds
            log.info("  h=%d: trained %d seed models, avg pred std=%.4f",
                     h, len(models), preds.std())

        if PNL_HORIZON not in all_preds_by_horizon:
            continue

        # Variant 1: single-horizon h=48 baseline
        test_v1 = test.copy()
        test_v1["pred"] = all_preds_by_horizon[PNL_HORIZON]
        out_v1 = portfolio_pnl_turnover_aware(
            test_v1, all_preds_by_horizon[PNL_HORIZON],
            top_frac=TOP_FRAC_K7, cost_bps_per_leg=COST_BPS_PER_LEG,
            sample_every=PNL_HORIZON, beta_neutral=False,
        )
        if out_v1.get("n_bars", 0) > 0:
            for _, row in out_v1["df"].iterrows():
                cycle_records["single_h48"].append({
                    "fold": fold["fid"], "spread": row["spread_ret_bps"],
                    "alpha": row["spread_alpha_bps"], "net": row["net_bps"],
                })

        # Variant 2: multi-horizon ensemble with z-score averaging
        test_v2 = test.copy()
        z_scores = []
        for h in HORIZONS:
            if h not in all_preds_by_horizon:
                continue
            test_v2[f"pred_h{h}"] = all_preds_by_horizon[h]
            z = zscore_per_bar(test_v2, f"pred_h{h}").fillna(0).values
            z_scores.append(z)
        if z_scores:
            ens_z = np.mean(z_scores, axis=0)
            out_v2 = portfolio_pnl_turnover_aware(
                test_v2, ens_z, top_frac=TOP_FRAC_K7,
                cost_bps_per_leg=COST_BPS_PER_LEG,
                sample_every=PNL_HORIZON, beta_neutral=False,
            )
            if out_v2.get("n_bars", 0) > 0:
                for _, row in out_v2["df"].iterrows():
                    cycle_records["multi_zscore"].append({
                        "fold": fold["fid"], "spread": row["spread_ret_bps"],
                        "alpha": row["spread_alpha_bps"], "net": row["net_bps"],
                    })

        # Variant 3: multi-horizon ensemble with raw averaging
        test_v3 = test.copy()
        raw_preds = []
        for h in HORIZONS:
            if h not in all_preds_by_horizon:
                continue
            raw_preds.append(all_preds_by_horizon[h])
        if raw_preds:
            ens_raw = np.mean(raw_preds, axis=0)
            out_v3 = portfolio_pnl_turnover_aware(
                test_v3, ens_raw, top_frac=TOP_FRAC_K7,
                cost_bps_per_leg=COST_BPS_PER_LEG,
                sample_every=PNL_HORIZON, beta_neutral=False,
            )
            if out_v3.get("n_bars", 0) > 0:
                for _, row in out_v3["df"].iterrows():
                    cycle_records["multi_raw"].append({
                        "fold": fold["fid"], "spread": row["spread_ret_bps"],
                        "alpha": row["spread_alpha_bps"], "net": row["net_bps"],
                    })

    # Summarize
    log.info("\n=== AGGREGATE RESULTS (turnover-aware, %d-fold WF, h=%d cadence) ===",
             N_FOLDS, PNL_HORIZON)
    log.info("  %-30s %5s %12s %12s %12s %12s %16s",
             "config", "n_cyc", "spread/cyc", "alpha/cyc", "net/cyc",
             "Sharpe", "95% CI")
    cycle_per_year = 365 * 24 / (PNL_HORIZON * 5 / 60)  # 5min bars to cycles/year

    def _summarize(records: list, name: str) -> None:
        if not records:
            log.info("  %-30s NO DATA", name); return
        df = pd.DataFrame(records)
        spreads = df["spread"].values  # already in bps
        alphas = df["alpha"].values
        nets = df["net"].values
        sharpe_estimator = lambda x: x.mean() / x.std() * np.sqrt(cycle_per_year) \
                                       if x.std() > 0 else 0
        sh, lo, hi = block_bootstrap_ci(nets, statistic=sharpe_estimator,
                                         block_size=7, n_boot=2000)
        log.info("  %-30s %5d %+10.2fbps %+10.2fbps %+10.2fbps %+10.2f  [%+.2f, %+.2f]",
                 name, len(df), spreads.mean(), alphas.mean(),
                 nets.mean(), sh, lo, hi)

    _summarize(cycle_records["single_h48"], "single h=48 (baseline)")
    _summarize(cycle_records["multi_zscore"], "multi-h zscore-avg")
    _summarize(cycle_records["multi_raw"], "multi-h raw-avg")


if __name__ == "__main__":
    main()
