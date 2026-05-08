"""Extend the (fixed PEAD) xyz strategy with pre/post-market features
computed from cached Polygon 5m data.

Pre/post-market features (12 names with 2y of cached Polygon data):
  E_after_hours_ret    : log(close[20:00 ET] / close[16:00 ET])  per day
  E_pre_market_gap     : log(open[09:30 ET] / prev_close[16:00 ET]) per day
  E_after_hours_vol_z  : z-score of after-hours volume vs trailing 22d
  E_pre_market_vol_z   : z-score of pre-market volume vs trailing 22d
  E_event_after_hours  : after_hours_ret on the announcement-day timestamp
                          (only populated on actual earnings days)
  E_overnight_total    : prev_close → next_open total move (= AH + PM)

These features are only available for the 12 names with Polygon 5m cache.
For other names (the 88 S&P 100 we use in training), they're NaN. LGBM
handles missing values; the features will only matter for the 12 names.

Coverage: ~2 years (Jun 2024 → May 2026). Earlier folds will have all-NaN
extended-hours features for all names — the LGBM essentially ignores them
in those folds. They contribute meaningfully only in the latest folds.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe
from data_collectors.polygon_loader import fetch_aggs
from ml.research.alpha_v7_multi import (
    add_returns_and_basket, add_features_A,
)
from ml.research.alpha_v7_weekly import (
    metrics_weekly, bootstrap_ci_weekly, make_folds,
)
import lightgbm as lgb
from ml.research.alpha_v7_multi import LGB_PARAMS, SEEDS

def fit_predict_nan_tolerant(train: pd.DataFrame, test: pd.DataFrame,
                              features: list[str], label: str,
                              core_features: list[str] | None = None) -> pd.DataFrame:
    """LGBM handles NaN inputs natively. Drop only on the label; let
    LGBM learn to split on missing values for optional features. core_features
    (if provided) must be non-NaN to be in train/test."""
    drop_on = ([label] if not core_features else core_features + [label])
    train_ = train.dropna(subset=drop_on)
    if len(train_) < 1000:
        return pd.DataFrame()
    sub = test.dropna(subset=core_features) if core_features else test.copy()
    if sub.empty:
        return pd.DataFrame()
    preds = []
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(train_[features], train_[label])
        preds.append(m.predict(sub[features]))
    sub = sub.copy()
    sub["pred"] = np.mean(preds, axis=0)
    return sub
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz import construct_portfolio_subset, gate_by_dispersion
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100
from ml.research.alpha_v7_freq_sweep import add_residual_and_label, metrics_freq, annualized_unconditional
from ml.research.alpha_v7_pead_fixed import (
    add_features_B_fixed, _to_effective_event_date, run_strategy as run_baseline,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

POLY_NAMES = ["NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META",
              "AAPL", "MSFT", "ORCL", "INTC", "MU", "NFLX"]
HOLD_DAYS = 3
COST_BPS_SIDE = 2.5
TOP_K = 5


# ---- compute extended-hours features per symbol from Polygon 5m -------

def compute_eh_features_for_symbol(sym: str) -> pd.DataFrame:
    """For one symbol, compute daily extended-hours features. Returns a
    DataFrame with [ts (midnight UTC), symbol, E_after_hours_ret,
    E_pre_market_gap, E_after_hours_vol_z, E_pre_market_vol_z,
    E_overnight_total]."""
    df = fetch_aggs(sym, "5m")
    if df.empty:
        return pd.DataFrame()
    df["ts_et"] = df["ts"].dt.tz_convert("America/New_York")
    df["et_date"] = df["ts_et"].dt.date
    df["et_hour"] = df["ts_et"].dt.hour
    df["et_minute"] = df["ts_et"].dt.minute
    df["session"] = pd.cut(
        df["et_hour"] + df["et_minute"] / 60.0,
        bins=[0, 9.5, 16, 20, 24],
        labels=["pre", "rth", "post", "late_night"],
        include_lowest=True,
    )

    rows = []
    grouped = df.groupby("et_date")
    prev_rth_close = None
    prev_post_close = None
    pre_vols = []
    ah_vols = []
    for date, day_df in grouped:
        rth = day_df[day_df["session"] == "rth"]
        pre = day_df[day_df["session"] == "pre"]
        post = day_df[day_df["session"] == "post"]
        rth_open = rth["open"].iloc[0] if not rth.empty else np.nan
        rth_close = rth["close"].iloc[-1] if not rth.empty else np.nan
        post_close = post["close"].iloc[-1] if not post.empty else np.nan
        # Calculations
        ah_ret = np.log(post_close / rth_close) if (
            not np.isnan(rth_close) and not np.isnan(post_close)) else np.nan
        if prev_post_close is not None and not np.isnan(prev_post_close) and not np.isnan(rth_open):
            pm_gap = np.log(rth_open / prev_post_close)
        elif prev_rth_close is not None and not np.isnan(prev_rth_close) and not np.isnan(rth_open):
            # fallback: if no prev post, use prev rth close
            pm_gap = np.log(rth_open / prev_rth_close)
        else:
            pm_gap = np.nan
        ah_vol = post["volume"].sum() if not post.empty else 0
        pm_vol = pre["volume"].sum() if not pre.empty else 0
        rows.append({
            "et_date": date,
            "rth_close": rth_close,
            "rth_open": rth_open,
            "post_close": post_close,
            "E_after_hours_ret": ah_ret,
            "E_pre_market_gap": pm_gap,
            "ah_vol": ah_vol,
            "pm_vol": pm_vol,
        })
        # Update prev close for next iteration
        if not np.isnan(rth_close):
            prev_rth_close = rth_close
        if not np.isnan(post_close):
            prev_post_close = post_close

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["E_overnight_total"] = out["E_after_hours_ret"].fillna(0) + out["E_pre_market_gap"].fillna(0)
    # Volume z-scores (rolling 22d)
    out["ah_vol_mean_22d"] = out["ah_vol"].rolling(22).mean()
    out["ah_vol_std_22d"] = out["ah_vol"].rolling(22).std()
    out["E_after_hours_vol_z"] = ((out["ah_vol"] - out["ah_vol_mean_22d"])
                                   / out["ah_vol_std_22d"].replace(0, np.nan)).clip(-5, 5)
    out["pm_vol_mean_22d"] = out["pm_vol"].rolling(22).mean()
    out["pm_vol_std_22d"] = out["pm_vol"].rolling(22).std()
    out["E_pre_market_vol_z"] = ((out["pm_vol"] - out["pm_vol_mean_22d"])
                                  / out["pm_vol_std_22d"].replace(0, np.nan)).clip(-5, 5)

    out["ts"] = pd.to_datetime(out["et_date"]).dt.tz_localize("UTC").astype("datetime64[ns, UTC]")
    out["symbol"] = sym
    keep = ["ts", "symbol", "E_after_hours_ret", "E_pre_market_gap",
            "E_after_hours_vol_z", "E_pre_market_vol_z", "E_overnight_total"]
    return out[keep]


def add_features_E(panel: pd.DataFrame, names: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Compute extended-hours features for cached Polygon names. If `names`
    is None, auto-detect from cache directory."""
    if names is None:
        from pathlib import Path
        cache_dir = Path("/home/yuqing/ctaNew/data/ml/cache")
        names = sorted([
            p.stem.replace("poly_", "").replace("_5m", "")
            for p in cache_dir.glob("poly_*_5m.parquet")
        ])
    log.info("Computing extended-hours features for %d names...", len(names))
    eh_frames = []
    for sym in names:
        eh = compute_eh_features_for_symbol(sym)
        if not eh.empty:
            log.info("  %-6s: %d days of EH features (%s -> %s)", sym, len(eh),
                     eh["ts"].iloc[0].strftime("%Y-%m-%d"),
                     eh["ts"].iloc[-1].strftime("%Y-%m-%d"))
            eh_frames.append(eh)
    if not eh_frames:
        return panel, []
    eh_all = pd.concat(eh_frames, ignore_index=True)
    eh_all["ts"] = pd.to_datetime(eh_all["ts"], utc=True).astype("datetime64[ns, UTC]")
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True).astype("datetime64[ns, UTC]")
    out = panel.merge(eh_all, on=["ts", "symbol"], how="left")
    feat_names = ["E_after_hours_ret", "E_pre_market_gap",
                  "E_after_hours_vol_z", "E_pre_market_vol_z", "E_overnight_total"]
    return out, feat_names


# ---- run with extended hours -------------------------------------------

def run_with_eh(panel: pd.DataFrame, earnings: pd.DataFrame, anchors: pd.DataFrame) -> tuple:
    panel = panel.copy()
    panel = add_residual_and_label(panel, HOLD_DAYS)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel, feats_E = add_features_E(panel)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    label = f"fwd_resid_{HOLD_DAYS}d"
    feats = feats_A + feats_B + feats_E + ["sym_id"]
    log.info("\nfeatures: A=%d  B=%d  E=%d  total=%d",
             len(feats_A), len(feats_B), len(feats_E), len(feats))

    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)
    # Core features = A + sym_id (always required). E + B optional (LGBM handles NaN).
    core = feats_A + ["sym_id"]
    all_pnls = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        test_pred = fit_predict_nan_tolerant(train, test, feats, label, core_features=core)
        if test_pred.empty:
            continue
        lp = construct_portfolio_subset(
            test_pred, "pred", label,
            allowed_symbols=set(XYZ_IN_SP100),
            top_k=TOP_K, cost_bps=COST_BPS_SIDE * 2,
            hold_days=HOLD_DAYS,
        )
        if not lp.empty:
            all_pnls.append(lp)
    if not all_pnls:
        return pd.DataFrame(), {}, {}
    pnl = pd.concat(all_pnls, ignore_index=True)
    pnl_g = gate_by_dispersion(pnl, regime, threshold_pctile=0.6)
    return pnl_g, metrics_freq(pnl_g, HOLD_DAYS), annualized_unconditional(pnl_g, HOLD_DAYS)


def main() -> None:
    log.info("loading S&P 100 + earnings + anchors...")
    panel, earnings, _ = load_universe()
    if panel.empty:
        return
    anchors_df = pd.read_parquet(Path("/home/yuqing/ctaNew/data/ml/cache/yf_SPY_1d_anchor.parquet"))
    # Use load_anchors via the regime module
    from ml.research.alpha_v7_multi import load_anchors
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)

    log.info("\n=== Baseline: fixed PEAD + A only (no extended hours), 3d hold ===")
    pnl_b, m_b, ann_b = run_baseline(panel, earnings, anchors, use_fixed_pead=True)
    log.info("  n_rebal=%d  net/d=%+.2fbps  active_Sh=%+.2f  uncond_Sh=%+.2f  ann_ret=%+.2f%%",
             m_b["n_rebal"], m_b["net_bps_per_day"],
             m_b["active_sharpe_annu"], ann_b["unconditional_sharpe"],
             ann_b["annual_return_pct"])

    log.info("\n=== Extended (fixed PEAD + A + E features), 3d hold ===")
    pnl_e, m_e, ann_e = run_with_eh(panel, earnings, anchors)
    log.info("  n_rebal=%d  net/d=%+.2fbps  active_Sh=%+.2f  uncond_Sh=%+.2f  ann_ret=%+.2f%%",
             m_e["n_rebal"], m_e["net_bps_per_day"],
             m_e["active_sharpe_annu"], ann_e["unconditional_sharpe"],
             ann_e["annual_return_pct"])

    delta_active = m_e["active_sharpe_annu"] - m_b["active_sharpe_annu"]
    delta_uncond = ann_e["unconditional_sharpe"] - ann_b["unconditional_sharpe"]
    log.info("\n  >>> Sharpe lift from extended-hours features: active %+.2f  uncond %+.2f",
             delta_active, delta_uncond)

    # Per-year, focusing on recent 2y where E features are available
    if not pnl_e.empty:
        pnl_y = pnl_e.copy()
        pnl_y["year"] = pnl_y["ts"].dt.year
        log.info("\n=== Per-year breakdown (E features kick in 2024+) ===")
        log.info("  %-6s %5s %12s %12s %10s",
                 "year", "n_reb", "gross/d", "net/d", "active_Sh")
        for y, g in pnl_y.groupby("year"):
            qm = metrics_freq(g, HOLD_DAYS)
            log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+10.2f",
                     y, qm["n_rebal"], qm["gross_bps_per_day"],
                     qm["net_bps_per_day"], qm["active_sharpe_annu"])


if __name__ == "__main__":
    main()
