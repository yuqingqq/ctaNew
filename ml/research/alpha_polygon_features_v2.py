"""Per-feature analysis on the v2-cleaned Polygon panel.

For each feature in the v6_clean port:
  1. Spearman IC vs fwd_resid_48_clean (per-ts cross-sectional rank corr)
  2. Standalone long-short top-K=3 portfolio gross alpha per 4h
  3. 95% block-bootstrap CIs on both

Uses the v2 panel (calendar-correct windows, leave-one-out basket,
overnight-masked label, per-ts-demeaned target). This isolates whether
individual features have ANY signal after the cleaning, separate from
the LGBM's ability to combine them.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml.research.alpha_polygon_lgbm_v2 import (
    UNIVERSE, FEATURES, H, RTH_BARS_PER_DAY, TOP_K,
    load_panel, add_returns, add_loo_basket, add_residualization,
    add_features, add_label, portfolio, metrics,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BOOT_N = 2000
BLOCK_DAYS = 30
COST_BPS = 24


# Group features for readable reporting
FEATURE_GROUPS = {
    "BASE": [
        "return_1d", "ema_slope_20_1h", "atr_pct", "volume_ma_50",
        "bars_since_high", "hour_cos", "hour_sin",
    ],
    "CROSS_DOM": [
        "dom_level_vs_bk", "dom_change_12b_vs_bk", "dom_change_48b_vs_bk",
        "dom_change_78b_vs_bk", "dom_z_1d_vs_bk", "dom_z_5d_vs_bk",
    ],
    "CROSS_BK": [
        "bk_ret_1h", "bk_ret_4h", "bk_realized_vol_1h",
        "beta", "corr_1d_vs_bk", "corr_change_3d_vs_bk",
    ],
    "CROSS_IDIO": [
        "idio_ret_1h_vs_bk", "idio_ret_4h_vs_bk",
        "idio_vol_1h_vs_bk", "idio_vol_1d_vs_bk",
    ],
    "FLOW": [
        "obv_z_1d", "obv_signal", "vwap_zscore", "mfi",
        "price_volume_corr_10", "price_volume_corr_20",
    ],
    "XS_RANK": [
        "return_1d_xs_rank", "atr_pct_xs_rank", "ema_slope_20_1h_xs_rank",
        "idio_vol_1d_vs_bk_xs_rank", "obv_z_1d_xs_rank",
        "vwap_zscore_xs_rank", "bars_since_high_xs_rank",
    ],
}


# ---- IC per ts ---------------------------------------------------------

def per_ts_ic(panel: pd.DataFrame, feat: str, label: str) -> pd.Series:
    sub = panel.dropna(subset=[feat, label])
    def _ic(g):
        if len(g) < 4:
            return np.nan
        r, _ = spearmanr(g[feat], g[label])
        return r
    return sub.groupby("ts").apply(_ic).dropna()


def block_bootstrap_ic(daily_ic: pd.Series, n: int = BOOT_N,
                       block_days: int = 5) -> tuple[float, float]:
    """Block bootstrap CI for mean IC."""
    daily = daily_ic.resample("1D").mean().dropna()
    arr = daily.values
    if len(arr) < block_days * 2:
        return np.nan, np.nan
    n_blocks = max(1, len(arr) // block_days)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n):
        starts = rng.integers(0, len(arr) - block_days + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_days] for s in starts])
        means.append(sample.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def block_bootstrap_pnl_sharpe(pnl: pd.DataFrame, cost_bps: float = COST_BPS,
                                block_days: int = BLOCK_DAYS,
                                n_boot: int = BOOT_N) -> tuple[float, float]:
    if pnl.empty:
        return np.nan, np.nan
    pnl = pnl.copy()
    pnl["net"] = pnl["spread_alpha"] - cost_bps / 1e4
    pnl["date"] = pnl["ts"].dt.date
    daily = pnl.groupby("date")["net"].sum()
    arr = daily.values
    if len(arr) < block_days * 2:
        return np.nan, np.nan
    n_blocks = max(1, len(arr) // block_days)
    rng = np.random.default_rng(42)
    sh = []
    for _ in range(n_boot):
        starts = rng.integers(0, len(arr) - block_days + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_days] for s in starts])
        if sample.std() > 0:
            sh.append(sample.mean() / sample.std() * np.sqrt(252))
    if not sh:
        return np.nan, np.nan
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


# ---- main --------------------------------------------------------------

def main() -> None:
    log.info("loading + featurizing v2 panel...")
    panel = load_panel()
    panel = add_returns(panel)
    panel = add_loo_basket(panel)
    panel = add_residualization(panel)
    panel = add_features(panel)
    panel = add_label(panel, H)

    label_clean = f"fwd_resid_{H}_clean"
    label_demean = f"fwd_resid_{H}_demean"
    label_pnl = f"fwd_resid_{H}"  # raw label for portfolio P&L

    n_clean = panel[label_clean].notna().sum()
    log.info("  panel rows=%d, clean-label rows=%d (%.0f%%)",
             len(panel), n_clean, 100 * n_clean / len(panel))

    # ---- IC against demeaned clean label ------------------------------
    log.info("\n=== Per-feature Spearman IC vs %s (h=%d=4h) ===", label_demean, H)
    log.info("  %-32s %8s %12s %5s",
             "feature", "IC", "95% CI", "n_ts")

    ic_results = []
    for grp, feats in FEATURE_GROUPS.items():
        log.info("  --- %s ---", grp)
        for f in feats:
            if f not in panel.columns:
                continue
            ic = per_ts_ic(panel, f, label_demean)
            if len(ic) == 0:
                continue
            mu = ic.mean()
            lo, hi = block_bootstrap_ic(ic)
            sig = "***" if (not np.isnan(lo) and (lo > 0 or hi < 0)) else "   "
            log.info("  %-32s %+8.4f  [%+.4f, %+.4f]  %5d %s",
                     f, mu, lo, hi, len(ic), sig)
            ic_results.append({
                "feature": f, "group": grp, "ic": mu,
                "ic_lo": lo, "ic_hi": hi, "n_ts": len(ic),
            })

    # ---- Standalone long-short portfolio per feature -------------------
    log.info("\n=== Per-feature long-short top-K=%d portfolio (cost=%d bps) ===",
             TOP_K, COST_BPS)
    log.info("  Each feature is used as the ranking signal; rebalance every "
             "%d bars at clean-label start-of-day rebalance points.", H)
    log.info("  %-32s %5s %12s %12s %12s %16s",
             "feature", "n", "gross/4h", "net/4h", "Sharpe", "Sh 95% CI")

    pnl_results = []
    for grp, feats in FEATURE_GROUPS.items():
        log.info("  --- %s ---", grp)
        for f in feats:
            if f not in panel.columns:
                continue
            pnl = portfolio(panel, f, label_pnl, top_k=TOP_K)
            if pnl.empty:
                continue
            m = metrics(pnl, cost_bps=COST_BPS)
            lo, hi = block_bootstrap_pnl_sharpe(pnl)
            sig = "***" if (not np.isnan(lo) and (lo > 0 or hi < 0)) else "   "
            log.info("  %-32s %5d %+10.1fbps %+10.1fbps %+10.2f  [%+.2f, %+.2f] %s",
                     f, m["n"], m["gross_bps"], m["net_bps"],
                     m["net_sharpe"], lo, hi, sig)
            pnl_results.append({
                "feature": f, "group": grp,
                "gross_bps": m["gross_bps"], "net_bps": m["net_bps"],
                "net_sharpe": m["net_sharpe"], "sh_lo": lo, "sh_hi": hi,
                "n": m["n"],
            })

    # ---- summary ------------------------------------------------------
    log.info("\n=== SUMMARY ===")
    n_features = len(ic_results)
    n_sig_ic = sum(1 for r in ic_results
                    if not np.isnan(r["ic_lo"])
                    and (r["ic_lo"] > 0 or r["ic_hi"] < 0))
    n_sig_pnl = sum(1 for r in pnl_results
                     if not np.isnan(r["sh_lo"])
                     and (r["sh_lo"] > 0 or r["sh_hi"] < 0))
    log.info("  features tested:           %d", n_features)
    log.info("  features with IC CI excluding 0:    %d", n_sig_ic)
    log.info("  features with Sharpe CI excluding 0: %d", n_sig_pnl)
    log.info("  expected by chance (α=5%%):  %.1f", n_features * 0.05)
    if ic_results:
        ics = [r["ic"] for r in ic_results]
        log.info("  IC distribution: mean=%+.4f  std=%.4f  range=[%+.4f, %+.4f]",
                 np.mean(ics), np.std(ics), min(ics), max(ics))
    if pnl_results:
        sharpes = [r["net_sharpe"] for r in pnl_results]
        log.info("  Sharpe distribution: mean=%+.2f  std=%.2f  range=[%+.2f, %+.2f]",
                 np.mean(sharpes), np.std(sharpes), min(sharpes), max(sharpes))


if __name__ == "__main__":
    main()
