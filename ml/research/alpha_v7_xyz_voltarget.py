"""A+B + dispersion gating + vol-target sizing on 29-name xyz universe.

Hypothesis: equal-weight sizing on the 29-name universe puts too much
risk in extreme-vol names (MSTR, COIN, RIVN, GME). Vol-target sizing
(weight_i ∝ 1/idio_vol_i) levels risk contributions and should preserve
alpha while cutting variance.

Compared:
  - Equal-weight 15-name (prior best, Sharpe +1.22)
  - Equal-weight 29-name (current, Sharpe +0.66)
  - Vol-target 29-name (this run)
  - Vol-target 15-name (control)
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.alpha_v7_multi import (
    BETA_WINDOW, LGB_PARAMS, SEEDS, TOP_K,
    load_anchors, add_returns_and_basket,
    add_features_A, add_features_B,
)
from ml.research.alpha_v7_weekly import (
    HOLD_DAYS, COST_PER_TRADE_BPS,
    add_residual_5d, fit_predict, metrics_weekly, bootstrap_ci_weekly, make_folds,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz import gate_by_dispersion
from ml.research.alpha_v7_xyz_full import (
    ALL_XYZ_US_EQUITY, XYZ_EXTRAS, load_combined_universe,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def construct_portfolio_voltarget(test_pred: pd.DataFrame, signal: str,
                                    pnl_label: str, allowed_symbols: set,
                                    top_k: int = 5,
                                    cost_bps: float = COST_PER_TRADE_BPS,
                                    hold_days: int = HOLD_DAYS,
                                    use_voltarget: bool = True,
                                    vol_col: str = "A_idio_vol_22d") -> pd.DataFrame:
    """Vol-target portfolio: weight_i ∝ 1 / idio_vol_22d_i (normalized so
    each leg sums to 1.0). Equal-weight if use_voltarget=False."""
    needed = [signal, pnl_label]
    if use_voltarget and vol_col in test_pred.columns:
        needed.append(vol_col)
    sub = test_pred[test_pred["symbol"].isin(allowed_symbols)].dropna(
        subset=needed).copy()
    if use_voltarget and vol_col not in sub.columns:
        log.warning("    vol_col=%s not in panel; falling back to equal-weight", vol_col)
        use_voltarget = False

    unique_ts = sorted(sub["ts"].unique())
    if not unique_ts:
        return pd.DataFrame()
    rebal_ts = unique_ts[::hold_days]
    rows = []
    prev_long_w: dict = {}
    prev_short_w: dict = {}
    for ts in rebal_ts:
        bar = sub[sub["ts"] == ts]
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values(signal)
        long_leg = bar.tail(top_k)
        short_leg = bar.head(top_k)

        if use_voltarget:
            inv_vol_l = 1.0 / long_leg[vol_col].clip(lower=1e-6)
            inv_vol_s = 1.0 / short_leg[vol_col].clip(lower=1e-6)
            w_l = (inv_vol_l / inv_vol_l.sum()).values
            w_s = (inv_vol_s / inv_vol_s.sum()).values
        else:
            w_l = np.ones(top_k) / top_k
            w_s = np.ones(top_k) / top_k

        long_alpha = (long_leg[pnl_label].values * w_l).sum()
        short_alpha = (short_leg[pnl_label].values * w_s).sum()
        spread = long_alpha - short_alpha

        # turnover: weighted weight changes
        cur_long_w = dict(zip(long_leg["symbol"].values, w_l))
        cur_short_w = dict(zip(short_leg["symbol"].values, w_s))
        all_long_syms = set(prev_long_w) | set(cur_long_w)
        all_short_syms = set(prev_short_w) | set(cur_short_w)
        long_turn = sum(abs(cur_long_w.get(s, 0) - prev_long_w.get(s, 0))
                         for s in all_long_syms)
        short_turn = sum(abs(cur_short_w.get(s, 0) - prev_short_w.get(s, 0))
                          for s in all_short_syms)
        # turnover scaled to "fraction of portfolio rebalanced"
        turnover = (long_turn + short_turn) / 2.0
        cost = turnover * cost_bps / 1e4

        rows.append({
            "ts": ts, "spread_alpha": spread,
            "long_alpha": long_alpha, "short_alpha": short_alpha,
            "turnover": turnover, "cost": cost,
            "net_alpha": spread - cost, "n_universe": len(bar),
        })
        prev_long_w, prev_short_w = cur_long_w, cur_short_w
    return pd.DataFrame(rows)


def run_config(panel: pd.DataFrame, anchors: pd.DataFrame, regime: pd.DataFrame,
               folds, allowed: set, label_name: str,
               top_k: int = 5, use_voltarget: bool = True) -> pd.DataFrame:
    feats_A = [c for c in panel.columns if c.startswith("A_")]
    feats_B = [c for c in panel.columns if c.startswith("B_")]
    feats = feats_A + feats_B + ["sym_id"]
    label = "fwd_resid_5d"
    pnl_label = "fwd_resid_5d"

    all_pnls = []
    for fold in folds:
        train_end, test_start, test_end = fold
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        test_pred = fit_predict(train, test, feats, label)
        if test_pred.empty:
            continue
        # fit_predict's `sub = test.dropna(subset=features).copy()` keeps ALL columns
        # of `test`, so A_idio_vol_22d / A_vol_22d are already present. No merge needed.
        lp = construct_portfolio_voltarget(
            test_pred, "pred", pnl_label,
            allowed_symbols=allowed, top_k=top_k,
            cost_bps=COST_PER_TRADE_BPS, hold_days=HOLD_DAYS,
            use_voltarget=use_voltarget,
        )
        if not lp.empty:
            all_pnls.append(lp)
    return pd.concat(all_pnls, ignore_index=True) if all_pnls else pd.DataFrame()


def main() -> None:
    log.info("loading combined universe...")
    panel, earnings, surv = load_combined_universe(min_history_days=365 * 2)
    if panel.empty:
        return
    anchors = load_anchors()

    panel = add_returns_and_basket(panel)
    panel = add_residual_5d(panel)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes

    log.info("computing regime indicators...")
    regime = compute_regime_indicators(panel, anchors)

    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)
    xyz_29 = set(s for s in ALL_XYZ_US_EQUITY if s in surv)
    xyz_15 = set([s for s in ALL_XYZ_US_EQUITY if s in surv
                   and s not in XYZ_EXTRAS])
    log.info("xyz tradeable: 29=%d, 15=%d", len(xyz_29), len(xyz_15))

    configs = [
        ("EW 15-name (prior best)", xyz_15, 5, False),
        ("EW 29-name", xyz_29, 5, False),
        ("VolTarget 15-name", xyz_15, 5, True),
        ("VolTarget 29-name", xyz_29, 5, True),
        ("VolTarget 29-name K=4", xyz_29, 4, True),
        ("VolTarget 29-name K=7", xyz_29, 7, True),
    ]

    log.info("\n=== UNGATED ===")
    log.info("  %-32s %5s %12s %10s %18s",
             "config", "n", "net/5d", "net_Sh", "95% CI")
    pnl_results = {}
    for name, allowed, k, vt in configs:
        log.info("  running %s ...", name)
        pnl = run_config(panel, anchors, regime, folds, allowed, name, top_k=k, use_voltarget=vt)
        if pnl.empty:
            continue
        pnl_results[name] = pnl
        m = metrics_weekly(pnl)
        lo, hi = bootstrap_ci_weekly(pnl)
        log.info("  %-32s %5d %+10.2fbps %+8.2f  [%+.2f, %+.2f]",
                 name, m["n_rebal"], m["net_bps_per_5d"],
                 m["net_sharpe_annu"], lo, hi)

    log.info("\n=== DISPERSION-GATED (top 40%) ===")
    log.info("  %-32s %5s %12s %10s %18s",
             "config", "n", "net/5d", "net_Sh", "95% CI")
    for name, pnl in pnl_results.items():
        gated = gate_by_dispersion(pnl, regime, threshold_pctile=0.6)
        if gated.empty:
            continue
        m = metrics_weekly(gated)
        lo, hi = bootstrap_ci_weekly(gated)
        log.info("  %-32s %5d %+10.2fbps %+8.2f  [%+.2f, %+.2f]",
                 name, m["n_rebal"], m["net_bps_per_5d"],
                 m["net_sharpe_annu"], lo, hi)

    # Per-year for best gated
    if pnl_results:
        best_name = max(pnl_results.keys(),
                        key=lambda k: metrics_weekly(gate_by_dispersion(pnl_results[k], regime, 0.6)).get("net_sharpe_annu", -10))
        log.info("\n=== Per-year gated, best=%s ===", best_name)
        gated = gate_by_dispersion(pnl_results[best_name], regime, 0.6)
        gated["year"] = gated["ts"].dt.year
        log.info("  %-6s %5s %12s %12s %10s %8s",
                 "year", "n_reb", "gross/5d", "net/5d", "net_Sh", "hit")
        for y, g in gated.groupby("year"):
            qm = metrics_weekly(g)
            log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+8.2f %7.0f%%",
                     y, qm["n_rebal"], qm["gross_bps_per_5d"],
                     qm["net_bps_per_5d"], qm["net_sharpe_annu"],
                     100 * qm["hit_rate"])


if __name__ == "__main__":
    main()
