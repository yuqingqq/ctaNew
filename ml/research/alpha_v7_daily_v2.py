"""Push daily-optimized strategy further. Layers:

  Sweep 1: exit buffer M ∈ {1, 2, 3, 5, 7} — find optimal hysteresis depth
  Sweep 2: forward horizon h ∈ {3, 5, 7, 10, 15} — find optimal training horizon
  Variant: multi-horizon ensemble (avg predictions of models trained on 3, 5, 10)
  Variant: signal-magnitude weighting (positions sized by |prediction - median|)
  Variant: asymmetric hysteresis (tighter exit on losers, looser on winners)
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe
from ml.research.alpha_v7_multi import (
    LGB_PARAMS, SEEDS, add_returns_and_basket, add_features_A, load_anchors,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100
from ml.research.alpha_v7_freq_sweep import (
    add_residual_and_label, metrics_freq, annualized_unconditional,
)
from ml.research.alpha_v7_pead_fixed import add_features_B_fixed
from ml.research.alpha_v7_honest import (
    fit_predict, gate_rolling, bootstrap_active_sharpe_ci,
)
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, make_folds, run_walk,
    metrics_for, boot_ci,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOP_K = 5
COST_BPS_SIDE = 1.5  # patient maker
GATE_PCTILE = 0.6
GATE_WINDOW = 252


def daily_portfolio_signal_weighted(test_pred: pd.DataFrame, signal: str, pnl_label: str,
                                       allowed: set, top_k: int, exit_buffer: int,
                                       cost_bps_side: float) -> pd.DataFrame:
    """Hysteresis + signal-magnitude weighting.
    Within top-K (long), weight ∝ |pred - median|. Distant predictions get more weight."""
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label]).copy()
    rows = []
    cur_long, cur_short = set(), set()
    prev_l_w, prev_s_w = {}, {}
    for ts, bar in sub.groupby("ts"):
        if len(bar) < 2 * top_k + exit_buffer:
            continue
        bar = bar.sort_values(signal).reset_index(drop=True)
        n = len(bar)
        bar["rank_top"] = n - 1 - bar.index
        bar["rank_bot"] = bar.index
        median_pred = bar[signal].median()

        # Update long set with hysteresis
        new_long = set(cur_long)
        for s in list(new_long):
            r = bar[bar["symbol"] == s]
            if r.empty or r["rank_top"].iloc[0] > top_k + exit_buffer - 1:
                new_long.discard(s)
        candidates = bar[bar["rank_top"] < top_k]["symbol"].tolist()
        for s in candidates:
            if len(new_long) >= top_k: break
            new_long.add(s)
        if len(new_long) > top_k:
            ranked = bar[bar["symbol"].isin(new_long)].sort_values("rank_top")
            new_long = set(ranked.head(top_k)["symbol"])

        new_short = set(cur_short)
        for s in list(new_short):
            r = bar[bar["symbol"] == s]
            if r.empty or r["rank_bot"].iloc[0] > top_k + exit_buffer - 1:
                new_short.discard(s)
        candidates_s = bar[bar["rank_bot"] < top_k]["symbol"].tolist()
        for s in candidates_s:
            if len(new_short) >= top_k: break
            new_short.add(s)
        if len(new_short) > top_k:
            ranked = bar[bar["symbol"].isin(new_short)].sort_values("rank_bot")
            new_short = set(ranked.head(top_k)["symbol"])

        if not new_long or not new_short:
            cur_long, cur_short = new_long, new_short
            continue

        # Signal-magnitude weights within each leg
        long_bar = bar[bar["symbol"].isin(new_long)].copy()
        short_bar = bar[bar["symbol"].isin(new_short)].copy()
        long_bar["mag"] = (long_bar[signal] - median_pred).clip(lower=1e-8)
        short_bar["mag"] = (median_pred - short_bar[signal]).clip(lower=1e-8)
        w_l = (long_bar["mag"] / long_bar["mag"].sum()).values
        w_s = (short_bar["mag"] / short_bar["mag"].sum()).values

        long_a = (long_bar[pnl_label].values * w_l).sum()
        short_a = (short_bar[pnl_label].values * w_s).sum()
        spread = long_a - short_a

        cur_l = dict(zip(long_bar["symbol"].values, w_l))
        cur_s = dict(zip(short_bar["symbol"].values, w_s))
        all_l = set(prev_l_w) | set(cur_l)
        all_s = set(prev_s_w) | set(cur_s)
        long_turn = sum(abs(cur_l.get(s, 0) - prev_l_w.get(s, 0)) for s in all_l)
        short_turn = sum(abs(cur_s.get(s, 0) - prev_s_w.get(s, 0)) for s in all_s)
        turnover = (long_turn + short_turn) / 2.0
        cost = turnover * cost_bps_side * 2 / 1e4
        rows.append({"ts": ts, "spread_alpha": spread, "long_alpha": long_a,
                     "short_alpha": short_a, "turnover": turnover, "cost": cost,
                     "net_alpha": spread - cost, "n_universe": n})
        cur_long, cur_short = new_long, new_short
        prev_l_w, prev_s_w = cur_l, cur_s
    return pd.DataFrame(rows)


def run_walk_multihorizon(panel, feats, train_labels, folds, port_fn, port_kwargs):
    """Train one model per horizon, average predictions, then build portfolio."""
    all_pnls = []
    for fold in folds:
        te, ts, te2 = fold
        train = panel[panel["ts"] <= te].copy()
        test = panel[(panel["ts"] >= ts) & (panel["ts"] <= te2)].copy()
        ensemble_preds = []
        sub_test = None
        for label in train_labels:
            train_ = train.dropna(subset=feats + [label])
            if len(train_) < 1000:
                continue
            sub = test.dropna(subset=feats).copy()
            if sub_test is None:
                sub_test = sub.copy()
            preds_seed = []
            for seed in SEEDS:
                m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
                m.fit(train_[feats], train_[label])
                preds_seed.append(m.predict(sub[feats]))
            ensemble_preds.append(np.mean(preds_seed, axis=0))
        if not ensemble_preds or sub_test is None:
            continue
        sub_test["pred"] = np.mean(ensemble_preds, axis=0)
        lp = port_fn(sub_test, "pred", **port_kwargs)
        if not lp.empty:
            all_pnls.append(lp)
    return pd.concat(all_pnls, ignore_index=True) if all_pnls else pd.DataFrame()


# ---- main ---------------------------------------------------------------

def main() -> None:
    log.info("loading panel...")
    panel, earnings, _ = load_universe()
    if panel.empty:
        return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    # build labels for multiple horizons
    for h in (1, 3, 5, 7, 10, 15):
        panel = add_residual_and_label(panel, h)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    feats = feats_A + feats_B + ["sym_id"]
    allowed = set(XYZ_IN_SP100)
    folds = make_folds(panel)

    log.info("\n=== Sweep 1: exit_buffer M, h=5 fixed ===")
    log.info("  %-10s  %5s  %10s  %18s  %10s  %12s  %10s",
             "M", "n_reb", "active_Sh", "95% CI", "uncond_Sh", "annu_cost", "turn%/d")
    sweep_m = {}
    for M in (1, 2, 3, 4, 5):
        pnl_pre = run_walk(panel, feats, "fwd_resid_5d", folds,
                            daily_portfolio_hysteresis,
                            {"pnl_label": "fwd_resid_1d", "allowed": allowed,
                             "top_k": TOP_K, "exit_buffer": M,
                             "cost_bps_side": COST_BPS_SIDE})
        if pnl_pre.empty or len(pnl_pre) < 30:
            log.info("  M=%-8d  too few trades (%d), skipping", M, len(pnl_pre))
            continue
        if pnl_pre.empty or len(pnl_pre) < 30:
            continue
        pnl = gate_rolling(pnl_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
        if pnl.empty:
            continue
        m = metrics_for(pnl, 1)
        lo, hi = boot_ci(pnl, 1)
        sweep_m[M] = (m, lo, hi)
        log.info("  M=%-8d  %5d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f  %12.0f  %8.0f%%",
                 M, m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["uncond_sharpe"], m["annual_cost_bps"], m["avg_turnover_pct"])

    log.info("\n=== Sweep 2: forward horizon h, M=2 fixed ===")
    log.info("  %-10s  %5s  %10s  %18s  %10s  %12s  %10s",
             "h", "n_reb", "active_Sh", "95% CI", "uncond_Sh", "annu_cost", "turn%/d")
    sweep_h = {}
    for h in (3, 5, 7, 10, 15):
        pnl_pre = run_walk(panel, feats, f"fwd_resid_{h}d", folds,
                            daily_portfolio_hysteresis,
                            {"pnl_label": "fwd_resid_1d", "allowed": allowed,
                             "top_k": TOP_K, "exit_buffer": 2,
                             "cost_bps_side": COST_BPS_SIDE})
        if pnl_pre.empty or len(pnl_pre) < 30:
            continue
        pnl = gate_rolling(pnl_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
        if pnl.empty:
            continue
        m = metrics_for(pnl, 1)
        lo, hi = boot_ci(pnl, 1)
        sweep_h[h] = (m, lo, hi)
        log.info("  h=%-8d  %5d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f  %12.0f  %8.0f%%",
                 h, m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["uncond_sharpe"], m["annual_cost_bps"], m["avg_turnover_pct"])

    log.info("\n=== Multi-horizon ensemble (avg of h=3,5,10), M=2 ===")
    pnl_mh_pre = run_walk_multihorizon(
        panel, feats, ["fwd_resid_3d", "fwd_resid_5d", "fwd_resid_10d"], folds,
        daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": allowed,
         "top_k": TOP_K, "exit_buffer": 2, "cost_bps_side": COST_BPS_SIDE},
    )
    pnl_mh = gate_rolling(pnl_mh_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m_mh = metrics_for(pnl_mh, 1)
    lo_mh, hi_mh = boot_ci(pnl_mh, 1)
    log.info("  multi-h (3,5,10)  %5d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f  %12.0f  %8.0f%%",
             m_mh["n_rebal"], m_mh["active_sharpe"], lo_mh, hi_mh,
             m_mh["uncond_sharpe"], m_mh["annual_cost_bps"], m_mh["avg_turnover_pct"])

    log.info("\n=== Signal-magnitude weighting (h=5, M=2) ===")
    pnl_sw_pre = run_walk(panel, feats, "fwd_resid_5d", folds,
                           daily_portfolio_signal_weighted,
                           {"pnl_label": "fwd_resid_1d", "allowed": allowed,
                            "top_k": TOP_K, "exit_buffer": 2,
                            "cost_bps_side": COST_BPS_SIDE})
    pnl_sw = gate_rolling(pnl_sw_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m_sw = metrics_for(pnl_sw, 1)
    lo_sw, hi_sw = boot_ci(pnl_sw, 1)
    log.info("  sig-mag wt        %5d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f  %12.0f  %8.0f%%",
             m_sw["n_rebal"], m_sw["active_sharpe"], lo_sw, hi_sw,
             m_sw["uncond_sharpe"], m_sw["annual_cost_bps"], m_sw["avg_turnover_pct"])

    # Final: best of all variants
    log.info("\n=== HEADLINE: best variant per category ===")
    best_m = max(sweep_m.keys(), key=lambda k: sweep_m[k][0]["active_sharpe"])
    best_h = max(sweep_h.keys(), key=lambda k: sweep_h[k][0]["active_sharpe"])
    log.info("  best M=%d (Sharpe %+.2f), best h=%d (Sharpe %+.2f)",
             best_m, sweep_m[best_m][0]["active_sharpe"],
             best_h, sweep_h[best_h][0]["active_sharpe"])

    log.info("\n=== Combined: best M + best h + signal-mag weight ===")
    pnl_combo_pre = run_walk(panel, feats, f"fwd_resid_{best_h}d", folds,
                              daily_portfolio_signal_weighted,
                              {"pnl_label": "fwd_resid_1d", "allowed": allowed,
                               "top_k": TOP_K, "exit_buffer": best_m,
                               "cost_bps_side": COST_BPS_SIDE})
    pnl_combo = gate_rolling(pnl_combo_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    m_c = metrics_for(pnl_combo, 1)
    lo_c, hi_c = boot_ci(pnl_combo, 1)
    log.info("  combo h=%d M=%d sigwt  %5d  %+8.2f  [%+5.2f,%+5.2f]  %+8.2f  %12.0f  %8.0f%%",
             best_h, best_m, m_c["n_rebal"], m_c["active_sharpe"], lo_c, hi_c,
             m_c["uncond_sharpe"], m_c["annual_cost_bps"], m_c["avg_turnover_pct"])

    # Per-year for the combo
    if not pnl_combo.empty:
        pnl_combo["year"] = pnl_combo["ts"].dt.year
        log.info("\n=== Per-year for combo ===")
        log.info("  %-6s %5s %12s %12s %10s",
                 "year", "n_reb", "gross/d", "net/d", "active_Sh")
        for y, g in pnl_combo.groupby("year"):
            qm = metrics_for(g, 1)
            log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+10.2f",
                     y, qm["n_rebal"], g["spread_alpha"].mean() * 1e4,
                     qm["net_bps_per_rebal"], qm["active_sharpe"])


if __name__ == "__main__":
    main()
