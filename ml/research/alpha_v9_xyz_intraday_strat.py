"""Standalone intraday-feature L/S backtest on xyz Tier A+B.

Strategy spec (mirrors v7 C5L for fair comparison):
  - Universe: TIER_AB (11 names)
  - Signal: first_vs_last_xs (cross-sectional residual within TIER_AB)
  - Daily rebalance, top-K long / bot-K short, hysteresis M_exit=1
  - Dispersion gate: same as v7 (60th-pctile of trailing 252-day cross-sectional dispersion)
  - Cost 0.8 bps/side, P&L on fwd_resid_1d
  - Window: 2024-06 to 2026-05 (Polygon coverage)

Compare to v7 baseline on the SAME window for apples-to-apples (v7's +3.29
is on full 2016-2026 sample — would be unfair vs intraday-strat which has
only the recent 2y).

Tested K ∈ {3, 4, 5} for sensitivity.

Usage:
    python -m ml.research.alpha_v9_xyz_intraday_strat
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, metrics_for, boot_ci,
)
from ml.research.alpha_v9_xyz_pm import load_or_compute_regime
from ml.research.alpha_v9_xyz_intraday_ic import session_features
from ml.research.alpha_v7_tier_a import TIER_AB

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE = CACHE / "v7_tier_a_walkfwd_preds.parquet"
GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8


def build_intraday_signal_panel() -> pd.DataFrame:
    """Build panel with (ts, symbol, signal, fwd_resid_1d) where signal is
    first_vs_last_xs computed cross-sectionally over TIER_AB names."""
    log.info("loading cached preds for fwd_resid_1d ...")
    preds = pd.read_parquet(PRED_CACHE)
    preds["date"] = pd.to_datetime(preds["ts"]).dt.tz_convert(None).dt.normalize()
    preds = preds[preds["symbol"].isin(TIER_AB)].copy()

    rows = []
    for sym in TIER_AB:
        poly_path = CACHE / f"poly_{sym}_5m.parquet"
        if not poly_path.exists():
            log.info("  %s: no polygon cache; skip", sym); continue
        poly = pd.read_parquet(poly_path)
        feats = session_features(poly)
        feats["symbol"] = sym
        # Join with preds (provides fwd_resid_1d)
        sub = preds[preds["symbol"] == sym][["date", "ts", "fwd_resid_1d"]]
        merged = feats.merge(sub, on="date", how="inner")
        rows.append(merged)
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["first_vs_last", "first_30_ret", "vwap_dev",
                            "fwd_resid_1d"]).reset_index(drop=True)

    # Cross-sectional residualization within TIER_AB, per date
    for c in ["first_vs_last", "first_30_ret", "vwap_dev"]:
        df[c + "_xs"] = df[c] - df.groupby("date")[c].transform("median")
    log.info("  panel: %d rows, %d names, ts %s..%s",
             len(df), df["symbol"].nunique(),
             df["ts"].min().date(), df["ts"].max().date())
    return df


def evaluate(panel: pd.DataFrame, regime: pd.DataFrame, *,
              signal: str, K: int, M_exit: int, name: str,
              use_gate: bool = True) -> dict | None:
    log.info(">>> %s  signal=%s K=%d M=%d gate=%s",
             name, signal, K, M_exit, use_gate)
    pnl_pre = daily_portfolio_hysteresis(
        panel, signal, "fwd_resid_1d",
        set(TIER_AB), K, M_exit, COST_BPS_SIDE)
    if pnl_pre.empty:
        log.warning("  empty pnl"); return None
    if use_gate:
        pnl = gate_rolling(pnl_pre, regime, pctile=GATE_PCTILE,
                              window_days=GATE_WINDOW)
    else:
        pnl = pnl_pre.copy()
    if pnl.empty:
        log.warning("  empty after gate"); return None
    m = metrics_for(pnl, 1)
    lo, hi = boot_ci(pnl, 1)
    log.info("  n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f net=%+.2f bps  turn=%.1f%%",
             m["n_rebal"], m["active_sharpe"], lo, hi, m["uncond_sharpe"],
             m["net_bps_per_rebal"], m["avg_turnover_pct"])
    return {"name": name, "metrics": m, "ci": (lo, hi), "pnl": pnl}


def main() -> None:
    panel = build_intraday_signal_panel()
    regime = load_or_compute_regime()

    # Restrict regime to the panel window for sane comparison; gate uses
    # trailing 252d so it auto-aligns.

    # ---- v7 baseline restricted to same 2024-2026 window ----
    log.info("\n=== v7 BASELINE on the 2024-2026 window (same dates as intraday) ===")
    v7_preds = pd.read_parquet(PRED_CACHE)
    v7_preds["date"] = pd.to_datetime(v7_preds["ts"]).dt.tz_convert(None).dt.normalize()
    panel_dates = set(panel["date"])
    v7_window = v7_preds[v7_preds["date"].isin(panel_dates)].copy()
    log.info("  v7 preds in same window: %d rows", len(v7_window))
    v7_pnl_pre = daily_portfolio_hysteresis(
        v7_window, "pred", "fwd_resid_1d",
        set(TIER_AB), 4, 1, COST_BPS_SIDE)
    v7_pnl = gate_rolling(v7_pnl_pre, regime, pctile=GATE_PCTILE,
                             window_days=GATE_WINDOW)
    v7m = metrics_for(v7_pnl, 1)
    v7lo, v7hi = boot_ci(v7_pnl, 1)
    log.info("  v7 (K=4, M=1, gate, same window): n=%d Sh=%+.2f [%+.2f,%+.2f] net=%+.2f bps",
             v7m["n_rebal"], v7m["active_sharpe"], v7lo, v7hi, v7m["net_bps_per_rebal"])

    # ---- intraday-signal variants ----
    log.info("\n=== Intraday-signal standalone L/S (Tier A+B, 2024-2026) ===")
    out = []
    out.append(evaluate(panel, regime, signal="first_vs_last_xs", K=4, M_exit=1,
                          name="first_vs_last K=4 M=1 (gate)"))
    out.append(evaluate(panel, regime, signal="first_vs_last_xs", K=3, M_exit=1,
                          name="first_vs_last K=3 M=1 (gate)"))
    out.append(evaluate(panel, regime, signal="first_vs_last_xs", K=5, M_exit=1,
                          name="first_vs_last K=5 M=1 (gate)"))
    out.append(evaluate(panel, regime, signal="first_vs_last_xs", K=4, M_exit=2,
                          name="first_vs_last K=4 M=2 (gate)"))
    out.append(evaluate(panel, regime, signal="first_30_ret_xs", K=4, M_exit=1,
                          name="first_30_ret K=4 M=1 (gate)"))
    out.append(evaluate(panel, regime, signal="vwap_dev_xs", K=4, M_exit=1,
                          name="-vwap_dev K=4 M=1 (gate)"))
    # No-gate variants
    out.append(evaluate(panel, regime, signal="first_vs_last_xs", K=4, M_exit=1,
                          name="first_vs_last K=4 M=1 (NO gate)", use_gate=False))
    out.append(evaluate(panel, regime, signal="first_30_ret_xs", K=4, M_exit=1,
                          name="first_30_ret K=4 M=1 (NO gate)", use_gate=False))

    # ---- summary table ----
    log.info("\n=== SUMMARY (window 2024-2026, Tier A+B, cost 0.8 bps/side) ===")
    log.info("  %-40s %5s %10s %18s %12s",
             "config", "n", "Sh", "95% CI", "net bps/cyc")
    log.info("  %-40s %5d %+8.2f [%+5.2f,%+5.2f] %+10.2f",
             "v7 BASELINE same-window (K=4 M=1 gate)",
             v7m["n_rebal"], v7m["active_sharpe"], v7lo, v7hi,
             v7m["net_bps_per_rebal"])
    for r in out:
        if r is None: continue
        m = r["metrics"]; lo, hi = r["ci"]
        log.info("  %-40s %5d %+8.2f [%+5.2f,%+5.2f] %+10.2f",
                 r["name"], m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["net_bps_per_rebal"])

    # ---- per-year ----
    log.info("\n=== Per-year active Sharpe (best intraday config vs v7 same-window) ===")
    best = max([r for r in out if r is not None],
                key=lambda r: r["metrics"]["active_sharpe"])
    log.info("  best intraday: %s", best["name"])
    log.info("  %-6s %12s %12s",
             "year", "v7", best["name"][:18])
    for y in sorted(set(pd.to_datetime(v7_pnl["ts"]).dt.year) |
                     set(pd.to_datetime(best["pnl"]["ts"]).dt.year)):
        v7y = v7_pnl[pd.to_datetime(v7_pnl["ts"]).dt.year == y]
        bsy = best["pnl"][pd.to_datetime(best["pnl"]["ts"]).dt.year == y]
        sh_v7 = (metrics_for(v7y, 1)["active_sharpe"] if len(v7y) >= 5 else np.nan)
        sh_bs = (metrics_for(bsy, 1)["active_sharpe"] if len(bsy) >= 5 else np.nan)
        log.info("  %-6d %+12.2f %+12.2f", y, sh_v7, sh_bs)

    # ---- correlation between intraday and v7 P&L ----
    if best:
        merged = v7_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "v7"}).merge(
            best["pnl"][["ts", "net_alpha"]].rename(columns={"net_alpha": "id"}),
            on="ts")
        if len(merged) > 30:
            rho = merged[["v7", "id"]].corr().iloc[0, 1]
            log.info("\n  corr(v7 net_alpha, intraday net_alpha) on shared ts: %+.3f", rho)
            log.info("  → high corr means redundant; low corr means orthogonal alpha")
            # Naive equal-weight blend
            merged["blend"] = (merged["v7"] + merged["id"]) / 2
            sh_blend = (merged["blend"].mean() / merged["blend"].std()
                          * np.sqrt(252) if merged["blend"].std() > 0 else 0)
            log.info("  Equal-weight blend (v7 + intraday)/2:  Sh=%+.2f",
                     sh_blend)


if __name__ == "__main__":
    main()
