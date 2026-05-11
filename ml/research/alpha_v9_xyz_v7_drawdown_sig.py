"""Phase 1: v7 drawdown signature on full 2016-2026 history.

Goal: identify indicators that LEAD v7's losing periods, on enough data
(10 years, multiple regimes) to honestly tune and validate later.

Approach:
  - Build v7 portfolio P&L on Tier A+B for full cached-preds window
  - Compute candidate indicators per ts (pred-side, regime-side)
  - Test lead-lag IC: indicator(t) vs v7_net_alpha[t..t+L] for L ∈ {0, 5, 22}
  - Conditional Sharpe by indicator quartile (Q1 vs Q4 forward 22d Sh)
  - Identify v7 drawdown clusters and check which indicators were elevated/depressed
    in the days leading up

Output: a ranked list of indicators by their forward-22d-IC against v7 P&L.
Anything with |IC| > 0.10 and consistent sign across drawdown events is a
candidate for Phase 2 backtest.

Indicators tested:
  pred_disp_TAB        cross-sectional std of v7 preds within Tier A+B
  pred_disp_full       cross-sectional std of v7 preds within full SP100
  pred_mag_TAB         mean abs of v7 preds within Tier A+B
  pred_topK_mag        mean pred for Tier A+B top-K (long candidate strength)
  pred_botK_mag        |mean pred for Tier A+B bot-K| (short candidate strength)
  pred_spread_K        top-K − bot-K predicted spread (the ex-ante alpha leg gap)
  realized_disp_22d    from regime data (cross-sectional 22d-return std)

All indicators .shift(1) before correlating to avoid look-ahead.
Indicators are also 22d-smoothed for noise reduction.

Usage:
    python -m ml.research.alpha_v9_xyz_v7_drawdown_sig
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import daily_portfolio_hysteresis
from ml.research.alpha_v9_xyz_pm import load_or_compute_regime
from ml.research.alpha_v7_tier_a import TIER_AB

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE_V7 = CACHE / "v7_tier_a_walkfwd_preds.parquet"
GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8
TOP_K = 4
M_EXIT = 1


def build_v7_pnl_and_indicators() -> pd.DataFrame:
    log.info("loading v7 cached preds (full history) ...")
    preds = pd.read_parquet(PRED_CACHE_V7)
    preds["ts"] = pd.to_datetime(preds["ts"], utc=True)
    log.info("  v7 preds: %d rows, %d names, ts %s..%s",
             len(preds), preds["symbol"].nunique(),
             preds["ts"].min().date(), preds["ts"].max().date())

    # v7 portfolio P&L on Tier A+B
    regime = load_or_compute_regime()
    pre = daily_portfolio_hysteresis(preds, "pred", "fwd_resid_1d",
                                       set(TIER_AB), TOP_K, M_EXIT, COST_BPS_SIDE)
    v7_pnl = gate_rolling(pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    v7_pnl["ts"] = pd.to_datetime(v7_pnl["ts"], utc=True)
    log.info("  v7_pnl: %d cycles  ts %s..%s", len(v7_pnl),
             v7_pnl["ts"].min().date(), v7_pnl["ts"].max().date())

    # Indicators per ts: pred-side
    preds_TAB = preds[preds["symbol"].isin(TIER_AB)].copy()

    # Per-ts indicator computation
    rows = []
    for ts, g in preds_TAB.groupby("ts"):
        if len(g) < 2 * TOP_K + M_EXIT:
            continue
        sorted_p = g.sort_values("pred")
        topK = sorted_p.tail(TOP_K)["pred"].mean()
        botK = sorted_p.head(TOP_K)["pred"].mean()
        rows.append({
            "ts": ts,
            "pred_disp_TAB": g["pred"].std(),
            "pred_mag_TAB": g["pred"].abs().mean(),
            "pred_topK_mag": topK,
            "pred_botK_mag": abs(botK),
            "pred_spread_K": topK - botK,
        })
    ind_TAB = pd.DataFrame(rows)

    # Full-SP100 pred dispersion (using all available names that day)
    rows_full = []
    for ts, g in preds.groupby("ts"):
        if len(g) < 50: continue
        rows_full.append({"ts": ts, "pred_disp_full": g["pred"].std()})
    ind_full = pd.DataFrame(rows_full)

    # Realized cross-sectional dispersion from regime data
    reg = regime[["ts", "disp_22d"]].copy()
    reg["ts"] = pd.to_datetime(reg["ts"], utc=True)
    reg = reg.rename(columns={"disp_22d": "realized_disp_22d"})

    # Merge all indicators on v7_pnl ts (only ts where v7 traded)
    panel = v7_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "v7_net"})
    panel = panel.merge(ind_TAB, on="ts", how="left")
    panel = panel.merge(ind_full, on="ts", how="left")
    panel = panel.merge(reg, on="ts", how="left")
    panel = panel.sort_values("ts").reset_index(drop=True)
    log.info("  panel: %d ts × %d indicator cols", len(panel),
             sum(1 for c in panel.columns if c not in ("ts", "v7_net")))
    return panel


def smooth_and_shift(panel: pd.DataFrame, indicators: list[str],
                       window: int = 22) -> pd.DataFrame:
    df = panel.copy()
    for c in indicators:
        df[c + "_sm"] = df[c].rolling(window, min_periods=5).mean().shift(1)
    return df


def lead_lag_ic(df: pd.DataFrame, ind_col: str, leads: list[int]) -> dict:
    """For each lead L, compute Pearson IC between ind_col(t) and forward
    cumulative v7_net[t..t+L]."""
    out = {}
    for L in leads:
        if L == 0:
            fwd = df["v7_net"]
        else:
            # cumulative net_alpha over forward L bars (rolling sum, shifted)
            fwd = df["v7_net"].rolling(L, min_periods=max(L // 2, 1)).sum().shift(-L + 1)
        valid = df[ind_col].notna() & fwd.notna()
        if valid.sum() < 30:
            out[L] = (np.nan, 0); continue
        ic = np.corrcoef(df[ind_col][valid], fwd[valid])[0, 1]
        out[L] = (ic, int(valid.sum()))
    return out


def conditional_sharpe(df: pd.DataFrame, ind_col: str, fwd_window: int = 22
                        ) -> dict:
    """For each indicator quartile, compute forward fwd_window-day v7 Sharpe."""
    df = df.dropna(subset=[ind_col]).copy()
    df["fwd_sum"] = df["v7_net"].rolling(fwd_window, min_periods=fwd_window // 2).sum().shift(-fwd_window + 1)
    df = df.dropna(subset=["fwd_sum"])
    if len(df) < 100: return {}
    df["q"] = pd.qcut(df[ind_col], 4, labels=[1, 2, 3, 4], duplicates="drop")
    out = {}
    for q in [1, 2, 3, 4]:
        sub = df[df["q"] == q]
        if len(sub) < 10: continue
        s = sub["v7_net"]
        sh = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        out[q] = {"n": len(sub), "v7_sh": sh,
                   "fwd_mean_bps": sub["fwd_sum"].mean() * 1e4 / fwd_window,
                   "ind_range": (sub[ind_col].min(), sub[ind_col].max())}
    return out


def main() -> None:
    panel = build_v7_pnl_and_indicators()
    indicators = ["pred_disp_TAB", "pred_disp_full", "pred_mag_TAB",
                   "pred_topK_mag", "pred_botK_mag", "pred_spread_K",
                   "realized_disp_22d"]
    panel = smooth_and_shift(panel, indicators, window=22)

    # Drop early periods where smooth is NaN
    panel_full = panel.copy()
    sm_cols = [c + "_sm" for c in indicators]
    panel_clean = panel_full.dropna(subset=sm_cols + ["v7_net"]).reset_index(drop=True)
    log.info("\n  panel after smoothing: %d obs (out of %d)",
             len(panel_clean), len(panel_full))

    # ---- v7 P&L summary ----
    log.info("\n=== v7 P&L on full history (Tier A+B, K=4 M=1, gate, 0.8 bps) ===")
    s = panel_clean["v7_net"]
    sh = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
    log.info("  n=%d  Sh=%+.2f  net=%+.2f bps/cyc  total=%+.1f bps",
             len(s), sh, s.mean() * 1e4, s.sum() * 1e4)

    # ---- per-year v7 Sharpe ----
    log.info("\n=== v7 per-year Sharpe ===")
    pp = panel_clean.copy()
    pp["year"] = pp["ts"].dt.year
    log.info("  %-6s %5s %10s", "year", "n", "Sharpe")
    for y, g in pp.groupby("year"):
        if len(g) < 5: continue
        s = g["v7_net"]
        yr_sh = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        log.info("  %-6d %5d %+10.2f", y, len(g), yr_sh)

    # ---- lead-lag IC ----
    log.info("\n=== Lead-lag Pearson IC: indicator(t) vs v7 forward cumulative net_alpha ===")
    log.info("  %-22s %10s %10s %10s",
             "indicator (22d sm)", "lead 0", "lead 5", "lead 22")
    leads = [0, 5, 22]
    for ind in indicators:
        ic = lead_lag_ic(panel_clean, ind + "_sm", leads)
        log.info("  %-22s %+10.4f %+10.4f %+10.4f",
                 ind,
                 ic[0][0] if not np.isnan(ic[0][0]) else 0,
                 ic[5][0] if not np.isnan(ic[5][0]) else 0,
                 ic[22][0] if not np.isnan(ic[22][0]) else 0)

    # ---- conditional Sharpe by indicator quartile ----
    log.info("\n=== Conditional v7 Sharpe by indicator quartile (Q1 = lowest, Q4 = highest) ===")
    log.info("  %-22s %10s %10s %10s %10s   Q4-Q1",
             "indicator (22d sm)", "Q1 v7Sh", "Q2 v7Sh", "Q3 v7Sh", "Q4 v7Sh")
    for ind in indicators:
        cs = conditional_sharpe(panel_clean, ind + "_sm", fwd_window=22)
        if not cs:
            log.info("  %-22s    insufficient data", ind); continue
        sh1 = cs.get(1, {}).get("v7_sh", np.nan)
        sh2 = cs.get(2, {}).get("v7_sh", np.nan)
        sh3 = cs.get(3, {}).get("v7_sh", np.nan)
        sh4 = cs.get(4, {}).get("v7_sh", np.nan)
        spread = sh4 - sh1 if not (np.isnan(sh4) or np.isnan(sh1)) else np.nan
        log.info("  %-22s %+10.2f %+10.2f %+10.2f %+10.2f   %+8.2f",
                 ind, sh1, sh2, sh3, sh4, spread)

    # ---- v7 drawdown identification ----
    log.info("\n=== v7 drawdown clusters (rolling-22d Sharpe < 0 spans) ===")
    p = panel_clean.copy()
    p["roll22_sh"] = (p["v7_net"].rolling(22, min_periods=10).mean()
                       / p["v7_net"].rolling(22, min_periods=10).std()
                       * np.sqrt(252))
    p["dd_flag"] = (p["roll22_sh"] < 0).astype(int)
    # Compress consecutive dd_flag spans
    p["span_id"] = (p["dd_flag"].diff() != 0).cumsum()
    spans = p.groupby("span_id").agg(
        flag=("dd_flag", "first"),
        start_ts=("ts", "first"),
        end_ts=("ts", "last"),
        n=("v7_net", "size"),
        cum_pnl=("v7_net", "sum"),
    ).reset_index()
    dd_spans = spans[(spans["flag"] == 1) & (spans["n"] >= 5)]
    log.info("  identified %d v7 drawdown spans (≥5 cycles, rolling22d Sh<0)",
             len(dd_spans))
    log.info("  %-12s %-12s %5s %10s",
             "start", "end", "n", "cum bps")
    for _, r in dd_spans.iterrows():
        log.info("  %-12s %-12s %5d %+10.2f",
                 r["start_ts"].strftime("%Y-%m-%d"),
                 r["end_ts"].strftime("%Y-%m-%d"),
                 r["n"], r["cum_pnl"] * 1e4)

    # ---- indicator levels in drawdowns vs normal periods ----
    log.info("\n=== Indicator levels: drawdown periods vs normal periods (mean ± std) ===")
    log.info("  %-22s %18s %18s %10s",
             "indicator", "drawdown", "normal", "Z-diff")
    p_dd = p[p["dd_flag"] == 1]
    p_normal = p[p["dd_flag"] == 0]
    for ind in indicators:
        col = ind + "_sm"
        if col not in p.columns: continue
        dd_mean, dd_std = p_dd[col].mean(), p_dd[col].std()
        nm_mean, nm_std = p_normal[col].mean(), p_normal[col].std()
        # Z-score difference: how many SDs apart are the means
        pooled_std = np.sqrt((dd_std**2 + nm_std**2) / 2) if not (np.isnan(dd_std) or np.isnan(nm_std)) else np.nan
        z = (dd_mean - nm_mean) / pooled_std if pooled_std > 0 else 0
        log.info("  %-22s %+8.4f±%6.4f %+8.4f±%6.4f %+10.2f",
                 ind, dd_mean, dd_std, nm_mean, nm_std, z)


if __name__ == "__main__":
    main()
