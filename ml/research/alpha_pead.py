"""Post-Earnings-Announcement-Drift (PEAD) probe on S&P 100 universe.

Tests the documented equity anomaly that names with positive earnings
surprises drift up for 30-60 days, names with negative surprises drift down.

Setup:
  Universe: S&P 100 names with full data + earnings calendar (~80-95 after
            filtering for >5y history and ≥4 earnings events).
  Data: yfinance daily, 2013-01-01 to today.
  Anchor: equal-weight in-universe basket (residualization).
  Signal: PEAD score for symbol s at day t =
            surprise_pct(s, last_earnings_event(s, t))
            if 1 <= days_since_earnings(s, t) <= max_drift_days else NaN.
  Universe at each day t: subset of names with valid PEAD signal.
  Strategy: long top-K=5 positive surprises, short bottom-K=5 negative surprises
            from valid sub-universe. Rebalance daily, hold 1 day.

Cost model:
  Turnover-based: per-day turnover × per-trade cost. Sticky PEAD signals
  mean low daily turnover → low effective cost. Compare to v6_clean's
  fixed 24 bps per rebalance which over-counts for sticky strategies.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"

BETA_WINDOW = 60                    # 60-day rolling beta vs basket
MAX_DRIFT_DAYS = 60                 # PEAD signal alive 1..60 days post earnings
TOP_K = 5                           # 5 long, 5 short out of valid sub-universe
COST_PER_TRADE_BPS = 5              # 5 bps single-side per name turnover


# ---- features + label --------------------------------------------------

def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel["ret"] = (panel.groupby("symbol")["close"]
                    .transform(lambda s: np.log(s / s.shift(1))))
    return panel


def add_loo_basket(panel: pd.DataFrame) -> pd.DataFrame:
    grp_ts = panel.groupby("ts")["ret"]
    total = grp_ts.transform("sum")
    n = grp_ts.transform("count")
    panel["bk_ret"] = (total - panel["ret"].fillna(0)) / (n - 1).replace(0, np.nan)
    return panel


def add_residual(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    def _beta(g):
        cov = (g["ret"] * g["bk_ret"]).rolling(BETA_WINDOW).mean() - \
              g["ret"].rolling(BETA_WINDOW).mean() * g["bk_ret"].rolling(BETA_WINDOW).mean()
        var = g["bk_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        return (cov / var).clip(-5, 5).shift(1)
    panel["beta"] = panel.groupby("symbol", group_keys=False).apply(_beta).values
    panel["resid"] = panel["ret"] - panel["beta"] * panel["bk_ret"]
    panel["fwd_resid_1d"] = panel.groupby("symbol", group_keys=False)["resid"].shift(-1)
    return panel


# ---- PEAD signal -------------------------------------------------------

def add_pead_signal(panel: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """For each (sym, day), find most recent earnings event and compute:
    - days_since_earnings (None if no past event)
    - surprise_pct (most recent past surprise)
    - pead_signal: surprise_pct if 1 <= days_since <= MAX_DRIFT_DAYS, else NaN
    """
    earn = earnings.copy()
    # normalize earnings ts to date (and ensure consistent dtype with panel)
    earn["ts"] = pd.to_datetime(earn["ts"], utc=True).dt.normalize().astype("datetime64[ns, UTC]")
    earn = earn.sort_values(["symbol", "ts"]).reset_index(drop=True)

    # asof merge per symbol — ensure panel ts is also datetime64[ns, UTC]
    panel = panel.copy()
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True).astype("datetime64[ns, UTC]")
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    out = []
    for sym, g in panel.groupby("symbol"):
        e = earn[earn["symbol"] == sym][["ts", "surprise_pct"]].dropna(subset=["surprise_pct"])
        if e.empty:
            g = g.copy()
            g["surprise_pct"] = np.nan
            g["earnings_ts"] = pd.NaT
            out.append(g)
            continue
        # asof merge: for each panel day, find latest earnings with ts < panel_day
        e_sorted = e.sort_values("ts").rename(columns={"ts": "earnings_ts"})
        merged = pd.merge_asof(
            g.sort_values("ts"),
            e_sorted,
            left_on="ts", right_on="earnings_ts",
            allow_exact_matches=False,  # earnings effective from t+1
            direction="backward",
        )
        out.append(merged)

    panel = pd.concat(out, ignore_index=True)
    panel["days_since_earnings"] = (
        (panel["ts"] - panel["earnings_ts"]).dt.days
    )
    valid = (panel["days_since_earnings"].between(1, MAX_DRIFT_DAYS, inclusive="both"))
    panel["pead_signal"] = panel["surprise_pct"].where(valid, np.nan)
    return panel


# ---- portfolio with turnover-based cost --------------------------------

def construct_pead_portfolio(panel: pd.DataFrame,
                              top_k: int = TOP_K,
                              cost_bps: float = COST_PER_TRADE_BPS) -> pd.DataFrame:
    """At each day: rank valid names by pead_signal, long top-K, short bottom-K.
    Track per-day turnover = total |weight change| across all names.
    Per-day cost = turnover * cost_bps."""
    panel = panel.dropna(subset=["pead_signal", "fwd_resid_1d"]).copy()
    if panel.empty:
        return pd.DataFrame()

    rows = []
    prev_long: set = set()
    prev_short: set = set()

    for ts, bar in panel.groupby("ts"):
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values("pead_signal")
        long_leg = set(bar.tail(top_k)["symbol"])
        short_leg = set(bar.head(top_k)["symbol"])

        # turnover: # names entering or exiting each leg, normalized
        long_changes = len(long_leg.symmetric_difference(prev_long))
        short_changes = len(short_leg.symmetric_difference(prev_short))
        # each "change" is one buy + one sell of weight 1/k, times two
        total_turnover = (long_changes + short_changes) / (2 * top_k)
        cost = total_turnover * cost_bps / 1e4

        # spread alpha = mean fwd return of long minus mean of short
        long_alpha = bar[bar["symbol"].isin(long_leg)]["fwd_resid_1d"].mean()
        short_alpha = bar[bar["symbol"].isin(short_leg)]["fwd_resid_1d"].mean()
        spread = long_alpha - short_alpha
        rows.append({
            "ts": ts,
            "n_universe": len(bar),
            "spread_alpha": spread,
            "long_alpha": long_alpha,
            "short_alpha": short_alpha,
            "turnover": total_turnover,
            "cost": cost,
            "net_alpha": spread - cost,
        })
        prev_long, prev_short = long_leg, short_leg

    return pd.DataFrame(rows)


def metrics(pnl: pd.DataFrame) -> dict:
    if pnl.empty:
        return {"n": 0}
    n = len(pnl)
    g_sh = (pnl["spread_alpha"].mean() / pnl["spread_alpha"].std()
            * np.sqrt(252)) if pnl["spread_alpha"].std() > 0 else 0
    n_sh = (pnl["net_alpha"].mean() / pnl["net_alpha"].std()
            * np.sqrt(252)) if pnl["net_alpha"].std() > 0 else 0
    return {
        "n": n,
        "gross_bps_per_day": pnl["spread_alpha"].mean() * 1e4,
        "net_bps_per_day": pnl["net_alpha"].mean() * 1e4,
        "annual_cost_bps": pnl["cost"].mean() * 1e4 * 252,
        "avg_turnover_pct_per_day": pnl["turnover"].mean() * 100,
        "avg_universe_size": pnl["n_universe"].mean(),
        "gross_sharpe": g_sh,
        "net_sharpe": n_sh,
        "hit_rate": float((pnl["spread_alpha"] > 0).mean()),
    }


def bootstrap_ci(pnl: pd.DataFrame, block_days: int = 30,
                 n_boot: int = 2000) -> tuple[float, float]:
    if pnl.empty or len(pnl) < block_days * 2:
        return np.nan, np.nan
    arr = pnl["net_alpha"].values
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
    log.info("loading S&P 100 universe...")
    panel, earnings, surv = load_universe()
    if panel.empty:
        log.error("no universe data")
        return

    log.info("universe: %d names, %d daily rows, %d earnings events",
             len(surv), len(panel), len(earnings))

    panel = add_returns(panel)
    panel = add_loo_basket(panel)
    panel = add_residual(panel)
    panel = add_pead_signal(panel, earnings)

    log.info("residualization sanity: median beta=%.2f IQR=[%.2f,%.2f]",
             panel["beta"].median(),
             panel["beta"].quantile(0.25), panel["beta"].quantile(0.75))

    n_total = len(panel)
    n_pead_valid = panel["pead_signal"].notna().sum()
    log.info("PEAD valid (sym, day) pairs: %d / %d (%.1f%%)",
             n_pead_valid, n_total, 100 * n_pead_valid / n_total)

    pnl = construct_pead_portfolio(panel, top_k=TOP_K,
                                    cost_bps=COST_PER_TRADE_BPS)
    if pnl.empty:
        log.error("portfolio empty")
        return

    log.info("\n=== PEAD strategy results (top_k=%d, cost=%.0f bps/trade-side) ===",
             TOP_K, COST_PER_TRADE_BPS)
    m = metrics(pnl)
    lo, hi = bootstrap_ci(pnl)
    log.info("  n_days                = %d", m["n"])
    log.info("  avg_universe_size     = %.1f names with valid PEAD signal", m["avg_universe_size"])
    log.info("  avg_turnover_per_day  = %.1f%%", m["avg_turnover_pct_per_day"])
    log.info("  annualized cost       = %.0f bps/year", m["annual_cost_bps"])
    log.info("  gross alpha / day     = %+.2f bps", m["gross_bps_per_day"])
    log.info("  net alpha / day       = %+.2f bps", m["net_bps_per_day"])
    log.info("  hit rate              = %.0f%%", 100 * m["hit_rate"])
    log.info("  gross Sharpe (annu)   = %+.2f", m["gross_sharpe"])
    log.info("  net Sharpe (annu)     = %+.2f  [%+.2f, %+.2f] 95%% CI",
             m["net_sharpe"], lo, hi)

    # Per-year breakdown
    log.info("\n=== Per-year ===")
    pnl_year = pnl.copy()
    pnl_year["year"] = pnl_year["ts"].dt.year
    log.info("  %-6s %5s %12s %12s %10s %8s",
             "year", "n", "gross/d", "net/d", "net_Sh", "hit")
    for y, g in pnl_year.groupby("year"):
        m = metrics(g)
        log.info("  %-6d %5d %+10.2fbps %+10.2fbps %+8.2f %7.0f%%",
                 y, m["n"], m["gross_bps_per_day"], m["net_bps_per_day"],
                 m["net_sharpe"], 100 * m["hit_rate"])

    # Cost sensitivity
    log.info("\n=== Cost sensitivity ===")
    log.info("  %-15s %-15s %-12s %-15s",
             "cost/trade(bps)", "annu_cost(bps)", "net_Sh", "95% CI")
    for c in (0, 2, 5, 10, 20):
        pnl_c = construct_pead_portfolio(panel, top_k=TOP_K, cost_bps=c)
        if pnl_c.empty:
            continue
        m = metrics(pnl_c)
        lo, hi = bootstrap_ci(pnl_c)
        log.info("  %-15d %+13.0f %+12.2f  [%+.2f, %+.2f]",
                 c, m["annual_cost_bps"], m["net_sharpe"], lo, hi)


if __name__ == "__main__":
    main()
