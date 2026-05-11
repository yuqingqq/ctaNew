"""Stage 1 IC probe: 30min cross-sectional residual on Polygon 5m × 2y.

Question: does intraday alpha exist at the 30min cross-sectional residual level
on the 11-name xyz tier_ab universe? If yes (mean per-bar rank IC > +0.04),
proceed to full walk-forward backtest. If no, abandon.

Single-pass per-bar IC. NOT a backtest. NOT a model fit. Just sanity.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"

TIER_AB = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "MU", "NFLX", "NVDA",
           "ORCL", "PLTR", "TSLA"]

# Use full S&P 100 for basket residualization. We'll filter to tier_ab for IC.
SP100_TICKERS_QUICK = sorted({p.name.split("_")[1] for p in CACHE.glob("poly_*_5m.parquet")})

# RTH window in UTC (EDT 09:30-16:00 = UTC 13:30-20:00 most of the year;
# during EST it's UTC 14:30-21:00. Polygon timestamps are UTC. We'll filter
# by hour-of-day to capture both DST regimes.)
def is_rth_utc(ts: pd.Series) -> pd.Series:
    """RTH in UTC roughly 13:30-21:00. We keep the wider band."""
    minutes_utc = ts.dt.hour * 60 + ts.dt.minute
    return (minutes_utc >= 13 * 60 + 30) & (minutes_utc <= 21 * 60)


def load_all_5m() -> pd.DataFrame:
    log.info("loading Polygon 5m for %d names...", len(SP100_TICKERS_QUICK))
    rows = []
    for sym in SP100_TICKERS_QUICK:
        path = CACHE / f"poly_{sym}_5m.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        # Some caches might lack symbol col
        if "symbol" not in df.columns:
            df["symbol"] = sym
        rows.append(df[["ts", "symbol", "open", "high", "low", "close", "volume", "vwap"]])
    if not rows:
        raise RuntimeError("no Polygon 5m data found")
    panel = pd.concat(rows, ignore_index=True)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    log.info("  loaded %d rows, %d symbols, range %s → %s",
             len(panel), panel["symbol"].nunique(),
             panel["ts"].min(), panel["ts"].max())
    return panel


def resample_to_30m(panel5m: pd.DataFrame) -> pd.DataFrame:
    log.info("resampling 5m → 30min, keeping RTH only...")
    # Filter RTH first (saves work)
    panel5m = panel5m[is_rth_utc(panel5m["ts"])].copy()
    log.info("  RTH 5m bars: %d", len(panel5m))

    rows = []
    for sym, g in panel5m.groupby("symbol"):
        g = g.set_index("ts").sort_index()
        agg = g.resample("30min", origin="start_day").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum",
        }).dropna(subset=["close"])
        agg["symbol"] = sym
        agg = agg.reset_index()
        # Re-filter RTH at 30min level (resample can produce off-RTH bars at boundaries)
        agg = agg[is_rth_utc(agg["ts"])]
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["symbol", "ts"]).reset_index(drop=True)
    log.info("  30min bars: %d (%d symbols)", len(out), out["symbol"].nunique())
    return out


def compute_features_and_target(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Returns (panel_with_features, feature_cols). Includes fwd_resid_30m target."""
    log.info("computing returns, basket, residual, features, target...")
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)

    panel["ret_30m"] = g["close"].apply(lambda s: np.log(s / s.shift(1)))

    # Per-ts equal-weight basket (leave-one-out)
    grp_ts = panel.groupby("ts")["ret_30m"]
    total = grp_ts.transform("sum")
    n = grp_ts.transform("count")
    panel["bk_ret_30m"] = (total - panel["ret_30m"].fillna(0)) / (n - 1).replace(0, np.nan)

    # Rolling beta (60 bars ≈ 30 RTH hours ≈ 4-5 days), shifted
    BETA_BARS = 60
    def _beta(group):
        cov = ((group["ret_30m"] * group["bk_ret_30m"]).rolling(BETA_BARS).mean()
                - group["ret_30m"].rolling(BETA_BARS).mean() * group["bk_ret_30m"].rolling(BETA_BARS).mean())
        var = group["bk_ret_30m"].rolling(BETA_BARS).var().replace(0, np.nan)
        return (cov / var).clip(-5, 5).shift(1)
    panel["beta"] = g.apply(_beta).values
    panel["resid_30m"] = panel["ret_30m"] - panel["beta"] * panel["bk_ret_30m"]

    # Forward 30min residual (1-bar-ahead), shifted
    panel["fwd_resid_30m"] = g["resid_30m"].apply(lambda s: s.shift(-1))

    # ===== Features (all .shift(1) for PIT) =====
    feats = []

    # Multi-horizon trailing returns
    for h_name, h_bars in [("30m", 1), ("1h", 2), ("2h", 4), ("4h", 8)]:
        col = f"f_ret_{h_name}"
        panel[col] = g["ret_30m"].apply(lambda s, h=h_bars: s.rolling(h).sum().shift(1))
        feats.append(col)

    # Multi-horizon trailing residual returns
    for h_name, h_bars in [("30m", 1), ("1h", 2), ("4h", 8)]:
        col = f"f_idio_{h_name}"
        panel[col] = g["resid_30m"].apply(lambda s, h=h_bars: s.rolling(h).sum().shift(1))
        feats.append(col)

    # Realized volatility (1h, 4h windows)
    for h_name, h_bars in [("1h", 2), ("4h", 8), ("1d", 13)]:
        col = f"f_vol_{h_name}"
        panel[col] = g["ret_30m"].apply(lambda s, h=h_bars: s.rolling(h).std().shift(1))
        feats.append(col)

    # VWAP-distance proxy: distance from session-start (using day's first close as ref)
    panel["date"] = panel["ts"].dt.date
    sess_close = panel.groupby(["symbol", "date"])["close"].transform("first")
    panel["f_dist_from_open"] = ((panel["close"] - sess_close) / sess_close).shift(1)
    g2 = panel.groupby("symbol", group_keys=False)
    panel["f_dist_from_open"] = g2["f_dist_from_open"].apply(lambda s: s)  # ensure groupby refreshed
    feats.append("f_dist_from_open")

    # Time of day (RTH minutes since 13:30 UTC)
    panel["f_minutes_since_open"] = (
        (panel["ts"].dt.hour * 60 + panel["ts"].dt.minute) - (13 * 60 + 30)
    ).astype(float)
    feats.append("f_minutes_since_open")

    # Volume z-score (vs same-time-of-day past 20 sessions)
    panel["log_vol"] = np.log1p(panel["volume"])
    # Simple rolling z-score over last 60 same-time-of-day samples
    panel["f_vol_z"] = (
        panel.groupby(["symbol"])["log_vol"]
        .transform(lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std().replace(0, np.nan))
    ).shift(1)
    feats.append("f_vol_z")

    # Volatility z-score (recent vol vs 1-day vol)
    panel["f_vol_z_1h_vs_1d"] = (panel["f_vol_1h"] / panel["f_vol_1d"].replace(0, np.nan))
    feats.append("f_vol_z_1h_vs_1d")

    log.info("  features: %s", feats)
    log.info("  panel: %d rows", len(panel))
    return panel, feats


def per_bar_rank_ic(df: pd.DataFrame, feat: str, target: str = "fwd_resid_30m"
                     ) -> tuple[float, int]:
    """Mean per-bar Spearman IC across all bars with ≥3 valid (feat, target) pairs."""
    valid = df.dropna(subset=[feat, target])
    if valid.empty: return float("nan"), 0
    ics = []
    for ts, g in valid.groupby("ts"):
        if len(g) < 3: continue
        ic = g[feat].rank().corr(g[target].rank())
        if not np.isnan(ic):
            ics.append(ic)
    return (float(np.mean(ics)) if ics else float("nan")), len(ics)


def main() -> None:
    panel5m = load_all_5m()
    panel = resample_to_30m(panel5m)
    panel, feats = compute_features_and_target(panel)

    # Filter to tier_ab universe for IC measurement
    sub = panel[panel["symbol"].isin(TIER_AB)].copy()
    log.info("\ntier_ab subset: %d rows, %d symbols",
             len(sub), sub["symbol"].nunique())

    log.info("\n=== Per-feature IC (mean per-bar Spearman rank IC, 11 names) ===")
    log.info(f"{'feature':<22} {'IC':>8} {'n_bars':>8}")
    feature_results = []
    for f in feats:
        ic, n_bars = per_bar_rank_ic(sub, f)
        feature_results.append((f, ic, n_bars))
        log.info(f"{f:<22} {ic:>+8.4f} {n_bars:>8d}")

    # Combined: simple sum of standardized features as a "naive" composite
    # Normalize each feature per-bar (cross-sectional z), then sum
    for f, _, _ in feature_results:
        sub[f"{f}_z"] = sub.groupby("ts")[f].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0
        )
    sub["composite"] = sub[[f"{f}_z" for f, _, _ in feature_results]].sum(axis=1)
    ic_comp, n_comp = per_bar_rank_ic(sub, "composite")
    log.info(f"\n  COMPOSITE (sum of z-scores)  IC={ic_comp:+.4f}  n_bars={n_comp}")

    # IC by time-of-day
    log.info("\n=== IC by time-of-day bin ===")
    log.info(f"{'minutes_after_open':<22} {'IC':>8} {'n_bars':>8}")
    sub["tod_bin"] = (sub["f_minutes_since_open"] // 60).astype(int)  # hourly bins
    for tod, g in sub.groupby("tod_bin"):
        ic, n_bars = per_bar_rank_ic(g, "composite")
        log.info(f"  bin={tod:>2d} (≈ {tod}h after open)  IC={ic:>+.4f}  n_bars={n_bars:>4d}")

    # IC over time (calendar quarters)
    log.info("\n=== IC by calendar quarter ===")
    sub["q"] = sub["ts"].dt.to_period("Q")
    log.info(f"{'quarter':<10} {'composite_IC':>15} {'n_bars':>8}")
    for q, g in sub.groupby("q"):
        ic, n_bars = per_bar_rank_ic(g, "composite")
        log.info(f"  {str(q):<10} {ic:>+15.4f} {n_bars:>8d}")

    # Top-K dispersion test: among the 5 strongest by composite, what's mean fwd?
    # This is the actual portfolio P&L proxy at 30min cadence.
    log.info("\n=== Long-short top-3 / bot-3 by composite, no cost, no gate ===")
    valid_sub = sub.dropna(subset=["composite", "fwd_resid_30m"])
    spreads = []
    for ts, g in valid_sub.groupby("ts"):
        if len(g) < 9: continue
        g = g.sort_values("composite", ascending=False)
        long_a = g.head(3)["fwd_resid_30m"].mean()
        short_a = g.tail(3)["fwd_resid_30m"].mean()
        spreads.append((ts, long_a - short_a))
    if spreads:
        sp_df = pd.DataFrame(spreads, columns=["ts", "spread"])
        # Annualize: 13 bars × 252 days ≈ 3276 30min bars per year
        bars_per_yr = 13 * 252
        ann_ret_pct = sp_df["spread"].mean() * bars_per_yr * 100
        ann_sh = sp_df["spread"].mean() / sp_df["spread"].std() * np.sqrt(bars_per_yr)
        cum_bps = sp_df["spread"].sum() * 1e4
        hit = (sp_df["spread"] > 0).mean() * 100
        log.info(f"  n_bars={len(sp_df)}  mean_spread={sp_df['spread'].mean()*1e4:+.4f} bps  "
                 f"std={sp_df['spread'].std()*1e4:.2f} bps")
        log.info(f"  cum gross={cum_bps:+.1f} bps  hit={hit:.0f}%  "
                 f"ann_ret={ann_ret_pct:+.2f}%  raw_Sharpe={ann_sh:+.2f}")

    log.info("\n=== Decision criterion ===")
    if ic_comp > 0.04:
        log.info("  composite IC > +0.04 → PROCEED to Stage 2 walk-forward backtest")
    elif ic_comp > 0.02:
        log.info("  composite IC %+.4f marginal → consider feature redesign before backtest", ic_comp)
    else:
        log.info("  composite IC %+.4f weak → ABANDON intraday redesign, look elsewhere", ic_comp)


if __name__ == "__main__":
    main()
