"""Stage 1 IC probe: alpha-residual on Hyperliquid xyz US-equity perps.

Question: do the same kind of cross-sectional features that work on the
25-symbol crypto basket (alpha_v4_xs / v6_clean) show any IC against
forward residual returns on xyz US equities?

Setup:
  - Universe: top liquid US equities on the xyz dex (~18 names)
  - Beta anchor: xyz:SP500 (built-in equity index perp on the same dex)
  - Bars: 1h, last ~6 months (~4000 bars per symbol — full history available)
  - Residual: ret_s,t - beta_s,t * ret_SP500,t, beta from trailing 7d (168 bar)
    rolling regression, .shift(1) to be point-in-time
  - Forward target: h=48 bar (~48h) forward residual return
  - Features (all .shift(1)):
      mom_24      : trailing 24h log return
      mom_resid_24: trailing 24h sum of residual returns
      reversal    : -mom_24 / vol_24
      rsi_proxy   : fraction of last 24 bars with positive return
      vol_z       : 24h vol z-scored vs trailing 7d
  - Each feature is cross-sectionally ranked per timestamp before IC
  - IC = per-timestamp Spearman(rank(feat), rank(fwd_resid_h)), averaged
  - 95% CI from block bootstrap (7-day blocks)
  - Reported separately for full session and US-RTH-only bars

NOT a backtest, NOT a model fit. Just IC sanity. Cost / Sharpe come later.
"""
from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---- config -------------------------------------------------------------

API_URL = "https://api.hyperliquid.xyz/info"
INTERVAL = "1h"
DAYS_BACK = 180
HORIZON_BARS = 48          # ~48h forward residual
BETA_WINDOW_BARS = 168     # 7d rolling beta
MOM_WINDOW = 24            # 1d momentum / vol
VOL_WIN_LONG = 168         # 7d for vol z-score
EMBARGO_BARS = 24          # gap between feature ts and label ts beyond horizon
BOOT_N = 1000
BLOCK_DAYS = 7

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Top US equities on xyz by daily volume (queried 2026-05-06)
EQUITY_UNIVERSE = [
    "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META", "COIN", "MSTR",
    "PLTR", "INTC", "MU", "SNDK", "HOOD", "AAPL", "MSFT", "ORCL",
    "NFLX", "CRCL",
]
# Compared anchors:
#   SP500   - broad market perp, 49d history (short)
#   XYZ100  - HL's tech-100 basket perp, 206d history (longest)
#   BASKET  - equal-weighted mean return of EQUITY_UNIVERSE itself (no perp needed)
EXTERNAL_ANCHORS = ["SP500", "XYZ100"]
ANCHORS = ["SP500", "XYZ100", "BASKET"]


# ---- data fetch ---------------------------------------------------------

def _fetch_chunk(coin: str, start_ms: int, end_ms: int) -> list[dict]:
    payload = {
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": INTERVAL,
                "startTime": start_ms, "endTime": end_ms},
    }
    r = requests.post(API_URL, json=payload, timeout=30)
    r.raise_for_status()
    d = r.json()
    return d if isinstance(d, list) else []


def fetch_xyz(symbol: str, days_back: int = DAYS_BACK) -> pd.DataFrame:
    """Fetch 1h candles for xyz:<symbol>. Cached to parquet."""
    cache = CACHE_DIR / f"xyz_{symbol}_{INTERVAL}.parquet"
    coin = f"xyz:{symbol}"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days_back * 24 * 3600 * 1000

    # paginate 5000 bars at a time (HL limit)
    all_rows: list[dict] = []
    cursor = start_ms
    step_ms = 4900 * 60 * 60 * 1000  # 4900 1h bars per request
    while cursor < end_ms:
        chunk = _fetch_chunk(coin, cursor, min(cursor + step_ms, end_ms))
        if not chunk:
            cursor += step_ms
            continue
        all_rows.extend(chunk)
        cursor = chunk[-1]["t"] + 60 * 60 * 1000
        time.sleep(0.05)

    if not all_rows:
        log.warning("no data for %s", coin)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume", "n": "n_trades"})
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["symbol"] = symbol
    df = df[["ts", "symbol", "open", "high", "low", "close", "volume", "n_trades"]]
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df.to_parquet(cache)
    return df


def load_panel() -> pd.DataFrame:
    """Long-format panel of all symbols + external beta anchors."""
    frames = []
    all_syms = EQUITY_UNIVERSE + EXTERNAL_ANCHORS
    for sym in all_syms:
        df = fetch_xyz(sym)
        if df.empty:
            continue
        log.info("  %-7s n=%d  first=%s  last=%s",
                 sym, len(df),
                 df["ts"].iloc[0].strftime("%Y-%m-%d"),
                 df["ts"].iloc[-1].strftime("%Y-%m-%d"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- features -----------------------------------------------------------

def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel["ret"] = (panel.groupby("symbol")["close"]
                    .transform(lambda s: np.log(s / s.shift(1))))
    return panel


def _anchor_returns(panel: pd.DataFrame, anchor: str) -> pd.Series:
    """Return a Series indexed by ts giving the anchor's return at each bar."""
    if anchor in EXTERNAL_ANCHORS:
        sub = panel[panel["symbol"] == anchor][["ts", "ret"]]
        return sub.set_index("ts")["ret"]
    if anchor == "BASKET":
        eq = panel[panel["symbol"].isin(EQUITY_UNIVERSE)]
        return eq.groupby("ts")["ret"].mean()
    raise ValueError(anchor)


def residualize(panel: pd.DataFrame, anchor: str) -> pd.DataFrame:
    """Add ret_anchor, beta_<anchor>, resid_<anchor> columns. Beta is rolling
    7d cov/var, .shift(1) so it's point-in-time."""
    anchor_ret = _anchor_returns(panel, anchor).rename("ret_anchor")
    out = panel.merge(anchor_ret, left_on="ts", right_index=True, how="left")

    def _beta(g: pd.DataFrame) -> pd.Series:
        cov = g["ret"].rolling(BETA_WINDOW_BARS).cov(g["ret_anchor"])
        var = g["ret_anchor"].rolling(BETA_WINDOW_BARS).var()
        return (cov / var).shift(1)

    out = out.sort_values(["symbol", "ts"]).reset_index(drop=True)
    out["beta"] = (out.groupby("symbol", group_keys=False)
                   .apply(_beta).values)
    out["resid"] = out["ret"] - out["beta"] * out["ret_anchor"]
    return out


def anchor_diagnostics(panel: pd.DataFrame, anchor: str) -> dict:
    """Report median beta, IQR, and var(resid)/var(ret) for an anchor."""
    sub = panel[(panel["symbol"].isin(EQUITY_UNIVERSE))
                & panel["beta"].notna() & panel["resid"].notna()]
    if sub.empty:
        return {"anchor": anchor, "n_bars": 0}
    var_ratio = (sub.groupby("symbol")["resid"].var()
                 / sub.groupby("symbol")["ret"].var()).median()
    return {
        "anchor": anchor,
        "n_bars": sub["ts"].nunique(),
        "median_beta": sub["beta"].median(),
        "beta_iqr_lo": sub["beta"].quantile(0.25),
        "beta_iqr_hi": sub["beta"].quantile(0.75),
        "var_ratio": var_ratio,
    }


def add_anchor_independent_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features that only use raw return (not anchor-dependent)."""
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)
    panel["mom_24"] = g["ret"].apply(
        lambda s: s.rolling(MOM_WINDOW).sum().shift(1))
    panel["vol_24"] = g["ret"].apply(
        lambda s: s.rolling(MOM_WINDOW).std().shift(1))
    panel["reversal"] = -panel["mom_24"] / panel["vol_24"]
    panel["rsi_proxy"] = g["ret"].apply(
        lambda s: (s > 0).rolling(MOM_WINDOW).mean().shift(1))
    vol_long = g["ret"].apply(lambda s: s.rolling(VOL_WIN_LONG).std().shift(1))
    vol_long_mean = g["ret"].apply(
        lambda s: s.rolling(VOL_WIN_LONG).std().rolling(VOL_WIN_LONG).mean().shift(1))
    vol_long_std = g["ret"].apply(
        lambda s: s.rolling(VOL_WIN_LONG).std().rolling(VOL_WIN_LONG).std().shift(1))
    panel["vol_z"] = (vol_long - vol_long_mean) / vol_long_std
    return panel


def add_anchor_features_and_label(panel: pd.DataFrame) -> pd.DataFrame:
    """mom_resid_24 (feature) and fwd_resid_h (label) depend on which anchor
    was used to compute `resid`. Caller must residualize() first."""
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)
    panel["mom_resid_24"] = g["resid"].apply(
        lambda s: s.rolling(MOM_WINDOW).sum().shift(1))
    panel["fwd_resid_h"] = (g["resid"].apply(
        lambda s: s.rolling(HORIZON_BARS).sum().shift(-HORIZON_BARS))).values
    return panel


# ---- IC + bootstrap -----------------------------------------------------

def cross_sectional_rank(panel: pd.DataFrame, col: str) -> pd.Series:
    return panel.groupby("ts")[col].rank(pct=True)


def per_ts_ic(panel: pd.DataFrame, feat_col: str) -> pd.Series:
    """Spearman IC between feature and fwd_resid_h, computed per timestamp.
    Caller is responsible for restricting `panel` to the equity universe."""
    sub = panel.dropna(subset=[feat_col, "fwd_resid_h"])

    def _ic(g: pd.DataFrame) -> float:
        if len(g) < 4:
            return np.nan
        r, _ = spearmanr(g[feat_col], g["fwd_resid_h"])
        return r

    return sub.groupby("ts").apply(_ic).dropna()


def block_bootstrap_ci(daily_ic: pd.Series, n: int = BOOT_N,
                       block_days: int = BLOCK_DAYS) -> tuple[float, float, float]:
    """Stationary block bootstrap on a daily series of mean IC."""
    daily = daily_ic.resample("1D").mean().dropna()
    arr = daily.values
    if len(arr) < block_days * 2:
        return float(arr.mean()), np.nan, np.nan
    n_blocks = max(1, len(arr) // block_days)
    rng = np.random.default_rng(42)
    means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(arr) - block_days + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_days] for s in starts])
        means[i] = sample.mean()
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---- session mask -------------------------------------------------------

def is_us_rth(ts: pd.Series) -> pd.Series:
    """US regular trading hours proxy: weekdays 14:00-21:00 UTC.
    (Covers 9:30-16:00 ET in EDT and most of EST cash session.)"""
    ts = pd.DatetimeIndex(ts)
    return (ts.dayofweek < 5) & (ts.hour >= 14) & (ts.hour < 21)


# ---- main ---------------------------------------------------------------

FEATURE_COLS = ["mom_24", "mom_resid_24", "reversal", "rsi_proxy", "vol_z"]


def report_ic(panel: pd.DataFrame, label: str) -> None:
    eq_only = panel[panel["symbol"].isin(EQUITY_UNIVERSE)].copy()
    log.info("--- IC: %s  (n_ts=%d, n_symbols=%d) ---",
             label, eq_only["ts"].nunique(), eq_only["symbol"].nunique())
    log.info("  %-15s  %8s  %8s  %8s  %6s",
             "feature", "IC", "lo95", "hi95", "n_ts")
    for f in FEATURE_COLS:
        feat_rank_col = f"{f}_rank"
        eq_only[feat_rank_col] = cross_sectional_rank(eq_only, f)
        ic = per_ts_ic(eq_only, feat_rank_col)
        if len(ic) == 0:
            log.info("  %-15s  %8s  %8s  %8s  %6d",
                     f, "n/a", "n/a", "n/a", 0)
            continue
        mu, lo, hi = block_bootstrap_ci(ic)
        log.info("  %-15s  %+8.4f  %+8.4f  %+8.4f  %6d",
                 f, mu, lo, hi, len(ic))


def run_anchor(base_panel: pd.DataFrame, anchor: str) -> None:
    log.info("")
    log.info("=" * 72)
    log.info("ANCHOR = %s", anchor)
    log.info("=" * 72)
    panel = residualize(base_panel.copy(), anchor)
    panel = add_anchor_features_and_label(panel)
    diag = anchor_diagnostics(panel, anchor)
    log.info("residualization: n_bars=%d  median_beta=%.2f  IQR=[%.2f,%.2f]  "
             "var(resid)/var(ret)=%.2f  (lower=better hedge)",
             diag["n_bars"], diag["median_beta"],
             diag["beta_iqr_lo"], diag["beta_iqr_hi"], diag["var_ratio"])
    panel["is_rth"] = is_us_rth(panel["ts"])
    report_ic(panel, "FULL SESSION (24/7)")
    report_ic(panel[panel["is_rth"]].copy(),
              "US RTH only (Mon-Fri 14-21 UTC)")


def main() -> None:
    log.info("loading xyz panel: %d equities + %d external anchors, %d days, %s bars",
             len(EQUITY_UNIVERSE), len(EXTERNAL_ANCHORS), DAYS_BACK, INTERVAL)
    panel = load_panel()
    if panel.empty:
        log.error("no data fetched")
        return
    panel = add_returns(panel)
    panel = add_anchor_independent_features(panel)
    for anchor in ANCHORS:
        run_anchor(panel, anchor)


if __name__ == "__main__":
    main()
