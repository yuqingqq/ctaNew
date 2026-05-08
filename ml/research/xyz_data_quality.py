"""xyz perp vs cash equity data-quality probe.

Question: are xyz US-equity perps faithful trackers of the actual cash
equities, and is the perp-vs-cash basis a real tradeable mean-reverting
signal? Before redesigning any ML system around equities, we first need to
know whether (a) the data is clean, (b) the perp tracks cash during RTH,
and (c) what the perp does when cash is closed.

Five liquid names: NVDA, TSLA, AMD, AAPL, AMZN.
Cash data: yfinance (free, no key).

Reports:
  Q1 daily-tracking : log(xyz_close / yf_close) at ~21:00 UTC over 60d
                      stats and correlation
  Q2 intraday basis : 1m-aligned perp vs cash during RTH (last 7d)
                      basis distribution, AC, half-life
  Q3 off-hours drift: xyz close-to-close vs cash close-to-open over weekends
  Q4 anomaly scan   : gaps, frozen prints, large 1m jumps in xyz
"""
from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_URL = "https://api.hyperliquid.xyz/info"
CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

NAMES = ["AAPL", "AMD", "AMZN", "COST", "GOOGL", "INTC", "LLY", "META",
         "MSFT", "MU", "NFLX", "NVDA", "ORCL", "PLTR", "TSLA"]


# ---- xyz fetch (1h and 1m) ---------------------------------------------

def _fetch(coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    payload = {"type": "candleSnapshot",
               "req": {"coin": coin, "interval": interval,
                       "startTime": start_ms, "endTime": end_ms}}
    r = requests.post(API_URL, json=payload, timeout=30)
    r.raise_for_status()
    d = r.json()
    return d if isinstance(d, list) else []


def fetch_xyz(symbol: str, interval: str, days_back: int) -> pd.DataFrame:
    cache = CACHE / f"xyz_{symbol}_{interval}_q.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        if (df["ts"].max() - df["ts"].min()).days >= days_back - 1:
            return df

    coin = f"xyz:{symbol}"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days_back * 24 * 3600 * 1000

    bar_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}[interval]
    step_ms = 4900 * bar_min * 60 * 1000

    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk = _fetch(coin, interval, cursor, min(cursor + step_ms, end_ms))
        if not chunk:
            cursor += step_ms
            continue
        rows.extend(chunk)
        cursor = chunk[-1]["t"] + bar_min * 60 * 1000
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume", "n": "n_trades"})
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df.to_parquet(cache)
    return df


# ---- yfinance fetch ----------------------------------------------------

def fetch_yf(symbol: str, interval: str, days_back: int) -> pd.DataFrame:
    """Cached yfinance fetch returning UTC-indexed OHLC."""
    import yfinance as yf
    cache = CACHE / f"yf_{symbol}_{interval}_{days_back}d.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    period = f"{days_back}d"
    df = yf.Ticker(symbol).history(period=period, interval=interval,
                                   auto_adjust=False, prepost=False)
    if df.empty:
        return df
    df.index = df.index.tz_convert("UTC")
    df = df.reset_index().rename(columns={"Datetime": "ts", "Date": "ts",
                                          "Open": "open", "High": "high",
                                          "Low": "low", "Close": "close",
                                          "Volume": "volume"})
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.to_parquet(cache)
    return df


# ---- Q1: daily tracking quality ----------------------------------------

def q1_daily_tracking() -> None:
    log.info("\n" + "=" * 72 + "\nQ1: DAILY TRACKING (xyz close at 21:00 UTC vs yf daily close)\n" + "=" * 72)
    rows = []
    for sym in NAMES:
        xyz = fetch_xyz(sym, "1h", 90)
        yf = fetch_yf(sym, "1d", 90)
        if xyz.empty or yf.empty:
            log.warning("  %s: empty data", sym)
            continue
        # Snap xyz to 21:00 UTC (close of US RTH ~ 4pm ET in EDT)
        xyz_close = xyz[xyz["ts"].dt.hour == 21][["ts", "close"]].copy()
        xyz_close["date"] = xyz_close["ts"].dt.date
        yf["date"] = yf["ts"].dt.date
        m = xyz_close[["date", "close"]].rename(columns={"close": "perp"}).merge(
            yf[["date", "close"]].rename(columns={"close": "cash"}), on="date")
        if m.empty:
            log.warning("  %s: no overlap", sym)
            continue
        m["basis_bps"] = 1e4 * np.log(m["perp"] / m["cash"])
        rows.append({
            "sym": sym,
            "n": len(m),
            "corr_lvl": m[["perp", "cash"]].corr().iloc[0, 1],
            "corr_ret": (np.log(m["perp"]).diff()
                         .corr(np.log(m["cash"]).diff())),
            "basis_mean_bps": m["basis_bps"].mean(),
            "basis_std_bps": m["basis_bps"].std(),
            "basis_p5": m["basis_bps"].quantile(0.05),
            "basis_p95": m["basis_bps"].quantile(0.95),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("  no Q1 data")
        return
    log.info("  %-6s %4s %9s %9s %12s %11s %10s %10s",
             "sym", "n", "corr_lvl", "corr_ret", "basis_mean", "basis_std",
             "p05", "p95")
    for _, r in df.iterrows():
        log.info("  %-6s %4d   %.5f   %.5f  %+8.1f bps  %7.1f bps  %+6.1f bps  %+6.1f bps",
                 r["sym"], r["n"], r["corr_lvl"], r["corr_ret"],
                 r["basis_mean_bps"], r["basis_std_bps"],
                 r["basis_p5"], r["basis_p95"])


# ---- Q2: intraday basis (last 7d, 1m, RTH only) ------------------------

def q2_intraday_basis() -> None:
    log.info("\n" + "=" * 72 + "\nQ2: INTRADAY BASIS (last 7d, 1m bars, RTH only)\n" + "=" * 72)
    for sym in NAMES:
        xyz = fetch_xyz(sym, "1m", 7)
        yf = fetch_yf(sym, "1m", 7)
        if xyz.empty or yf.empty:
            log.warning("  %s: empty data", sym)
            continue
        # yfinance 1m only contains RTH; align minute-exact
        yf_min = yf[["ts", "close"]].rename(columns={"close": "cash"})
        xyz_min = xyz[["ts", "close"]].rename(columns={"close": "perp"})
        m = yf_min.merge(xyz_min, on="ts")
        if len(m) < 100:
            log.warning("  %s: only %d aligned minutes, skipping", sym, len(m))
            continue
        m["basis_bps"] = 1e4 * np.log(m["perp"] / m["cash"])
        # Mean-reversion: AR(1) coefficient on basis
        b = m["basis_bps"].values
        rho1 = np.corrcoef(b[:-1], b[1:])[0, 1]
        rho10 = np.corrcoef(b[:-10], b[10:])[0, 1] if len(b) > 10 else np.nan
        # Half-life from AR(1): t_h = ln(0.5)/ln(rho1) minutes  (if rho1<1)
        hl = np.log(0.5) / np.log(rho1) if 0 < rho1 < 1 else np.nan
        log.info("  %-6s n=%5d  basis: mean=%+5.1fbps std=%4.1fbps "
                 "p05=%+5.1f p95=%+5.1f  AC1=%.3f AC10=%.3f half-life=%.1fmin",
                 sym, len(m), m["basis_bps"].mean(), m["basis_bps"].std(),
                 m["basis_bps"].quantile(0.05), m["basis_bps"].quantile(0.95),
                 rho1, rho10, hl if not np.isnan(hl) else -1)


# ---- Q3: off-hours / weekend drift -------------------------------------

def q3_offhours_drift() -> None:
    log.info("\n" + "=" * 72 + "\nQ3: OFF-HOURS DRIFT (xyz move while cash closed)\n" + "=" * 72)
    log.info("  fri-21UTC -> mon-14UTC perp move vs subsequent monday cash open jump")
    for sym in NAMES:
        xyz = fetch_xyz(sym, "1h", 90)
        yf = fetch_yf(sym, "1d", 90)
        if xyz.empty or yf.empty:
            continue
        xyz["dow"] = xyz["ts"].dt.dayofweek
        xyz["hr"] = xyz["ts"].dt.hour
        # Friday 21:00 UTC close
        fri = xyz[(xyz["dow"] == 4) & (xyz["hr"] == 21)][["ts", "close"]].copy()
        fri["date"] = fri["ts"].dt.date
        fri = fri.rename(columns={"close": "fri_close"})
        # Monday 14:00 UTC close (~ just before US RTH open)
        mon = xyz[(xyz["dow"] == 0) & (xyz["hr"] == 14)][["ts", "close"]].copy()
        mon["date"] = mon["ts"].dt.date - pd.Timedelta(days=3)  # match to fri
        mon = mon.rename(columns={"close": "mon_pre_open"})
        # Monday cash open
        yf["date"] = yf["ts"].dt.date - pd.Timedelta(days=0)  # daily 'open' is RTH open
        yf["dow"] = yf["ts"].dt.dayofweek
        mon_cash = yf[yf["dow"] == 0][["date", "open"]].copy()
        mon_cash["date"] = mon_cash["date"] - pd.Timedelta(days=3)
        mon_cash = mon_cash.rename(columns={"open": "mon_cash_open"})

        m = fri.merge(mon, on="date").merge(mon_cash, on="date")
        if len(m) < 3:
            continue
        m["perp_weekend_drift_bps"] = 1e4 * np.log(m["mon_pre_open"] / m["fri_close"])
        m["cash_weekend_gap_bps"] = 1e4 * np.log(m["mon_cash_open"] / m["fri_close"])
        m["perp_anticipated_bps"] = m["perp_weekend_drift_bps"] - m["cash_weekend_gap_bps"]
        log.info("  %-6s n_weekends=%d  "
                 "perp_drift=%+5.1f±%4.1fbps  cash_gap=%+5.1f±%4.1fbps  "
                 "perp-cash=%+5.1f±%4.1fbps",
                 sym, len(m),
                 m["perp_weekend_drift_bps"].mean(), m["perp_weekend_drift_bps"].std(),
                 m["cash_weekend_gap_bps"].mean(), m["cash_weekend_gap_bps"].std(),
                 m["perp_anticipated_bps"].mean(), m["perp_anticipated_bps"].std())


# ---- Q4: anomaly scan --------------------------------------------------

def q4_anomalies() -> None:
    log.info("\n" + "=" * 72 + "\nQ4: ANOMALY SCAN (xyz 1h: gaps, frozen, jumps)\n" + "=" * 72)
    for sym in NAMES:
        xyz = fetch_xyz(sym, "1h", 90)
        if xyz.empty:
            continue
        # gaps: missing hours
        ts = xyz["ts"]
        gaps = ((ts.diff().dt.total_seconds() / 3600).fillna(1) - 1).clip(lower=0)
        n_gap_bars = int(gaps.sum())
        # frozen prints: consecutive identical close
        frozen = (xyz["close"].diff() == 0).rolling(3).sum().eq(3).sum()
        # large 1h moves: |log return| > 5%
        ret = np.log(xyz["close"] / xyz["close"].shift(1))
        big = (ret.abs() > 0.05).sum()
        log.info("  %-6s rows=%5d  missing_hours=%5d  frozen3hr_runs=%4d  "
                 "moves>5%%=%3d  max_abs_ret=%.4f",
                 sym, len(xyz), n_gap_bars, int(frozen), int(big), ret.abs().max())


def main() -> None:
    log.info("xyz data quality probe: 5 names, last 60-90 days where applicable")
    q1_daily_tracking()
    q2_intraday_basis()
    q3_offhours_drift()
    q4_anomalies()


if __name__ == "__main__":
    main()
