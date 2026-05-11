"""Intraday PEAD diagnostic for v7 xyz.

For each earnings event in the 15 xyz-execution names (using cached
Polygon 5m × ~700d), decompose returns into:

  - overnight_gap   = day0_open / prev_close - 1
  - day0_intraday   = day0_close / day0_open - 1
  - day1_close      = day1_close / day0_close - 1   (currently captured by v7)
  - day2_close      = day2_close / day1_close - 1
  - day3_close      = day3_close / day2_close - 1

Day 0 = effective event day = first trading day whose CLOSE fully reflects
the announcement (per `_to_effective_event_date`):
  AMC → next BDay   (announcement Mon ≥16 ET → day0 = Tue)
  BMO → same day    (announcement Tue <9:30 ET → day0 = Tue)
  DMT → same day    (announcement during regular session → day0 = same day)

Returns are signed by surprise direction (signed = raw × sign(surprise_pct))
so positive = drift consistent with surprise direction. Hit rate = fraction
with signed_return > 0.

Returns are RAW (not basket-residualized). PEAD moves are typically 5-15%,
basket day-moves 1-2%, so basket noise doesn't change the qualitative
answer. A residualized version would refine point estimates by ~1 bps.

The point of this probe: quantify what intraday rebal at day0-open could
capture vs what v7 already captures at day0-close.

Usage:
    python -m ml.research.alpha_v9_xyz_pead_intraday
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
XYZ_NAMES = ["AAPL", "AMD", "AMZN", "COST", "GOOGL", "INTC", "LLY", "META",
             "MSFT", "MU", "NFLX", "NVDA", "ORCL", "PLTR", "TSLA"]


def to_effective_event_date(ts: pd.Timestamp) -> tuple[pd.Timestamp, str]:
    """Returns (effective_et_date, event_class) where class ∈ {AMC, BMO, DMT}."""
    et = ts.tz_convert("America/New_York")
    minute_of_day = et.hour * 60 + et.minute
    if et.hour >= 16:                       # AMC: ≥ 16:00 ET
        eff = (et.normalize() + BusinessDay(1)).date()
        cls = "AMC"
    elif minute_of_day < 9 * 60 + 30:       # BMO: < 9:30 ET
        eff = et.normalize().date()
        cls = "BMO"
    else:                                    # DMT: during regular session
        eff = et.normalize().date()
        cls = "DMT"
    return pd.Timestamp(eff), cls


def build_session_table(poly: pd.DataFrame) -> pd.DataFrame:
    """From raw 5m bars, build a daily session table with open/close at the
    NYSE regular session boundaries (9:30-16:00 ET)."""
    df = poly.copy()
    df["ts_et"] = df["ts"].dt.tz_convert("America/New_York")
    df["date"] = df["ts_et"].dt.normalize().dt.tz_localize(None)
    df["min_of_day"] = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    # Regular session: bars whose START minute is in [9:30, 16:00)
    reg = df[(df["min_of_day"] >= 9 * 60 + 30) & (df["min_of_day"] < 16 * 60)]
    sess = reg.groupby("date").agg(
        sess_open=("open", "first"),
        sess_close=("close", "last"),
        n_bars=("close", "count"),
    ).reset_index()
    # require ≥ 30 bars to count as a real session (skip half-days, holidays)
    sess = sess[sess["n_bars"] >= 30].sort_values("date").reset_index(drop=True)
    return sess


def event_returns(sess: pd.DataFrame, eff_date: pd.Timestamp,
                    n_after: int = 3) -> dict | None:
    """Compute decomposed returns around effective event date."""
    # Find session indices
    eff_d = eff_date.tz_localize(None) if eff_date.tzinfo else eff_date
    eff_idx = sess.index[sess["date"] == eff_d]
    if len(eff_idx) == 0:
        return None
    i = int(eff_idx[0])
    if i == 0 or i + n_after >= len(sess):
        return None  # need prev session and n_after future sessions
    prev_close = sess.iloc[i - 1]["sess_close"]
    day0_open = sess.iloc[i]["sess_open"]
    day0_close = sess.iloc[i]["sess_close"]
    out = {
        "overnight_gap": day0_open / prev_close - 1,
        "day0_intraday": day0_close / day0_open - 1,
        "day0_full":     day0_close / prev_close - 1,
    }
    for k in range(1, n_after + 1):
        ck = sess.iloc[i + k]["sess_close"]
        c_prev = sess.iloc[i + k - 1]["sess_close"]
        out[f"day{k}_close"] = ck / c_prev - 1
    return out


def main() -> None:
    rows = []
    for sym in XYZ_NAMES:
        poly_path = CACHE / f"poly_{sym}_5m.parquet"
        earn_path = CACHE / f"yf_{sym}_earnings.parquet"
        if not poly_path.exists() or not earn_path.exists():
            log.info("  %s: missing cache; skip", sym); continue
        poly = pd.read_parquet(poly_path)
        earn = pd.read_parquet(earn_path)
        sess = build_session_table(poly)
        if sess.empty:
            log.info("  %s: empty session table", sym); continue
        sess_dates = pd.to_datetime(sess["date"])
        first = sess_dates.min(); last = sess_dates.max()
        n_events_in_window = 0
        for _, e in earn.iterrows():
            ts = e["ts"]
            if pd.isna(ts):
                continue
            eff_date, cls = to_effective_event_date(ts)
            if eff_date < first or eff_date > last:
                continue
            r = event_returns(sess, eff_date)
            if r is None:
                continue
            n_events_in_window += 1
            sp = e.get("surprise_pct", np.nan)
            sign = np.sign(sp) if pd.notna(sp) else np.nan
            row = {"symbol": sym, "earnings_ts": ts, "eff_date": eff_date,
                   "class": cls, "surprise_pct": sp, "sign": sign}
            row.update(r)
            rows.append(row)
        log.info("  %-6s sessions=%4d  events=%2d  events_in_window=%d",
                 sym, len(sess), len(earn), n_events_in_window)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["sign"])
    df["sign"] = df["sign"].astype(int)
    log.info("\nTotal usable events (with surprise_pct sign): %d", len(df))
    if df.empty: return

    # Signed returns (multiply each window by sign(surprise))
    windows = ["overnight_gap", "day0_intraday", "day0_full",
               "day1_close", "day2_close", "day3_close"]
    for w in windows:
        df[f"signed_{w}"] = df[w] * df["sign"]

    # ---- aggregate by class ----
    log.info("\n=== Mean SIGNED return by event class (bps; positive = drift in surprise direction) ===")
    log.info("  %-6s %5s %12s %12s %12s %12s %12s %12s",
             "class", "n",
             "overnight", "day0_intra", "day0_full",
             "day1_close", "day2_close", "day3_close")
    for cls, g in df.groupby("class"):
        means = {w: g[f"signed_{w}"].mean() * 1e4 for w in windows}
        log.info("  %-6s %5d  %+10.0f  %+10.0f  %+10.0f  %+10.0f  %+10.0f  %+10.0f",
                 cls, len(g), means["overnight_gap"], means["day0_intraday"],
                 means["day0_full"], means["day1_close"],
                 means["day2_close"], means["day3_close"])
    means_all = {w: df[f"signed_{w}"].mean() * 1e4 for w in windows}
    log.info("  %-6s %5d  %+10.0f  %+10.0f  %+10.0f  %+10.0f  %+10.0f  %+10.0f",
             "ALL", len(df), means_all["overnight_gap"], means_all["day0_intraday"],
             means_all["day0_full"], means_all["day1_close"],
             means_all["day2_close"], means_all["day3_close"])

    log.info("\n=== Median SIGNED return by event class (bps) ===")
    log.info("  %-6s %5s %12s %12s %12s %12s %12s %12s",
             "class", "n",
             "overnight", "day0_intra", "day0_full",
             "day1_close", "day2_close", "day3_close")
    for cls, g in df.groupby("class"):
        meds = {w: g[f"signed_{w}"].median() * 1e4 for w in windows}
        log.info("  %-6s %5d  %+10.0f  %+10.0f  %+10.0f  %+10.0f  %+10.0f  %+10.0f",
                 cls, len(g), meds["overnight_gap"], meds["day0_intraday"],
                 meds["day0_full"], meds["day1_close"],
                 meds["day2_close"], meds["day3_close"])

    log.info("\n=== Hit rate (fraction signed_return > 0) by class ===")
    log.info("  %-6s %5s %12s %12s %12s %12s %12s %12s",
             "class", "n",
             "overnight", "day0_intra", "day0_full",
             "day1_close", "day2_close", "day3_close")
    for cls, g in df.groupby("class"):
        hits = {w: (g[f"signed_{w}"] > 0).mean() * 100 for w in windows}
        log.info("  %-6s %5d  %10.0f%%  %10.0f%%  %10.0f%%  %10.0f%%  %10.0f%%  %10.0f%%",
                 cls, len(g), hits["overnight_gap"], hits["day0_intraday"],
                 hits["day0_full"], hits["day1_close"],
                 hits["day2_close"], hits["day3_close"])

    # ---- cumulative drift starting from prev_close ----
    # Currently v7 captures: day1_close (Tue close → Wed close), then days 2-N.
    # Intraday rebal at day0_open could additionally capture: day0_intraday.
    # Intraday rebal pre-announcement (only possible for AMC) could
    # additionally capture: overnight_gap + day0_intraday. But that requires
    # forecasting the surprise — out of scope for a deterministic overlay.
    log.info("\n=== Capture decomposition (signed mean bps, AMC + BMO only) ===")
    sub = df[df["class"].isin(["AMC", "BMO"])]
    n = len(sub)
    capture = {
        "v7 today (day1_close + day2 + day3)":
            sub["signed_day1_close"].mean() + sub["signed_day2_close"].mean()
            + sub["signed_day3_close"].mean(),
        "+ day0_intraday (open-rebal overlay)":
            sub["signed_day0_intraday"].mean(),
        "(unreachable: overnight gap pre-announcement)":
            sub["signed_overnight_gap"].mean(),
    }
    for label, v in capture.items():
        log.info("  %-50s  %+8.0f bps", label, v * 1e4)
    log.info("  N=%d events", n)

    # ---- by name ----
    log.info("\n=== day0_intraday signed mean by name (AMC + BMO only, bps) ===")
    sub = df[df["class"].isin(["AMC", "BMO"])]
    by_sym = sub.groupby("symbol")["signed_day0_intraday"].agg(["mean", "median", "count"])
    by_sym["mean_bps"] = by_sym["mean"] * 1e4
    by_sym["med_bps"] = by_sym["median"] * 1e4
    by_sym = by_sym.sort_values("mean_bps", ascending=False)
    for sym, r in by_sym.iterrows():
        log.info("  %-6s n=%2d  mean=%+8.0f bps  median=%+8.0f bps",
                 sym, int(r["count"]), r["mean_bps"], r["med_bps"])

    # ---- per-event distribution ----
    log.info("\n=== Day0 intraday signed return distribution (AMC+BMO, bps) ===")
    arr = sub["signed_day0_intraday"].to_numpy() * 1e4
    qs = np.percentile(arr, [5, 25, 50, 75, 95])
    log.info("  pctile  5%%=%+5.0f  25%%=%+5.0f  50%%=%+5.0f  75%%=%+5.0f  95%%=%+5.0f",
             *qs)
    log.info("  share > +50 bps: %.0f%%   share < -50 bps: %.0f%%",
             (arr > 50).mean() * 100, (arr < -50).mean() * 100)

    # bootstrap CI on day0_intraday mean
    rng = np.random.default_rng(42)
    n = len(arr)
    means = [arr[rng.integers(0, n, size=n)].mean() for _ in range(2000)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    log.info("  bootstrap 95%% CI on mean: [%+5.0f, %+5.0f] bps  (point=%+.0f)",
             lo, hi, arr.mean())


if __name__ == "__main__":
    main()
