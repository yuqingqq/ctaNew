"""Shared, reviewed building blocks for the research scripts (review 2026-09-28).

- TRADFI: Binance-USDM stock / ETF / commodity perps, verified by price level against the
  ticker (e.g. AAPLUSDT ~ $317, CLUSDT ~ $87 crude). Ticker collisions that are crypto
  tokens (F, T, DIA, SPX: all < $1) are deliberately NOT listed.
- panel_meta(): per symbol, the first minute with a price and every run of >= 24 h of zero
  volume. Binance Vision keeps publishing flat zero-volume bars after a delisting, so a
  price inside such a run is frozen, not traded. Computed from the per-symbol parquets one
  file at a time (small RSS) and cached next to the panel.
- Stale: which coins have a frozen run overlapping a window.
- mkt_median(): median log return of liquid, non-stale coins, excluding the coin itself.
- metrics_row(): Vision `metrics` rows stamped T describe [T, T+5 min); a row is usable at
  time X only if T + 5 min <= X (the old `T <= X` looked up to 5 minutes ahead).
- days_between(): every UTC date a lookback spans (the old helper skipped middle days).
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TRADFI = {"AAPL", "AMD", "AMZN", "ARM", "AVGO", "BABA", "CL", "COIN", "COPPER", "CRCL", "CRM", "DELL",
          "GOOGL", "HOOD", "IBM", "INTC", "META", "MSFT", "MSTR", "MU", "NATGAS", "NVDA", "ORCL", "PLTR",
          "QCOM", "QQQ", "SMCI", "SNOW", "SOXL", "SPY", "TQQQ", "TSLA", "TSM", "WMT", "XAG", "XPD", "XPT"}
STALE_MIN = 1440          # zero-volume run length (minutes) that marks frozen / delisted bars
METRICS_SPAN = pd.Timedelta(minutes=5)


def is_tradfi(symbol: str) -> bool:
    return symbol.endswith("USDT") and symbol[:-4] in TRADFI


def panel_meta(panel_dir: Path, cache_dir: Path, syms: list[str], idx: pd.DatetimeIndex) -> dict:
    """first_valid[i] and stale intervals (sym_i, start, end) in panel minute indices."""
    path = cache_dir / "meta.json"
    if path.exists():
        m = json.loads(path.read_text())
        if m.get("syms") == syms:
            return m
    T = len(idx)
    first_valid, st_i, st_a, st_b = [], [], [], []
    for i, s in enumerate(syms):
        df = pd.read_parquet(panel_dir / f"{s}.parquet", columns=["close", "quote_volume"])
        df = df[~df.index.duplicated()]
        pos = ((df.index - idx[0]) // pd.Timedelta("1min")).to_numpy()
        ok = (pos >= 0) & (pos < T) & np.isfinite(df["close"].to_numpy())
        pos, vol = pos[ok], df["quote_volume"].to_numpy()[ok]
        first_valid.append(int(pos[0]) if len(pos) else T)
        vfull = np.zeros(T, dtype=bool)          # True = a traded minute
        vfull[pos[vol > 0]] = True
        if len(pos):
            vfull[: pos[0]] = True                # before listing: absent, not "frozen"
        z = ~vfull
        if z.any():
            d = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
            starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
            for a, b in zip(starts, ends):
                if b - a + 1 >= STALE_MIN:
                    st_i.append(i); st_a.append(int(a)); st_b.append(int(b))
        del df
    m = {"syms": syms, "first_valid": first_valid, "stale_i": st_i, "stale_a": st_a, "stale_b": st_b}
    path.write_text(json.dumps(m))
    return m


class Stale:
    def __init__(self, meta: dict, n: int):
        self.i = np.asarray(meta["stale_i"], dtype=np.int64)
        self.a = np.asarray(meta["stale_a"], dtype=np.int64)
        self.b = np.asarray(meta["stale_b"], dtype=np.int64)
        self.n = n
        self.first_valid = np.asarray(meta["first_valid"], dtype=np.int64)

    def coin(self, i: int, j0: int, j1: int) -> bool:
        """True if coin i has a frozen run overlapping [j0, j1] or was not yet listed at j0."""
        if j0 < self.first_valid[i]:
            return True
        sel = self.i == i
        return bool(np.any((self.a[sel] <= j1) & (self.b[sel] >= j0)))

    def mask(self, j0: int, j1: int) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        hit = (self.a <= j1) & (self.b >= j0)
        m[self.i[hit]] = True
        m |= self.first_valid > j0
        return m


def mkt_median(C, LIQ, j0: int, j1: int, stale: Stale, exclude: int | None = None, liq_min: float = 5e6) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        col = np.log(np.asarray(C[:, j1], np.float64) / np.asarray(C[:, j0], np.float64))
    keep = (LIQ[:, j0 // 15] >= liq_min) & np.isfinite(col) & ~stale.mask(j0, j1)
    if exclude is not None:
        keep[exclude] = False
    return float(np.median(col[keep])) if keep.sum() >= 10 else float("nan")


def metrics_row(m: pd.DataFrame | None, at: pd.Timestamp, col: str) -> float:
    """Last Vision-metrics value whose 5-minute span ended at or before `at`."""
    if m is None or not len(m):
        return float("nan")
    x = m[m.index + METRICS_SPAN <= at]
    return float(x[col].iloc[-1]) if len(x) else float("nan")


def days_between(ts: pd.Timestamp, back_hours: float) -> list[str]:
    d0 = (ts - pd.Timedelta(hours=back_hours)).normalize()
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(d0, ts.normalize(), freq="D")]


def dir_simple(logret, direction) -> np.ndarray:
    """Direction-adjusted LOG return -> SIMPLE return of the position that follows the move.
    Pump (d=+1): x = r - m  -> exp(x) - 1        (long, relative to the market)
    Dump (d=-1): x = m - r  -> 1 - exp(-x)       (short: a -80% crash is +80%, never +400%)"""
    x = np.asarray(logret, dtype=float)
    d = np.broadcast_to(np.asarray(direction, dtype=float), x.shape)
    return np.where(d > 0, np.expm1(x), -np.expm1(-x))


def arith_pct_dir(logret, direction) -> float:
    v = dir_simple(logret, direction)
    v = v[np.isfinite(v)]
    return float(np.mean(v) * 100) if len(v) else float("nan")


def median_pct_dir(logret, direction) -> float:
    v = dir_simple(logret, direction)
    v = v[np.isfinite(v)]
    return float(np.median(v) * 100) if len(v) else float("nan")


def arith_pct(logret) -> float:
    x = np.asarray(logret, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(np.exp(x) - 1) * 100) if len(x) else float("nan")


def median_pct(logret) -> float:
    x = np.asarray(logret, dtype=float)
    x = x[np.isfinite(x)]
    return float((math.exp(np.median(x)) - 1) * 100) if len(x) else float("nan")
