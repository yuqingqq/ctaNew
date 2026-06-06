"""VALIDATION (read-only): confirm that ranking bars_since_high over the FULL 174-symbol
panel cohort reproduces the records' XLM xs_rank (0.749), vs the collapsed 94-cohort (0.67).

Computes bars_since_high at 06-04 12:00 for every panel symbol straight from the klines
archive (the 80 high-vol names still run through 06-04 12:00, so no fetch is needed for this
check), then ranks XLM over the full set and over the low-vol-94 subset. Touches no live state.
"""
from __future__ import annotations
import sys, glob, json, gc
from pathlib import Path
import pandas as pd, numpy as np
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
KLINES = REPO / "data/ml/test/parquet/klines"
PANEL = REPO / "outputs/vBTC_features/panel_expanded_v0.parquet"
B = pd.Timestamp("2026-06-04 12:00", tz="UTC")


def bsh_at(sym, B):
    """bars_since_high (rolling-288 high) at bar B, from the symbol's klines. Needs ~B-16d history."""
    files = sorted((KLINES / sym / "5m").glob("*.parquet"))
    files = [f for f in files if f.stem >= (B - pd.Timedelta(days=18)).strftime("%Y-%m-%d")]
    if not files:
        return np.nan
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in files], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df = df[df["open_time"] <= B]
    if len(df) < 300 or df["open_time"].max() < B:
        return np.nan
    c = pd.to_numeric(df["close"], errors="coerce")
    hi = c.rolling(288).max(); inh = (c == hi).astype(int)
    v = (1 - inh).groupby(inh.cumsum()).cumcount().iloc[-1]
    del df, c, hi, inh; gc.collect()
    return float(v)


def main():
    syms = sorted(s for s in pd.read_parquet(PANEL, columns=["symbol"]).symbol.unique() if s != "BTCUSDT")
    excl = set(json.load(open(REPO / "live/models/convexity_v1_universe.json"))["exclude_high_vol"])
    print(f"panel universe: {len(syms)} symbols (excl-high-vol set: {len(excl)})", flush=True)
    rows = []
    for i, s in enumerate(syms):
        try:
            v = bsh_at(s, B)
        except Exception as e:
            v = np.nan; print(f"  {s} ERR {str(e)[:50]}", flush=True)
        if v == v:
            rows.append((s, v))
        if (i + 1) % 40 == 0:
            print(f"  ...{i+1}/{len(syms)} processed", flush=True)
    fs = pd.Series(dict(rows))
    full_rank = fs.rank(pct=True)
    low = fs[~fs.index.isin(excl)]
    low_rank = low.rank(pct=True)
    print(f"\nsymbols with valid bars_since_high @ {B:%m-%d %H:%M}: {len(fs)}")
    print(f"  full cohort n={len(fs)}   XLM xs_rank = {full_rank.get('XLMUSDT', np.nan):.3f}")
    print(f"  low-vol-94 n={len(low)}   XLM xs_rank = {low_rank.get('XLMUSDT', np.nan):.3f}")
    print(f"  gen/records reported XLM xs_rank = 0.749")
    print(f"  -> full-cohort match: {abs(full_rank.get('XLMUSDT', -9) - 0.749) < 0.01}")


if __name__ == "__main__":
    main()
