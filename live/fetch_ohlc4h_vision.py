"""Fetch 4h OHLCV+amount for the convexity panel universe from Binance Vision monthly archives
(futures um), to compute Alpha191 factors. Robust to: header rows (2025+), ms->us timestamp switch
(Jan 2025), and 404s for months before a symbol listed. Output: data/ml/cache/alpha191_ohlc4h.parquet
restricted to the exact (symbol, open_time) rows in panel_expanded_v0 (the 4h decision grid).
"""
import io, sys, zipfile, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd, requests
warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); OUT = REPO/"data/ml/cache/alpha191_ohlc4h.parquet"
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
COLS = ["open_time","open","high","low","close","volume","close_time","quote_volume","count",
        "taker_buy_volume","taker_buy_quote_volume","ignore"]
KEEP = ["open_time","open","high","low","close","volume","quote_volume"]

pan = pd.read_parquet(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet", columns=["symbol","open_time"])
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
SYMS = sorted(pan["symbol"].unique())
grid = {s: set(g["open_time"]) for s, g in pan.groupby("symbol")}   # exact 4h rows we need per symbol
MONTHS = pd.period_range("2020-08", pan["open_time"].max().to_period("M"), freq="M")

def _ts(col):
    v = pd.to_numeric(col, errors="coerce")
    unit = "us" if v.dropna().median() > 1e15 else "ms"      # Binance switched ms->us in 2025-01
    return pd.to_datetime(v, unit=unit, utc=True)

def fetch_month(sym, per):
    url = f"{BASE}/{sym}/4h/{sym}-4h-{per.strftime('%Y-%m')}.zip"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200: return None
        z = zipfile.ZipFile(io.BytesIO(r.content)); n = z.namelist()[0]
        raw = z.read(n).decode(); hdr = 0 if raw.split(",",1)[0]=="open_time" else None
        d = pd.read_csv(io.StringIO(raw), header=hdr, names=None if hdr==0 else COLS)
        d.columns = COLS[:d.shape[1]]
        d["open_time"] = _ts(d["open_time"])
        for c in ["open","high","low","close","volume","quote_volume"]: d[c] = pd.to_numeric(d[c], errors="coerce")
        return d[KEEP]
    except Exception: return None

def fetch_sym(sym):
    parts = [fetch_month(sym, p) for p in MONTHS]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts: return sym, None
    d = pd.concat(parts, ignore_index=True).dropna(subset=["open_time"]).drop_duplicates("open_time")
    d = d[d["open_time"].isin(grid[sym])]                    # align to panel's 4h decision grid
    d["symbol"] = sym
    return sym, d

def main():
    rows, ok = [], 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(fetch_sym, s): s for s in SYMS}
        for i, f in enumerate(as_completed(futs), 1):
            sym, d = f.result()
            if d is not None and len(d): rows.append(d); ok += 1
            if i % 25 == 0: print(f"  {i}/{len(SYMS)} syms done, {ok} with data", flush=True)
    out = pd.concat(rows, ignore_index=True).sort_values(["symbol","open_time"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(OUT)
    cov = out.groupby("symbol").size()
    print(f"DONE: {out['symbol'].nunique()} syms, {len(out)} rows -> {OUT}")
    print(f"  panel rows {len(pan)}, fetched {len(out)} ({100*len(out)/len(pan):.0f}% coverage), median bars/sym {int(cov.median())}")

if __name__ == "__main__": main()
