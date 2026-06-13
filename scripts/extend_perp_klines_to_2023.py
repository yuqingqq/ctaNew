"""Extend Binance PERP 5m klines back to 2023-01 via Binance Vision monthly archives.

Stores in same format: data/ml/test/parquet/klines/<SYM>/5m/<YYYY-MM-DD>.parquet
(but we'll store monthly aggregates to reduce file count: <YYYY-MM>.parquet)

URL: https://data.binance.vision/data/futures/um/monthly/klines/<SYM>/5m/<SYM>-5m-<YYYY-MM>.zip
"""
from __future__ import annotations
import sys, time, io, zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

REPO = Path("/home/yuqing/ctaNew")
KLINES = REPO / "data/ml/test/parquet/klines"
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"

START = date(2023, 1, 1)
END = date(2025, 3, 1)  # existing data starts ~2025-03

KLINE_COLS = ["open_time","open","high","low","close","volume","close_time",
              "quote_volume","n_trades","taker_buy_base","taker_buy_quote","ignore"]


def fetch_monthly(sym, ym):
    url = f"{BASE}/{sym}/5m/{sym}-5m-{ym}.zip"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 404: return None
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, header=None, names=KLINE_COLS)
        # detect header row (some archives have header)
        if str(df.iloc[0]["open"]).lower() in ("open", "open_price"):
            df = df.iloc[1:].reset_index(drop=True)
        for c in ["open","high","low","close","volume","quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        ot = pd.to_numeric(df["open_time"], errors="coerce")
        unit = "us" if ot.max() > 1e14 else "ms"
        df["open_time"] = pd.to_datetime(ot, unit=unit, utc=True)
        return df[["open_time","open","high","low","close","volume","close_time","quote_volume"]]
    except Exception as e:
        return None


def main():
    candidates = [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT",
        "AVAXUSDT","LINKUSDT","DOTUSDT","ATOMUSDT","LTCUSDT","BCHUSDT","NEARUSDT",
        "UNIUSDT","TIAUSDT","SUIUSDT","SEIUSDT","INJUSDT","ARBUSDT","APTUSDT","OPUSDT",
        "AAVEUSDT","AXSUSDT","FILUSDT","ETCUSDT","TRBUSDT","WLDUSDT","ICPUSDT","ONDOUSDT",
        "PENDLEUSDT","LDOUSDT","JTOUSDT","ENAUSDT","HBARUSDT","TONUSDT","STRKUSDT",
        "WIFUSDT","ORDIUSDT","JUPUSDT","GMXUSDT","TAOUSDT","RUNEUSDT","SUSDT","ZECUSDT",
    ]
    print(f"Extending PERP 5m klines to 2023-01 for {len(candidates)} syms\n", flush=True)

    months = []
    cur = START
    while cur < END:
        months.append(f"{cur.year}-{cur.month:02d}")
        cur = date(cur.year+1, 1, 1) if cur.month == 12 else date(cur.year, cur.month+1, 1)

    for i, sym in enumerate(candidates, 1):
        sym_dir = KLINES / sym / "5m"
        if not sym_dir.exists():
            print(f"[{i}/{len(candidates)}] {sym}: no existing 5m dir; skip", flush=True)
            continue
        # Check if already extended
        existing = sorted(sym_dir.glob("*.parquet"))
        ext_marker = sym_dir / "2023-01.parquet"
        if ext_marker.exists():
            print(f"[{i}/{len(candidates)}] {sym}: already extended; skip", flush=True)
            continue
        added_months = 0
        added_rows = 0
        for ym in months:
            out_f = sym_dir / f"{ym}.parquet"
            if out_f.exists(): continue
            df = fetch_monthly(sym, ym)
            if df is not None and len(df) > 0:
                df.to_parquet(out_f, index=False)
                added_months += 1
                added_rows += len(df)
            time.sleep(0.05)
        print(f"[{i}/{len(candidates)}] {sym}: +{added_months} months, +{added_rows:,} bars", flush=True)

    print("\nDone extending PERP klines.", flush=True)


if __name__ == "__main__":
    main()
