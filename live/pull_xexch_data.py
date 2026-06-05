"""#185 — pull Coinbase (USD spot) + OKX (USDT swap) candles for the convexity v1
tradeable low-vol universe, to build a cross-exchange premium FEATURE.

Coinbase: no native 4h granularity -> pull 1h (3600s) and resample to 4h close.
OKX: native 4H bar.
Output: data/ml/cache/xexch/{coinbase,okx}/{BINANCE_SYM}.parquet  (open_time UTC, close).
"""
from __future__ import annotations
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew")
from data_collectors.coinbase_data_fetcher import CoinbaseDataFetcher
from data_collectors.okx_data_fetcher import OKXDataFetcher

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"data/ml/cache/xexch"
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)


def log(m): print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {m}", flush=True)


def main():
    u = json.load(open(REPO/"live/models/convexity_v1_universe.json"))
    syms = u["tradeable_low_vol"]
    cb = CoinbaseDataFetcher(); ok = OKXDataFetcher()
    cb_prods = set(cb.list_products()["id"]); ok_inst = set(ok.list_swap_instruments()["instId"])
    (OUT/"coinbase").mkdir(parents=True, exist_ok=True); (OUT/"okx").mkdir(parents=True, exist_ok=True)

    log(f"pulling {len(syms)} symbols  {START.date()} -> {END.date()}")
    cb_ok = ok_ok = 0
    for i, s in enumerate(syms):
        # ---- Coinbase 1h -> 4h ----
        prod = cb.binance_sym_to_coinbase(s)
        if prod in cb_prods:
            try:
                df = cb.fetch_backfill(prod, START, END, granularity=3600, sleep_ms=120)
                if len(df):
                    df = df.set_index("open_time")[["close"]].astype(float)
                    df.index = pd.to_datetime(df.index, utc=True)
                    h4 = df.resample("4h", label="left", closed="left").last().dropna()  # 4h close aligned to Binance
                    h4.reset_index().to_parquet(OUT/"coinbase"/f"{s}.parquet"); cb_ok += 1
            except Exception as e:
                log(f"  CB {s} ({prod}) FAIL {e}")
        # ---- OKX native 4H ----
        oksym = ok.binance_sym_to_okx(s)
        if oksym in ok_inst:
            try:
                df = ok.fetch_backfill(oksym, bar="4H", start=START, end=END)
                if len(df):
                    df = df[["open_time", "close"]].copy(); df["close"] = df["close"].astype(float)
                    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
                    df.to_parquet(OUT/"okx"/f"{s}.parquet"); ok_ok += 1
            except Exception as e:
                log(f"  OKX {s} ({oksym}) FAIL {e}")
        if (i+1) % 10 == 0:
            log(f"  {i+1}/{len(syms)}  CB_ok={cb_ok} OKX_ok={ok_ok}")
        time.sleep(0.05)
    log(f"DONE  Coinbase saved {cb_ok}  OKX saved {ok_ok}  -> {OUT}")


if __name__ == "__main__":
    main()
