"""Fetch Deribit DVOL (BTC & ETH implied-vol index, free public API) over the backtest window,
align to the 4h decision grid, and emit PIT features for the optimization loop (orthogonal data).

DVOL = Deribit's forward 30d implied-vol index — a LEADING regime/crowding signal orthogonal to
price/funding. Output: research/.../results/_cache/deribit_dvol.parquet
"""
from __future__ import annotations
import json, time, urllib.request
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"research/convexity_portable_2026-05-20/results/_cache/deribit_dvol.parquet"
API = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
START = pd.Timestamp("2024-12-01", tz="UTC")   # cover HL70 (2025-03+) + warmup; and 44-sym overlap
END   = pd.Timestamp("2026-05-11", tz="UTC")
RES   = 3600  # hourly


def fetch_dvol(ccy):
    rows=[]; cur=int(START.timestamp()*1000); end=int(END.timestamp()*1000)
    step=RES*1000*1000  # ~1000 hourly points per call
    while cur < end:
        hi=min(cur+step, end)
        url=f"{API}?currency={ccy}&start_timestamp={cur}&end_timestamp={hi}&resolution={RES}"
        for attempt in range(4):
            try:
                r=json.loads(urllib.request.urlopen(url,timeout=30).read())
                data=r.get("result",{}).get("data",[]); break
            except Exception as e:
                if attempt==3: print(f"  {ccy} chunk err {e}"); data=[]
                time.sleep(2)
        for d in data:  # [ts, open, high, low, close]
            rows.append((d[0], d[4]))
        cur=hi; time.sleep(0.2)
    df=pd.DataFrame(rows, columns=["ts","dvol"]).drop_duplicates("ts").sort_values("ts")
    df["open_time"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
    return df.set_index("open_time")["dvol"].astype(float)


def main():
    t0=time.time()
    print("=== fetch Deribit DVOL (BTC, ETH) ===", flush=True)
    btc=fetch_dvol("BTC"); eth=fetch_dvol("ETH")
    print(f"  BTC {len(btc)} pts {btc.index.min()}→{btc.index.max()}; ETH {len(eth)} pts", flush=True)
    # align to 4h decision grid
    grid=pd.date_range(START.ceil("h"), END, freq="4h", tz="UTC")
    def to4h(s):
        s=s.reindex(s.index.union(grid)).sort_index().ffill().reindex(grid)  # as-of backward (last published)
        return s
    df=pd.DataFrame({"dvol_btc":to4h(btc), "dvol_eth":to4h(eth)}, index=grid)
    df.index.name="open_time"
    # PIT features (expanding percentile + trailing change) — all use only past data
    for c in ["dvol_btc","dvol_eth"]:
        df[f"{c}_pctile"]=df[c].expanding(min_periods=180).apply(lambda x: (x.iloc[-1]>=x).mean(), raw=False)
        df[f"{c}_chg_1d"]=df[c]-df[c].shift(6)    # 6×4h = 1d
        df[f"{c}_chg_3d"]=df[c]-df[c].shift(18)
    df=df.reset_index()
    df.to_parquet(OUT, index=False)
    print(f"  saved {OUT.name}: {len(df)} 4h rows, cols {[c for c in df.columns if c!='open_time']}")
    print(f"  dvol_btc range {df['dvol_btc'].min():.1f}–{df['dvol_btc'].max():.1f}, "
          f"nan {df['dvol_btc'].isna().sum()}")
    print(f"DONE [{time.time()-t0:.0f}s]")


if __name__=="__main__":
    main()
