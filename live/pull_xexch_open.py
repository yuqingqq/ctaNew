import json,sys,time; from datetime import datetime,timezone; from pathlib import Path
import pandas as pd; sys.path.insert(0,"/home/yuqing/ctaNew")
from data_collectors.coinbase_data_fetcher import CoinbaseDataFetcher
from data_collectors.okx_data_fetcher import OKXDataFetcher
R=Path("/home/yuqing/ctaNew"); OUT=R/"data/ml/cache/xexch_open"; START=datetime(2025,1,1,tzinfo=timezone.utc); END=datetime.now(timezone.utc)
def log(m): print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {m}",flush=True)
u=json.load(open(R/"live/models/convexity_v1_universe.json")); syms=u["tradeable_low_vol"]
cb=CoinbaseDataFetcher(); ok=OKXDataFetcher()
cbp=set(cb.list_products()["id"]); oki=set(ok.list_swap_instruments()["instId"])
(OUT/"coinbase").mkdir(parents=True,exist_ok=True); (OUT/"okx").mkdir(parents=True,exist_ok=True)
cbn=okn=0
for i,s in enumerate(syms):
    p=cb.binance_sym_to_coinbase(s)
    if p in cbp:
        try:
            d=cb.fetch_backfill(p,START,END,granularity=3600,sleep_ms=120)  # 1h: bar opening hh:00 has open=price@hh:00
            if len(d):
                d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
                d=d[(d.open_time.dt.hour%4==0)&(d.open_time.dt.minute==0)][["open_time","open"]]
                d.to_parquet(OUT/"coinbase"/f"{s}.parquet"); cbn+=1
        except Exception as e: log(f" CB {s} {e}")
    o=ok.binance_sym_to_okx(s)
    if o in oki:
        try:
            d=ok.fetch_backfill(o,START,END,bar="4H")    # 4H bar opening hh:00 has open=price@hh:00
            if len(d):
                d["open_time"]=pd.to_datetime(d["open_time"],utc=True); d=d[["open_time","open"]]
                d.to_parquet(OUT/"okx"/f"{s}.parquet"); okn+=1
        except Exception as e: log(f" OKX {s} {e}")
    if (i+1)%20==0: log(f"{i+1}/{len(syms)} cb={cbn} okx={okn}")
log(f"DONE cb {cbn} okx {okn}")
