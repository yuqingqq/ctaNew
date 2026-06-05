"""Momentum preds at a configurable lookback (MOM_BARS env, 4h bars) for the trend-horizon sweep.
Tests whether high-vol 'trend' is robust across horizons or a mom_30d fluke. pred = xs-standardized mom_Nbar (PIT .shift(1)).
-> live/state/convexity/hl_mom{N}/{fullflow_hl60,v0full_hl60}.parquet"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.convexity_paper_bot as bot
N=int(os.environ.get("MOM_BARS","180")); OUT=REPO/f"live/state/convexity/hl_mom{N}"; OUT.mkdir(parents=True,exist_ok=True)
base=pd.read_parquet(REPO/"live/state/convexity/hl_wfund175/fullflow_hl60.parquet"); base["open_time"]=pd.to_datetime(base["open_time"],utc=True)
syms=sorted(base.symbol.unique()); rows=[]
for s in syms:
    c=bot.load_close_4h(s)
    if c.empty: continue
    mom=(c/c.shift(N)-1).shift(1)
    rows.append(pd.DataFrame({"symbol":s,"open_time":mom.index,"mom":mom.values}))
M=pd.concat(rows,ignore_index=True); M["open_time"]=pd.to_datetime(M["open_time"],utc=True)
d=base.drop(columns=["pred"]).merge(M,on=["symbol","open_time"],how="left")
g=d.groupby("open_time")["mom"]; d["pred"]=((d["mom"]-g.transform("mean"))/g.transform("std").replace(0,np.nan))
d=d.dropna(subset=["pred"]); keep=["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]
d[keep].to_parquet(OUT/"fullflow_hl60.parquet"); d[keep].to_parquet(OUT/"v0full_hl60.parquet")
print(f"mom_{N}bar preds: {d.symbol.nunique()} syms, {len(d)} rows -> {OUT}")
