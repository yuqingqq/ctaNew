"""Calibration curve: mean realized fwd 24h residual alpha by each model's OWN pred-decile. Shows WHERE the factor
changes predictive quality — if it flattens the extreme deciles (0=shorts, 9=longs) while improving the middle
ordering, that confirms: IC gain is in the untraded middle; tail-rank accuracy (what we trade) degrades.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
b=pd.read_parquet(f"{R}/live/state/convexity/hl_tgt_res_long/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"base"})
f=pd.read_parquet(f"{R}/live/state/convexity/hl_v4fac_long/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"fac"})
for x in (b,f): x["open_time"]=pd.to_datetime(x["open_time"],utc=True)
d=b.merge(f,on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])
def prof(col):
    d["dec"]=d.groupby("open_time")[col].transform(lambda s:pd.qcut(s,10,labels=False,duplicates="drop"))
    return d.groupby("dec")["fwd"].mean()
pb,pf=prof("base"),prof("fac")
print("realized fwd alpha (bps) by pred-decile — 0=SHORT tail (want most negative), 9=LONG tail (want most positive)\n")
print(f"  {'decile':>6s} {'base':>8s} {'+factor':>8s} {'Δ':>7s}   region")
for k in range(10):
    reg="SHORT tail (traded)" if k<=1 else "LONG tail (traded)" if k>=8 else "middle (NOT traded)"
    print(f"  {k:>6d} {pb[k]:+8.1f} {pf[k]:+8.1f} {pf[k]-pb[k]:+7.1f}   {reg}")
print(f"\n  short-tail (dec0) Δ: {pf[0]-pb[0]:+.1f} (higher=WORSE short)   long-tail (dec9) Δ: {pf[9]-pb[9]:+.1f} (lower=WORSE long)")
mid_b=(pb[3:7].max()-pb[3:7].min()); mid_f=(pf[3:7].max()-pf[3:7].min())
print(f"  middle deciles 3-6 spread: base {mid_b:.1f} -> factor {mid_f:.1f}")
print("DCDONE")
