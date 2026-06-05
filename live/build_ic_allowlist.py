"""Build a PIT per-cycle dynamic allowlist = symbols with positive trailing-90d time-series IC
(spearman(pred, realized return) over the prior 90d, using only data known at the cycle). Tests whether
gating on past predictability helps forward — the honest 'can we fix the drag' test."""
import sys; from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
REPO=Path("/home/yuqing/ctaNew")
ff=pd.read_parquet(REPO/"live/state/convexity/hl_lean175/fullflow_hl60.parquet",columns=["symbol","open_time","pred","return_pct","exit_time"])
ff["open_time"]=pd.to_datetime(ff["open_time"],utc=True); ff["exit_time"]=pd.to_datetime(ff["exit_time"],utc=True)
ff=ff.sort_values("open_time")
days=pd.DatetimeIndex(sorted(ff.open_time.dt.normalize().unique()))
days=days[days>=pd.Timestamp("2025-10-04",tz="UTC")]
rows=[]
for d in days:
    win=ff[(ff.exit_time<=d)&(ff.open_time>=d-pd.Timedelta(days=90))]   # PIT: only realized-by-d
    for s,gg in win.groupby("symbol"):
        g2=gg[["pred","return_pct"]].dropna()
        if len(g2)<30: continue
        ic=spearmanr(g2["pred"],g2["return_pct"]).correlation
        if np.isfinite(ic) and ic>0:                                    # allow only positive trailing IC
            rows.append((d,s))
al=pd.DataFrame(rows,columns=["open_time","symbol"])
# expand each day's allowlist to that day's 4h cycles
cyc=ff[["open_time"]].drop_duplicates(); cyc["day"]=cyc.open_time.dt.normalize()
al=al.rename(columns={"open_time":"day"}).merge(cyc,on="day").drop(columns="day")
out=REPO/"live/state/convexity/ic_allowlist.parquet"; al.to_parquet(out,index=False)
print(f"IC allowlist: {al.open_time.nunique()} cycles, avg {al.groupby('open_time').size().mean():.0f} syms allowed/cycle -> {out}")
