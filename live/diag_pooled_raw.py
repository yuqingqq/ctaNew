"""Confirm the xs_z-vs-raw gap: pooled wins in normalized (xs_z) space but the RAW-bps K=3 L/S is what trades.
Check: raw-bps leg returns, realized volatility of the picked names, and a naive equal-weight K=3 L/S Sharpe (raw).
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO)); CONV=REPO/"live/state/convexity"
ann=np.sqrt(6*365)

def load(sub):
    d=pd.read_parquet(CONV/sub/"v0full_hl60.parquet"); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    # per-name trailing realized vol proxy: |return_pct| rolling within symbol (PIT)
    d=d.sort_values(["symbol","open_time"])
    d["absret"]=d["return_pct"].abs()
    return d

for name,sub in [("PROD per-symbol","hl_residrev"),("POOLED","hl_pooled_residrev")]:
    d=load(sub)
    def legmetrics(x):
        x=x.dropna(subset=["pred"])
        if len(x)<6: return None
        L=x.nlargest(3,"pred"); S=x.nsmallest(3,"pred")
        return pd.Series({"longbps":L["return_pct"].mean()*1e4,"shortbps":S["return_pct"].mean()*1e4,
                          "ls_bps":(L["return_pct"].mean()-S["return_pct"].mean())*1e4,
                          "pick_vol_bps":pd.concat([L["absret"],S["absret"]]).mean()*1e4})
    m=d.groupby("open_time").apply(legmetrics).dropna()
    ls=m["ls_bps"]
    print(f"\n===== {name} =====")
    print(f"raw legs (bps/cycle): long3 {m['longbps'].mean():+.1f}  short3 {m['shortbps'].mean():+.1f}  L-S {ls.mean():+.1f}")
    print(f"naive equal-wt K=3 L/S RAW Sharpe = {ls.mean()/ls.std()*ann:+.3f}  (gross, no cost/beta-neut/sleeve)")
    print(f"avg |return| of PICKED names = {m['pick_vol_bps'].mean():.0f} bps  (volatility of what it trades)")
    # split L-S by cross-sectional dispersion tercile of the cycle
    disp=d.groupby("open_time")["return_pct"].std()*1e4
    j=pd.concat([ls.rename("ls"),disp.rename("disp")],axis=1).dropna()
    j["dt"]=pd.qcut(j["disp"],3,labels=["lo_disp","mid","hi_disp"])
    print("  L-S bps by cycle cross-sec dispersion:", {k:round(v,1) for k,v in j.groupby("dt")["ls"].mean().items()})
    print("  cycle count by dispersion where L-S<0:", {k:int((g['ls']<0).sum()) for k,g in j.groupby('dt')})
