"""#185 steelman — test okx premium as a SELECTION TILT (bypasses Ridge collinearity with resid_rev).
pred_adj = pred - LAMBDA * zscore_xs(okx_level).  Negative-IC signal => demote high-premium longs, promote as shorts.
Builds tilted copies of baseline preds for a few lambdas; harness run done by caller.
"""
import sys, os; from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.convexity_paper_bot as bot
XE=REPO/"data/ml/cache/xexch/okx"; CONV=REPO/"live/state/convexity"

# okx XS-demeaned premium, PIT shift1
prem={}
for f in XE.glob("*.parquet"):
    cv=pd.read_parquet(f); cv["open_time"]=pd.to_datetime(cv["open_time"],utc=True); cv=cv.set_index("open_time")["close"].astype(float)
    bn=bot.load_close_4h(f.stem)
    if bn is None or not len(bn): continue
    j=pd.concat([np.log(cv).rename("o"),np.log(bn).rename("b")],axis=1).dropna()
    if len(j)<100: continue
    prem[f.stem]=(j["o"]-j["b"]).rename(f.stem)
P=pd.concat(prem.values(),axis=1); Pxs=P.sub(P.median(axis=1),axis=0).shift(1)
# per-bar z-score so the tilt magnitude is comparable to pred (xs_z scale)
Pz=Pxs.sub(Pxs.mean(axis=1),axis=0).div(Pxs.std(axis=1).replace(0,np.nan),axis=0)
L=Pz.reset_index().melt(id_vars="open_time",var_name="symbol",value_name="okx_z").dropna()
L["open_time"]=pd.to_datetime(L["open_time"],utc=True)

LAMBDAS=[float(x) for x in os.environ.get("TILT_LAMBDAS","0.15,0.30,0.60").split(",")]
for lam in LAMBDAS:
    tag=str(lam).replace(".","p")
    for base_sub, out_sub in [("hl","hl_tilt_%s"%tag), ("hl_residrev","hl_residrev_tilt_%s"%tag)]:
        d=pd.read_parquet(CONV/base_sub/"v0full_hl60.parquet"); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
        d=d.merge(L,on=["symbol","open_time"],how="left"); d["okx_z"]=d["okx_z"].fillna(0.0)
        d["pred"]=d["pred"]-lam*d["okx_z"]; d=d.drop(columns=["okx_z"])
        od=CONV/out_sub; od.mkdir(parents=True,exist_ok=True)
        d.to_parquet(od/"v0full_hl60.parquet")
        import shutil; shutil.copy(CONV/"hl/fullflow_hl60.parquet", od/"fullflow_hl60.parquet")
    print(f"built tilt lambda={lam} -> hl_tilt_{tag}, hl_residrev_tilt_{tag}")
print("DONE")
