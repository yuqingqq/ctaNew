"""Retrain the tip-pruned books from H1-decided keep-sets (live/state/longtail/prune_keepsets.json).
bear-LONG book = keep-set(bear_long); bear-SHORT book = keep-set(bear_short). Residual target, same WF as gen_oos_v4.
Used only in their regime by the eval. Outputs hl_tipprune_bearlong / hl_tipprune_bearshort.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = list(tt.V0); V0_LEAN=[f for f in V0 if not f.startswith("funding")]
EMB = pd.Timedelta(days=1); HL = 60.0
ks=json.load(open(REPO/"live/state/longtail/prune_keepsets.json"))
START=pd.Timestamp("2023-01-01",tz="UTC"); END=pd.Timestamp("2025-10-01",tz="UTC")
CUTS=list(pd.date_range(START,END,freq="MS",tz="UTC"))
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)
PAN["resid_rev_3"]=(-a.transform(lambda s:s.shift(1).rolling(3).sum())).fillna(0.0)
_g=PAN.groupby("open_time"); _sd=_g["alpha_vs_btc_realized"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["alpha_vs_btc_realized"]-_g["alpha_vs_btc_realized"].transform("mean"))/_sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
def gen(feats,outpath):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        if not len(tr) or not len(te): continue
        t_end=tr["open_time"].max()
        for sym,g in tr.groupby("symbol"):
            if len(g)<300: continue
            try:
                s,h=x6.fit_preproc(g,feats); X=x6.apply_preproc(g,feats,s,h)
                w=np.exp(-((t_end-g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,g["xs_z"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                    "pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
            except Exception: pass
    out=pd.concat(rec,ignore_index=True); out["open_time"]=pd.to_datetime(out["open_time"],utc=True)
    outpath.parent.mkdir(parents=True,exist_ok=True); out.to_parquet(outpath); return out["symbol"].nunique(),len(out)
print(f"bear-LONG keep ({len(ks['bear_long'])}): {ks['bear_long']}",flush=True)
print(gen(ks["bear_long"],REPO/"live/state/convexity/hl_tipprune_bearlong/v0full_hl60.parquet"),flush=True)
print(f"bear-SHORT keep ({len(ks['bear_short'])}): {ks['bear_short']}",flush=True)
print(gen(ks["bear_short"],REPO/"live/state/convexity/hl_tipprune_bearshort/v0full_hl60.parquet"),flush=True)
print("TIPPRUNEDONE")
