"""OOS test of the {ret_3d, autocorr} prune. Same WF pipeline, RESIDUAL target (v4), on held-out 2023-01..2025-09.
Generates baseline (V0_LEAN+RR) and pruned (minus ret_3d,autocorr), both residual-target single-model.
Outputs: hl_v4base_oos, hl_v4pruned_oos.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0_LEAN); EMB=pd.Timedelta(days=1); HL=60.0; RR=["resid_rev_2","resid_rev_3"]
CUTS=list(pd.date_range("2023-01-01","2025-10-01",freq="MS",tz="UTC"))
print(f"OOS CUTS {CUTS[0].date()}..{CUTS[-1].date()} ({len(CUTS)-1} folds)",flush=True)
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)
PAN["resid_rev_3"]=(-a.transform(lambda s:s.shift(1).rolling(3).sum())).fillna(0.0)
g=PAN.groupby("open_time"); sd=g["alpha_vs_btc_realized"].transform("std").replace(0,np.nan)
PAN["z_res"]=((PAN["alpha_vs_btc_realized"]-g["alpha_vs_btc_realized"].transform("mean"))/sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
def gen(feats,outpath):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN["z_res"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        if not len(tr) or not len(te): continue
        t_end=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["z_res"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                    "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                    "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
            except Exception: pass
    out=pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    outpath.parent.mkdir(parents=True,exist_ok=True); out.to_parquet(outpath)
    print(f"  -> {outpath.parent.name}: {out['symbol'].nunique()} syms {len(out)} rows",flush=True)
print("=== BASELINE (V0_LEAN+RR, residual) ===",flush=True)
gen(V0+RR, REPO/"live/state/convexity/hl_v4base_oos/v0full_hl60.parquet")
print("=== PRUNED (minus ret_3d,autocorr) ===",flush=True)
gen([c for c in V0+RR if c not in {"ret_3d","autocorr_pctile_7d"}], REPO/"live/state/convexity/hl_v4pruned_oos/v0full_hl60.parquet")
print("OOSGENDONE")
