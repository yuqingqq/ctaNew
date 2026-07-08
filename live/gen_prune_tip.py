"""LOO prune test on the v4 model. For each weak-standalone feature, retrain (per-symbol RidgeCV WF, residual target)
WITHOUT it and measure the TIP (top-K/bottom-K L/S mean + per-cycle Sharpe). Drop = good if tip holds/improves.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0_LEAN); EMB=pd.Timedelta(days=1); HL=60.0; RR=["resid_rev_2","resid_rev_3"]
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
_last=pd.read_parquet(tt.PANEL,columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS=CUTS+[_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum()); PAN["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
for c in RR: PAN[c]=PAN[c].fillna(0.0)
PAN["fwd"]=a.transform(lambda s:s.shift(-1).rolling(6).sum().shift(-5))*1e4
g=PAN.groupby("open_time"); sd=g["alpha_vs_btc_realized"].transform("std").replace(0,np.nan)
PAN["z_res"]=((PAN["alpha_vs_btc_realized"]-g["alpha_vs_btc_realized"].transform("mean"))/sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
def gen(feats):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN["z_res"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["z_res"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"open_time":gte["open_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fwd":gte["fwd"].values}))
            except Exception: pass
    return pd.concat(rec,ignore_index=True).dropna(subset=["fwd"])
def tip(d,K):
    ls=[d[d.open_time==ot].nlargest(K,"pred")["fwd"].mean()-d[d.open_time==ot].nsmallest(K,"pred")["fwd"].mean() for ot in d.open_time.unique() if (d.open_time==ot).sum()>=2*K]
    s=pd.Series(ls); return s.mean(), s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan
BASE=V0+RR
WEAK=["return_1d","ret_3d","vwap_slope_96","autocorr_pctile_7d","obv_z_1d","resid_rev_2","resid_rev_3"]
variants=[("baseline(16)",BASE)]+[(f"-{w}",[x for x in BASE if x!=w]) for w in WEAK]+[("-resid_rev(both)",V0),("-all_weak(9)",[x for x in BASE if x not in WEAK])]
print(f"{'variant':22s} {'nfeat':>5s} {'tipK2_mean':>10s} {'tipK2_Sh':>8s} {'tipK3_Sh':>8s}  vs base",flush=True)
bs2=None
for lbl,fe in variants:
    d=gen(fe); m2,s2=tip(d,2); _,s3=tip(d,3)
    if lbl.startswith("baseline"): bs2=s2
    print(f"{lbl:22s} {len(fe):>5d} {m2:+10.1f} {s2:+8.2f} {s3:+8.2f}  {('' if bs2 is None else f'{s2-bs2:+.2f}Sh')}",flush=True)
print("PRUNEDONE")
