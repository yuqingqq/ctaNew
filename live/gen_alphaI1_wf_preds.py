"""I1 (model-feature) evaluation of the alpha-set survivors under the NEW framework, in the v4 frame:
single per-symbol RidgeCV, BOTH legs, RESIDUAL target, features = V0_LEAN + RR + <candidate alpha>.
Candidates = best survivor per earlier frame (beta-neutral versions — matches the residual-target frame):
  wq036 (pred-orthogonal winner), q158_IMIN60/q158_RANK60 (feature-frame survivors),
  alpha082 (hot-bull conditional), alpha065, alpha054 (side specialist).
Output: hl_i1_<name>/v0full_hl60.parquet (fold col included). Screen with tip_accuracy_v2 vs
hl_tgt_res_long — NOT with average IC.
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
CAND={"wq036":("alpha101","wq036"),"imin60":("alpha158","q158_IMIN60"),"rank60":("alpha158","q158_RANK60"),
      "a082":("alpha191","alpha082"),"a065":("alpha191","alpha065"),"a054":("alpha191","alpha054")}

PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)
PAN["resid_rev_3"]=(-a.transform(lambda s:s.shift(1).rolling(3).sum())).fillna(0.0)
g=PAN.groupby("open_time"); sd=g["alpha_vs_btc_realized"].transform("std").replace(0,np.nan)
PAN["z_res"]=((PAN["alpha_vs_btc_realized"]-g["alpha_vs_btc_realized"].transform("mean"))/sd).clip(-10,10)
for set_ in {v[0] for v in CAND.values()}:
    cols=[v[1] for v in CAND.values() if v[0]==set_]
    f=pd.read_parquet(REPO/f"data/ml/cache/{set_}_factors_betaneut.parquet",columns=["symbol","open_time"]+cols)
    f["open_time"]=pd.to_datetime(f["open_time"],utc=True)
    PAN=PAN.merge(f,on=["symbol","open_time"],how="left")
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
print(f"rows={len(PAN)} syms={PAN.symbol.nunique()}",flush=True)

def gen(feats,out,dropna_col):
    P=PAN.dropna(subset=[dropna_col]) if dropna_col else PAN
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=P[(P.exit_time<fc)&P["z_res"].notna()]; te=P[(P.open_time>=c0)&(P.open_time<c1)]
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
    o=pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): o[c]=pd.to_datetime(o[c],utc=True)
    p=REPO/f"live/state/convexity/{out}/v0full_hl60.parquet"; p.parent.mkdir(parents=True,exist_ok=True); o.to_parquet(p)
    print(f"{out}: {o['symbol'].nunique()} syms {len(o)} rows",flush=True)

for tag,(set_,col) in CAND.items():
    gen(V0+RR+[col], f"hl_i1_{tag}", col)
print("I1GENDONE")
