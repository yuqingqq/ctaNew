"""Q1 (user idea 2026-06-03): TWO vol-SEGMENTED pooled models instead of 175 per-symbol models.
Rationale: global pooling failed (one slope across heterogeneous syms), per-symbol is noisy (+0.017 IC).
Pooling WITHIN a vol bucket = homogeneous dynamics + lots of rows -> may capture pooling's data-efficiency
WITHOUT global pooling's heterogeneity problem. pool_hi trains on top-80-rvol names, pool_lo on the rest;
each = one RidgeCV on V0 (no flow — flow adds 0). Predict all syms with both -> ab_split routes by its vol membership.
-> live/state/convexity/hl_segpool175/{fullflow_hl60=pool_hi, v0full_hl60=pool_lo}.parquet
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
V0 = [f for f in tt.V0]; EMB = pd.Timedelta(days=1); HL = 60.0; SPLIT_N = 80
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
OUT = REPO/"live/state/convexity/hl_segpool175"; OUT.mkdir(parents=True, exist_ok=True)
_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS = CUTS + [_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized","rvol_7d"]+[f for f in V0 if f!="rvol_7d"])
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)]
g = PAN.groupby("open_time"); sd = g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"] = ((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)

def hivol_asof(fit_cut, win=30):
    lo = fit_cut - pd.Timedelta(days=win)
    rv = PAN[(PAN.open_time>=lo)&(PAN.open_time<fit_cut)].groupby("symbol")["rvol_7d"].mean()
    return set(rv.sort_values(ascending=False).head(SPLIT_N).index)

def fit_pool(tr):
    Xtr = tr[V0].fillna(0).to_numpy(); mu,sg = Xtr.mean(0),Xtr.std(0); sg[sg==0]=1
    w = np.exp(-((tr["open_time"].max()-tr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
    m = RidgeCV(alphas=(0.1,1,10,100)).fit((Xtr-mu)/sg, tr["xs_z"].to_numpy(), sample_weight=w)
    return m, mu, sg

def predict(m, mu, sg, te):
    return m.predict((te[V0].fillna(0).to_numpy()-mu)/sg)

rec_hi, rec_lo = [], []
for i in range(len(CUTS)-1):
    c0,c1 = CUTS[i],CUTS[i+1]; fc = c0-EMB
    tr = PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te = PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
    if len(tr)<5000 or len(te)==0: continue
    hv = hivol_asof(fc)
    trhi = tr[tr.symbol.isin(hv)]; trlo = tr[~tr.symbol.isin(hv)]
    if len(trhi)<2000 or len(trlo)<2000: continue
    mhi = fit_pool(trhi); mlo = fit_pool(trlo)
    base = dict(symbol=te["symbol"].values, open_time=te["open_time"].values, alpha_A=te["alpha_vs_btc_realized"].values,
                return_pct=te["return_pct"].values, exit_time=te["exit_time"].values, fold=i)
    rec_hi.append(pd.DataFrame({**base, "pred":predict(*mhi, te)}))
    rec_lo.append(pd.DataFrame({**base, "pred":predict(*mlo, te)}))
for rec,name in [(rec_hi,"fullflow_hl60"),(rec_lo,"v0full_hl60")]:
    o = pd.concat(rec, ignore_index=True)
    for c in ("open_time","exit_time"): o[c]=pd.to_datetime(o[c],utc=True)
    o.to_parquet(OUT/f"{name}.parquet")
print(f"segmented-pooled preds -> {OUT} (pool_hi=fullflow, pool_lo=v0full)")
