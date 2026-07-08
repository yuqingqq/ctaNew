"""Backtest candidate: add wq036 (WQ101 5-term candle/volume blend; pred-orthogonal margIC +0.023,
t +3.79, 4/4 folds, regime-sign-stable 2021-26 — see ALPHA_SETS_SURVEY.md) to V0_LEAN.
Row-matched baseline + treatment two-book WF preds, EXACT gen_alpha065 machinery (same cuts, HL=60,
xs_z target, per-symbol RidgeCV) so only the feature set differs. Two treatment variants:
raw wq036 (feature-frame faithful) and beta-neutralized wq036 (the validated idio signal).
  base_lean : V0_LEAN               -> hl_wq036base_lean/v0full_hl60.parquet
  long_lean : V0_LEAN+RR            -> hl_wq036long_lean/v0full_hl60.parquet
  base_bn   : V0_LEAN+wq036_bn      -> hl_wq036base_bn/v0full_hl60.parquet
  long_bn   : V0_LEAN+wq036_bn+RR   -> hl_wq036long_bn/v0full_hl60.parquet
  base_raw  : V0_LEAN+wq036_raw     -> hl_wq036base_raw/v0full_hl60.parquet
  long_raw  : V0_LEAN+wq036_raw+RR  -> hl_wq036long_raw/v0full_hl60.parquet
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0_LEAN=list(tt.V0_LEAN); EMB=pd.Timedelta(days=1); HL=60.0
RR=["resid_rev_2","resid_rev_3"]
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
      "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
_last=pd.read_parquet(tt.PANEL,columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS=CUTS+[_last["open_time"].max().normalize()+pd.Timedelta(days=1)]

PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0_LEAN)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum())
PAN["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
for c in RR: PAN[c]=PAN[c].fillna(0.0)
# merge wq036 raw + betaneut; drop rows where either is NaN so ALL six pred sets share EXACT rows
raw=pd.read_parquet(REPO/"data/ml/cache/alpha101_factors.parquet",columns=["symbol","open_time","wq036"]).rename(columns={"wq036":"wq036_raw"})
bn=pd.read_parquet(REPO/"data/ml/cache/alpha101_factors_betaneut.parquet",columns=["symbol","open_time","wq036"]).rename(columns={"wq036":"wq036_bn"})
for f in (raw,bn): f["open_time"]=pd.to_datetime(f["open_time"],utc=True)
PAN=PAN.merge(raw,on=["symbol","open_time"],how="left").merge(bn,on=["symbol","open_time"],how="left")
PAN=PAN.dropna(subset=["wq036_raw","wq036_bn"])
g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
print(f"rows={len(PAN)} (wq036 raw+bn merged, non-nan); syms={PAN.symbol.nunique()}",flush=True)

def gen(feats,outpath):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                        "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                        "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
            except Exception: pass
    out=pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    Path(outpath).parent.mkdir(parents=True,exist_ok=True); out.to_parquet(outpath)
    return out["symbol"].nunique(),len(out)

D=REPO/"live/state/convexity"
print("base_lean", gen(V0_LEAN,                    D/"hl_wq036base_lean/v0full_hl60.parquet"),flush=True)
print("long_lean", gen(V0_LEAN+RR,                 D/"hl_wq036long_lean/v0full_hl60.parquet"),flush=True)
print("base_bn  ", gen(V0_LEAN+["wq036_bn"],       D/"hl_wq036base_bn/v0full_hl60.parquet"),flush=True)
print("long_bn  ", gen(V0_LEAN+["wq036_bn"]+RR,    D/"hl_wq036long_bn/v0full_hl60.parquet"),flush=True)
print("base_raw ", gen(V0_LEAN+["wq036_raw"],      D/"hl_wq036base_raw/v0full_hl60.parquet"),flush=True)
print("long_raw ", gen(V0_LEAN+["wq036_raw"]+RR,   D/"hl_wq036long_raw/v0full_hl60.parquet"),flush=True)
print("GENDONE")
