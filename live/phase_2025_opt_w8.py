"""WAVE-8 (signal-layer probe on K=2): regenerate WF RidgeCV preds with HALF-LIFE + FEATURE-SUBSET variations and an
ENSEMBLE, replay each through K=2, vs canonical-preds K=2 (dense +2.02 / 2025 +1.04). Faithful gen() (HL=60 reproduces
the canonical baseline). HL is tuned-continuous => report nested-OOS-style caution; feature-subset/ensemble are structural.
Each variant: regen base(V0)+long(V0+RR) per-symbol RidgeCV over monthly folds -> mpit-curate (FRAC .52) -> replay K=2.
"""
import sys, os, time, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0); RR=["resid_rev_2","resid_rev_3"]; EMB=pd.Timedelta(days=1); ANN=np.sqrt(365)
OUT=REPO/"live/state/v3loop/opt2025_w8"; OUT.mkdir(parents=True,exist_ok=True)
LEDGER=REPO/"live/state/v3loop/opt2025/ledger.csv"
META=REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
t0=time.time()
CUTS=[pd.Timestamp(t,tz="UTC") for t in pd.date_range("2022-01-01","2026-06-01",freq="MS")]+[pd.Timestamp("2026-06-05",tz="UTC")]

# 175 panel (4h-sampled, all V0 feats incl bars_since_high_xs_rank), build resid_rev + xs_z
P=pd.read_parquet(META); P["open_time"]=pd.to_datetime(P["open_time"],utc=True); P["exit_time"]=pd.to_datetime(P["exit_time"],utc=True)
P=P.sort_values(["symbol","open_time"]).reset_index(drop=True)
a=P.groupby("symbol")["alpha_vs_btc_realized"]
P["resid_rev_2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)
P["resid_rev_3"]=(-a.transform(lambda s:s.shift(1).rolling(3).sum())).fillna(0.0)
g=P.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
P["xs_z"]=((P["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)

def gen(feats,HL,outp):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=P[(P.exit_time<fc)&P["xs_z"].notna()]; te=P[(P.open_time>=c0)&(P.open_time<c1)]
        if len(tr)==0 or len(te)==0: continue
        tend=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w=np.exp(-((tend-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                    "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                    "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
            except Exception: pass
    o=pd.concat(rec,ignore_index=True)
    for cc in ("open_time","exit_time"): o[cc]=pd.to_datetime(o[cc],utc=True)
    o.to_parquet(outp); return o

panv=P[["symbol","open_time","rvol_7d"]].copy(); FRAC=0.52
def excl_for(c0):
    lo=c0-pd.Timedelta(days=30); r=panv[(panv.open_time>=lo)&(panv.open_time<c0)].groupby("symbol")["rvol_7d"].mean().dropna()
    return set(r.sort_values(ascending=False).index[:int(round(FRAC*len(r)))])
EXCL={i:excl_for(CUTS[i]) for i in range(len(CUTS)-1)}
def mpit(infp,outfp):
    d=pd.read_parquet(infp); d["open_time"]=pd.to_datetime(d["open_time"],utc=True); keep=[]
    for i in range(len(CUTS)-1):
        w=d[(d.open_time>=CUTS[i])&(d.open_time<CUTS[i+1])]; keep.append(w[~w.symbol.isin(EXCL[i])])
    pd.concat(keep,ignore_index=True).to_parquet(outfp)

PROD=dict(COST_BPS_LEG="4.5",STRAT_K="2",STRAT_K_SHORT="2",STRAT_K_LONG="2",SIDE_MODE="default",XS_LEAN="1",
          CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",
          SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
def dsh(s): d=(s.fillna(0)/1e4).resample("1D").sum(); return float(d.mean()/d.std()*ANN) if d.std()>0 else np.nan
def replay_k2(tag, base_fp, long_fp):
    vd=OUT/tag; vd.mkdir(exist_ok=True)
    env=dict(os.environ); env.update(PROD); env.update(PYTHONPATH=str(REPO),CONVEXITY_STATE=str(vd),
        CONVEXITY_PREDS_PATH=str(base_fp),CONVEXITY_PREDS_LONG=str(long_fp),CONVEXITY_UNIVERSE_META=str(META))
    subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
        stdout=open(vd/"run.log","w"),stderr=subprocess.STDOUT)
    c=pd.read_csv(vd/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time").set_index("open_time")
    dense=dsh(c.loc['2025-01-01':'2026-06-04','pnl_bps']); y25=dsh(c.loc['2025-01-01':'2025-12-31','pnl_bps'])
    yrs=" ".join(f"y{yr}={dsh(gg['pnl_bps']):+.2f}" for yr,gg in c.groupby(c.index.year))
    print(f"RESULT_W8 {tag:20s} dense {dense:+.3f}  2025 {y25:+.3f}  | {yrs}  [{time.time()-t0:.0f}s]",flush=True)
    return dense,y25

# --- variants ---
# 1) HL=60 reproduction sanity (should ~match canonical K=2 +2.02)
for HL in [60.0,30.0,120.0,200.0]:
    gen(V0+RR,HL,OUT/f"long_hl{int(HL)}.parquet"); gen(V0,HL,OUT/f"short_hl{int(HL)}.parquet")
    mpit(OUT/f"short_hl{int(HL)}.parquet",OUT/f"base_hl{int(HL)}.parquet"); mpit(OUT/f"long_hl{int(HL)}.parquet",OUT/f"longc_hl{int(HL)}.parquet")
    replay_k2(f"hl{int(HL)}",OUT/f"base_hl{int(HL)}.parquet",OUT/f"longc_hl{int(HL)}.parquet")
# 2) feature subset: drop the 4 lowest-|IC| / most-redundant V0 feats (atr_pct, idio_vol_to_btc_1d, return_1d, corr_to_btc_1d)
DROP=["atr_pct","idio_vol_to_btc_1d","return_1d","corr_to_btc_1d"]
V0s=[f for f in V0 if f not in DROP]
gen(V0s+RR,60.0,OUT/"long_featsub.parquet"); gen(V0s,60.0,OUT/"short_featsub.parquet")
mpit(OUT/"short_featsub.parquet",OUT/"base_featsub.parquet"); mpit(OUT/"long_featsub.parquet",OUT/"longc_featsub.parquet")
replay_k2("featsub",OUT/"base_featsub.parquet",OUT/"longc_featsub.parquet")
# 3) ENSEMBLE: average HL30+HL60+HL120 preds (rank-noise reduction) -> base/long, then mpit+replay
def ens(tagfps,outfp):
    dfs=[pd.read_parquet(f) for f in tagfps]
    base=dfs[0][["symbol","open_time","alpha_A","return_pct","exit_time","fold"]].copy()
    base["open_time"]=pd.to_datetime(base["open_time"],utc=True); base["exit_time"]=pd.to_datetime(base["exit_time"],utc=True)
    preds=pd.concat([d.set_index(["symbol","open_time"])["pred"] for d in dfs],axis=1).mean(axis=1).rename("pred")
    base=base.set_index(["symbol","open_time"]).join(preds).reset_index().dropna(subset=["pred"]); base.to_parquet(outfp)
ens([OUT/f"short_hl{h}.parquet" for h in (30,60,120)],OUT/"short_ens.parquet")
ens([OUT/f"long_hl{h}.parquet" for h in (30,60,120)],OUT/"long_ens.parquet")
mpit(OUT/"short_ens.parquet",OUT/"base_ens.parquet"); mpit(OUT/"long_ens.parquet",OUT/"longc_ens.parquet")
replay_k2("ensemble",OUT/"base_ens.parquet",OUT/"longc_ens.parquet")
print("DONE phase_2025_opt wave-8",flush=True)
