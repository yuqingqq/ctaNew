"""LOOP iter-09 — strengthen SELECTION skill: add cross-sectional RANK features to the xs_z
per-sym model (extensible; injects cross-sectional info the per-sym model structurally lacks).

iter-08 placebo: per-cycle selection vs random only +2.5 bps (t~0.4, n.s.). The xs_z model ranks
broadly (IC +0.045) but extreme-top vs random is weak. Add within-cycle rank features
(rank of corr/rvol/atr/return/idio_vol) so the per-sym model sees its cross-sectional standing.

Compare per-sym Ridge xs_z target, recency60:
  base   : V0 features
  +xsrank: V0 + within-cycle rank features
Metric: H1/H2 long alpha + IC + MATCHED-PLACEBO edge (top-3 vs random, the real skill measure).
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location("x6",REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6=importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
PANEL=REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
H1S=pd.Timestamp("2025-10-04",tz="UTC"); H2S=pd.Timestamp("2026-01-22",tz="UTC"); H2E=pd.Timestamp("2026-05-26",tz="UTC")
FIT=pd.Timestamp("2025-10-02",tz="UTC"); K=3; HL=60.0; V0=x6.BASE+x6.COHORT_EXTRAS
XSR_SRC=["corr_to_btc_1d","rvol_7d","atr_pct","return_1d","ret_3d","idio_vol_to_btc_1d"]

def la(df,c,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    cyc=sub.groupby("open_time").apply(lambda x:(x.nlargest(K,c)["return_pct"].mean()-x["return_pct"].median())*1e4 if len(x)>=2*K else np.nan).dropna()
    a=cyc.values; return a.mean()
def ic(df,c,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]; v=[spearmanr(x[c],x["return_pct"])[0] for _,x in sub.groupby("open_time") if len(x)>=20]; return np.nanmean(v)
def placebo(df,c,s,e,seed=0):
    rng=np.random.default_rng(seed); sub=df[(df.open_time>=s)&(df.open_time<e)]; mo,pl=[],[]
    for ot,g in sub.groupby("open_time"):
        if len(g)<10: continue
        med=g["return_pct"].median(); mo.append(g.nlargest(K,c)["return_pct"].mean()-med)
        rv=g["return_pct"].values; pl.append(np.mean([rng.choice(rv,K,replace=False).mean()-med for _ in range(150)]))
    d=(np.array(mo)-np.array(pl))*1e4; return d.mean(), d.mean()/(d.std()/np.sqrt(len(d))) if d.std()>0 else np.nan

def main():
    t0=time.time(); print("=== LOOP iter-09: cross-sectional rank features on xs_z model ===\n",flush=True)
    pan=pd.read_parquet(PANEL,columns=["symbol","open_time","exit_time","return_pct"]+V0)
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"]).reset_index(drop=True)
    gc=pan.groupby("open_time"); mu=gc["return_pct"].transform("mean"); sd=gc["return_pct"].transform("std").replace(0,np.nan)
    pan["xs_z"]=((pan["return_pct"]-mu)/sd).clip(-10,10)
    xsr=[]
    for f in XSR_SRC:
        c="xsr_"+f; pan[c]=gc[f].rank(pct=True); xsr.append(c)
    tr=pan[pan.exit_time<FIT].copy(); te=pan[(pan.open_time>=H1S)&(pan.open_time<=H2E)].copy()
    t_end=tr["open_time"].max()

    def persym(feats):
        models,ss,hh={},{},{}
        for sym,g in tr.groupby("symbol"):
            if len(g)<300 or g["xs_z"].notna().sum()<300: continue
            s,h=x6.fit_preproc(g,feats); X=x6.apply_preproc(g,feats,s,h)
            w=np.exp(-((t_end-g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL); m=g["xs_z"].notna().values
            try: models[sym]=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X[m],g["xs_z"].to_numpy()[m],sample_weight=w[m]); ss[sym]=s; hh[sym]=h
            except: pass
        out=[]
        for sym,g in te.groupby("symbol"):
            if sym not in models: continue
            X=x6.apply_preproc(g,feats,ss[sym],hh[sym]); o=g[["symbol","open_time","return_pct"]].copy(); o["pred"]=models[sym].predict(X); out.append(o)
        return pd.concat(out,ignore_index=True)

    print(f"  {'config':<12}{'H2 long':>9}{'H2 IC':>9}{'H1 plac edge(t)':>18}{'H2 plac edge(t)':>18}")
    print("  "+"-"*64)
    for label,feats in [("base_V0",V0),("V0+xsrank",V0+xsr)]:
        d=persym(feats); h2=la(d,"pred",H2S,H2E); i2=ic(d,"pred",H2S,H2E)
        pe1,pt1=placebo(d,"pred",H1S,H2S); pe2,pt2=placebo(d,"pred",H2S,H2E)
        print(f"  {label:<12}{h2:>+8.1f}{i2:>+9.4f}{pe1:>+12.1f}(t{pt1:+.1f}){pe2:>+12.1f}(t{pt2:+.1f})",flush=True)
        d[["symbol","open_time","return_pct","pred"]].to_parquet(REPO/f"agents_system/research/outputs/loop/iter09_{label}.parquet")
    print(f"\n  WIN = +xsrank raises matched-placebo edge/t (real selection skill) vs base.")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
