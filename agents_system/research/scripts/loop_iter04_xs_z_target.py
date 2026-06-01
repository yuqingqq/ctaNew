"""LOOP iter-04 (P1) — Target that keeps BOTH cross-sectional axis AND magnitude.

iter-03: rank target flips H2 positive but loses H1 tail (rank compresses magnitude).
Test per-cycle cross-sectional z (keeps relative magnitude -> extreme pumps still rank high)
and a winsorized variant. Per-sym Ridge, recency60. Find a target that gets H1 tail + H2 broad.

  resid_z      : control (production)
  raw_rank     : iter-03 winner-for-H2
  xs_z         : per-cycle (ret - cycle_mean)/cycle_std       [cross-sec + magnitude]
  xs_z_winsor  : winsorize ret per cycle at [5,95]% then xs-z  [tame outliers, keep order+mag]
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
FIT=pd.Timestamp("2025-10-02",tz="UTC"); K=5; HL=60.0; FEAT=x6.BASE+x6.COHORT_EXTRAS

def la(df,c,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    cyc=sub.groupby("open_time").apply(lambda x:(x.nlargest(K,c)["return_pct"].mean()-x["return_pct"].median())*1e4 if len(x)>=2*K else np.nan).dropna()
    a=cyc.values; return a.mean(),(a.mean()/(a.std()/np.sqrt(len(a))) if a.std()>0 else np.nan)
def ic(df,c,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    v=[spearmanr(x[c],x["return_pct"])[0] for _,x in sub.groupby("open_time") if len(x)>=20]; return np.nanmean(v)

def main():
    t0=time.time(); print("=== LOOP iter-04: xs-z target (magnitude + cross-sec) ===\n",flush=True)
    pan=pd.read_parquet(PANEL,columns=["symbol","open_time","exit_time","return_pct","target_z"]+FEAT)
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"]).reset_index(drop=True)
    gc=pan.groupby("open_time")
    pan["raw_rank"]=gc["return_pct"].rank(pct=True)
    mu=gc["return_pct"].transform("mean"); sd=gc["return_pct"].transform("std").replace(0,np.nan)
    pan["xs_z"]=((pan["return_pct"]-mu)/sd).clip(-10,10)
    rclip=gc["return_pct"].transform(lambda x:x.clip(x.quantile(.05),x.quantile(.95)))
    muw=rclip.groupby(pan["open_time"]).transform("mean"); sdw=rclip.groupby(pan["open_time"]).transform("std").replace(0,np.nan)
    pan["xs_z_winsor"]=((rclip-muw)/sdw).clip(-10,10)
    tr=pan[pan.exit_time<FIT].copy(); te=pan[(pan.open_time>=H1S)&(pan.open_time<=H2E)].copy()
    t_end=tr["open_time"].max(); w_all=np.exp(-((t_end-tr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)

    def persym(tcol):
        models,ss,hh={},{},{}
        for sym,g in tr.groupby("symbol"):
            if len(g)<300 or g[tcol].notna().sum()<300: continue
            s,h=x6.fit_preproc(g,FEAT); X=x6.apply_preproc(g,FEAT,s,h)
            idx=g.index.values; wsel=w_all[tr.index.get_indexer(idx)]; m=g[tcol].notna().values
            try: models[sym]=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X[m],g[tcol].to_numpy()[m],sample_weight=wsel[m]); ss[sym]=s; hh[sym]=h
            except: pass
        out=[]
        for sym,g in te.groupby("symbol"):
            if sym not in models: continue
            X=x6.apply_preproc(g,FEAT,ss[sym],hh[sym]); o=g[["symbol","open_time","return_pct"]].copy(); o["pred"]=models[sym].predict(X); out.append(o)
        return pd.concat(out,ignore_index=True)

    print(f"  {'target':<14}{'H1 long':>9}{'(t)':>7}{'H2 long':>9}{'(t)':>7}{'H1 IC':>9}{'H2 IC':>9}")
    print("  "+"-"*60)
    for tcol in ["target_z","raw_rank","xs_z","xs_z_winsor"]:
        d=persym(tcol); h1=la(d,"pred",H1S,H2S); h2=la(d,"pred",H2S,H2E); i1=ic(d,"pred",H1S,H2S); i2=ic(d,"pred",H2S,H2E)
        print(f"  {tcol:<14}{h1[0]:>+8.1f}{h1[1]:>+7.1f}{h2[0]:>+8.1f}{h2[1]:>+7.1f}{i1:>+9.4f}{i2:>+9.4f}",flush=True)
    print(f"\n  recency-A: H1 ~+34/+2.9sh, H2 ~+4. WIN = target with H1 long near resid AND H2 long/IC up.")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
