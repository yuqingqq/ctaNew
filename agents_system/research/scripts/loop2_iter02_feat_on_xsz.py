"""LOOP2 iter-02 (P2.2) — raise BASE per-cycle IC: re-test feature groups on the xs_z target.
The target change (resid_z→xs_z) reopened the design space. Per-sym Ridge, recency60, xs_z target.
Measure per-cycle IC (the PnL driver per P2.1) + L/S edge, H1/H2, vs V0 baseline.
Groups (built from panel, PIT): momentum-divergence, funding-accumulators, vol-of-vol, longer-horizon ret.
Gate: a group that raises H2 IC robustly (both halves) advances to system replay + placebo.
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
H1S=pd.Timestamp("2025-10-04",tz="UTC");H2S=pd.Timestamp("2026-01-22",tz="UTC");H2E=pd.Timestamp("2026-05-26",tz="UTC")
FIT=pd.Timestamp("2025-10-02",tz="UTC");K=3;HL=60.0;V0=x6.BASE+x6.COHORT_EXTRAS;CPD=6

def ic(df,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]; v=[spearmanr(x["pred"],x["return_pct"])[0] for _,x in sub.groupby("open_time") if len(x)>=20]; return np.nanmean(v)
def lsedge(df,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    cyc=sub.groupby("open_time").apply(lambda x:(x.nlargest(K,"pred")["return_pct"].mean()+ (x["return_pct"].median()-x.nsmallest(K,"pred")["return_pct"].mean()) -x["return_pct"].median())*1e4 if len(x)>=2*K else np.nan).dropna()
    return cyc.mean()

def main():
    t0=time.time(); print("=== LOOP2 iter-02: features on xs_z target (raise base IC) ===\n",flush=True)
    pan=pd.read_parquet(PANEL,columns=["symbol","open_time","exit_time","return_pct"]+V0)
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True);pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"]).reset_index(drop=True)
    g=pan.groupby("symbol",group_keys=False)
    for dd in [7,30,60]: pan[f"ret_{dd}d"]=g["return_1d"].transform(lambda x:x.rolling(dd*CPD,min_periods=dd*CPD//2).sum().shift(1))
    pan["rvol_30d"]=g["rvol_7d"].transform(lambda x:x.rolling(180,min_periods=90).mean().shift(1))
    # groups
    pan["mom_div_1d_7d"]=pan["return_1d"]-pan["ret_7d"]/7; pan["mom_div_3d_30d"]=pan["ret_3d"]/3-pan["ret_30d"]/30
    pan["mom_div_7d_60d"]=pan["ret_7d"]/7-pan["ret_60d"]/60; pan["mom_accel"]=pan["return_1d"]-0.5*(pan["ret_7d"]/7+pan["ret_30d"]/30)
    momdiv=["mom_div_1d_7d","mom_div_3d_30d","mom_div_7d_60d","mom_accel"]
    pan["i_cum_f7"]=g["funding_rate"].transform(lambda x:x.rolling(42,min_periods=21).sum().shift(1))
    pan["i_cum_f30"]=g["funding_rate"].transform(lambda x:x.rolling(180,min_periods=90).sum().shift(1))
    pan["i_fvol7"]=g["funding_rate"].transform(lambda x:x.rolling(42,min_periods=21).std().shift(1))
    fund=["i_cum_f7","i_cum_f30","i_fvol7"]
    pan["k_rvolstd"]=g["rvol_7d"].transform(lambda x:x.rolling(180,min_periods=90).std().shift(1))
    pan["k_skew"]=g["return_1d"].transform(lambda x:x.rolling(180,min_periods=90).skew().shift(1))
    pan["k_volchg"]=pan["rvol_7d"]/pan["rvol_30d"].replace(0,np.nan)-1
    volvol=["k_rvolstd","k_skew","k_volchg"]
    longret=["ret_7d","ret_30d","ret_60d"]
    gc=pan.groupby("open_time");mu=gc["return_pct"].transform("mean");sd=gc["return_pct"].transform("std").replace(0,np.nan)
    pan["xs_z"]=((pan["return_pct"]-mu)/sd).clip(-10,10)
    tr=pan[pan.exit_time<FIT].copy();te=pan[(pan.open_time>=H1S)&(pan.open_time<=H2E)].copy();t_end=tr["open_time"].max()
    def run(feats):
        models,ss,hh={},{},{}
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300 or gg["xs_z"].notna().sum()<300: continue
            s,h=x6.fit_preproc(gg,feats);X=x6.apply_preproc(gg,feats,s,h)
            w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL);m=gg["xs_z"].notna().values
            try: models[sym]=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X[m],gg["xs_z"].to_numpy()[m],sample_weight=w[m]);ss[sym]=s;hh[sym]=h
            except: pass
        out=[]
        for sym,gg in te.groupby("symbol"):
            if sym not in models: continue
            X=x6.apply_preproc(gg,feats,ss[sym],hh[sym]);o=gg[["symbol","open_time","return_pct"]].copy();o["pred"]=models[sym].predict(X);out.append(o)
        return pd.concat(out,ignore_index=True)
    print(f"  {'featset':<16}{'H1 IC':>9}{'H2 IC':>9}{'H1 LS':>9}{'H2 LS':>9}")
    print("  "+"-"*52)
    for label,feats in [("V0",V0),("V0+momdiv",V0+momdiv),("V0+fund",V0+fund),("V0+volvol",V0+volvol),("V0+longret",V0+longret),("V0+ALL",V0+momdiv+fund+volvol+longret)]:
        d=run(feats);print(f"  {label:<16}{ic(d,H1S,H2S):>+9.4f}{ic(d,H2S,H2E):>+9.4f}{lsedge(d,H1S,H2S):>+9.1f}{lsedge(d,H2S,H2E):>+9.1f}",flush=True)
    print(f"\n  V0 baseline is the current production (xs_z). WIN = a group raises H2 IC in BOTH halves.")
    print(f"\nDONE [{time.time()-t0:.0f}s]")
if __name__=="__main__": main()
