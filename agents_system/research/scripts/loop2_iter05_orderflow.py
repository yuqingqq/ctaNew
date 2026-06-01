"""LOOP2 iter-05 (PUSH HARDER) — order-flow features on the xs_z target. The ONE genuinely orthogonal
free-ish signal (5m aggTrade flow: tfi, signed_vol_z, vpin, kyle_lambda, large-trade share, aggressor
ratio). Untested on this strategy. Can it raise BASE per-cycle IC (the binding constraint)?

PIT: aggregate 5m→4h (window ending at T), use trailing (shift≥1) flow features only — no look-ahead.
Test on the 71 flow-available symbols: per-sym Ridge xs_z + recency60, V0 vs V0+flow, IC + L/S edge H1/H2.
"""
import sys, time, glob, importlib.util
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
FIT=pd.Timestamp("2025-10-02",tz="UTC");K=3;HL=60.0;V0=x6.BASE+x6.COHORT_EXTRAS

def build_flow():
    rows=[]
    for fp in sorted(glob.glob(str(REPO/"data/ml/cache/flow_*.parquet"))):
        sym=Path(fp).stem.replace("flow_","")
        try: f=pd.read_parquet(fp)
        except: continue
        f=f[f.index.notna()].copy()
        # 4h aggregation, window labeled at right edge (bar ending at T covers [T-4h,T))
        agg=pd.DataFrame({
            "tfi": f["tfi"].resample("4h",label="right",closed="right").mean(),
            "sv_z": f["signed_volume_z"].resample("4h",label="right",closed="right").mean(),
            "vpin": f["vpin"].resample("4h",label="right",closed="right").mean(),
            "kyle": f["kyle_lambda"].resample("4h",label="right",closed="right").mean(),
            "aggr": f["aggressor_count_ratio"].resample("4h",label="right",closed="right").mean(),
            "lgvol": f["large_trade_volume"].resample("4h",label="right",closed="right").sum(),
            "totvol": f["total_volume"].resample("4h",label="right",closed="right").sum(),
            "bv": f["buy_volume"].resample("4h",label="right",closed="right").sum(),
            "sv": f["sell_volume"].resample("4h",label="right",closed="right").sum(),
        })
        agg["lg_share"]=agg["lgvol"]/agg["totvol"].replace(0,np.nan)
        agg["bs_imb"]=(agg["bv"]-agg["sv"])/(agg["bv"]+agg["sv"]).replace(0,np.nan)
        agg=agg[(agg.index.hour%4==0)]
        # PIT trailing features: shift(1) (use only flow strictly before decision T) + 1d trailing mean
        feats={}
        for c in ["tfi","sv_z","vpin","kyle","aggr","lg_share","bs_imb"]:
            feats["fl_"+c]=agg[c].shift(1)
            feats["fl_"+c+"_1d"]=agg[c].rolling(6,min_periods=3).mean().shift(1)
        ff=pd.DataFrame(feats); ff["symbol"]=sym; ff["open_time"]=ff.index
        rows.append(ff.reset_index(drop=True))
    F=pd.concat(rows,ignore_index=True)
    return F, [c for c in F.columns if c.startswith("fl_")]

def ic(df,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)];v=[spearmanr(x["pred"],x["return_pct"])[0] for _,x in sub.groupby("open_time") if len(x)>=15];return np.nanmean(v)
def lsedge(df,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    cyc=sub.groupby("open_time").apply(lambda x:(x.nlargest(K,"pred")["return_pct"].mean()+(x["return_pct"].median()-x.nsmallest(K,"pred")["return_pct"].mean())-x["return_pct"].median())*1e4 if len(x)>=2*K else np.nan).dropna()
    return cyc.mean()

def main():
    t0=time.time(); print("=== LOOP2 iter-05: ORDER-FLOW features on xs_z ===\n",flush=True)
    F,flowcols=build_flow(); print(f"  flow: {F.symbol.nunique()} syms, {len(flowcols)} features [{time.time()-t0:.0f}s]",flush=True)
    pan=pd.read_parquet(PANEL,columns=["symbol","open_time","exit_time","return_pct"]+V0)
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True);pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)]
    F["open_time"]=pd.to_datetime(F["open_time"],utc=True)
    pan=pan.merge(F,on=["symbol","open_time"],how="inner")  # flow-sym subset only
    gc=pan.groupby("open_time");mu=gc["return_pct"].transform("mean");sd=gc["return_pct"].transform("std").replace(0,np.nan)
    pan["xs_z"]=((pan["return_pct"]-mu)/sd).clip(-10,10)
    pan=pan.sort_values(["symbol","open_time"]).reset_index(drop=True)
    print(f"  flow-sym panel: {len(pan):,} rows, {pan.symbol.nunique()} syms\n",flush=True)
    tr=pan[pan.exit_time<FIT].copy();te=pan[(pan.open_time>=H1S)&(pan.open_time<=H2E)].copy();t_end=tr["open_time"].max()
    def run(feats):
        models,ss,hh={},{},{}
        for sym,g in tr.groupby("symbol"):
            if len(g)<300 or g["xs_z"].notna().sum()<300: continue
            s,h=x6.fit_preproc(g,feats);X=x6.apply_preproc(g,feats,s,h)
            w=np.exp(-((t_end-g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL);m=g["xs_z"].notna().values
            try: models[sym]=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X[m],g["xs_z"].to_numpy()[m],sample_weight=w[m]);ss[sym]=s;hh[sym]=h
            except: pass
        out=[]
        for sym,g in te.groupby("symbol"):
            if sym not in models: continue
            X=x6.apply_preproc(g,feats,ss[sym],hh[sym]);o=g[["symbol","open_time","return_pct"]].copy();o["pred"]=models[sym].predict(X);out.append(o)
        return pd.concat(out,ignore_index=True)
    print(f"  {'featset':<16}{'H1 IC':>9}{'H2 IC':>9}{'H1 LS':>9}{'H2 LS':>9}")
    print("  "+"-"*52)
    # subsets: all flow, and the orthogonal-looking ones
    core=[c for c in flowcols if any(k in c for k in ['tfi','sv_z','bs_imb','vpin','kyle','lg_share','aggr'])]
    for label,feats in [("V0(flowsyms)",V0),("V0+flow_all",V0+flowcols),("V0+flow_core",V0+core)]:
        d=run(feats);print(f"  {label:<16}{ic(d,H1S,H2S):>+9.4f}{ic(d,H2S,H2E):>+9.4f}{lsedge(d,H1S,H2S):>+9.1f}{lsedge(d,H2S,H2E):>+9.1f}",flush=True)
    print(f"\n  WIN = flow raises base IC in BOTH halves vs V0(flowsyms). (the binding-constraint lever)")
    print(f"\nDONE [{time.time()-t0:.0f}s]")
if __name__=="__main__": main()
