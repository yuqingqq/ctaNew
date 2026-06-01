"""LOOP2 iter-24 — UNIFIED full-flow single book. Now that ALL 175 universe syms have real
aggTrade flow, build ONE per-sym Ridge book on V0+flow for every flow-eligible sym (comparable
preds → no 2-book workaround). xs_z + recency60 + monthly-WF. Output preds in load_preds schema
(swap pred into baseline over OOS) so the production bot replay scores it directly.

Also emits a V0-only full-universe preds file (control) so the +3.02 one-price-full reproduces on
the SAME machinery, and a flow/price split tag for the contrast two-book.
"""
import sys, time, glob, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location("x6",REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6=importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
PANEL=REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
BASELINE=REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
OUT_FLOW=REPO/"live/state/convexity/unified_fullflow_preds.parquet"
OUT_V0  =REPO/"live/state/convexity/unified_v0full_preds.parquet"
V0=x6.BASE+x6.COHORT_EXTRAS; EMB=pd.Timedelta(days=1); HL=60.0
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]

def build_flow():
    rows=[]
    for fp in sorted(glob.glob(str(REPO/"data/ml/cache/flow_*.parquet"))):
        sym=Path(fp).stem.replace("flow_","")
        try: f=pd.read_parquet(fp)
        except: continue
        R=lambda c,how: getattr(f[c].resample("4h",label="right",closed="right"),how)()
        agg=pd.DataFrame({"tfi":R("tfi","mean"),"sv_z":R("signed_volume_z","mean"),"vpin":R("vpin","mean"),
            "kyle":R("kyle_lambda","mean"),"aggr":R("aggressor_count_ratio","mean"),
            "lgvol":R("large_trade_volume","sum"),"totvol":R("total_volume","sum"),"bv":R("buy_volume","sum"),"sv":R("sell_volume","sum")})
        agg["lg_share"]=agg["lgvol"]/agg["totvol"].replace(0,np.nan); agg["bs_imb"]=(agg["bv"]-agg["sv"])/(agg["bv"]+agg["sv"]).replace(0,np.nan)
        agg=agg[agg.index.hour%4==0]; feats={}
        for c in ["tfi","sv_z","vpin","kyle","aggr","lg_share","bs_imb"]:
            feats["fl_"+c]=agg[c].shift(1); feats["fl_"+c+"_1d"]=agg[c].rolling(6,min_periods=3).mean().shift(1)
        ff=pd.DataFrame(feats); ff["symbol"]=sym; ff["open_time"]=ff.index; rows.append(ff.reset_index(drop=True))
    F=pd.concat(rows,ignore_index=True); F["open_time"]=pd.to_datetime(F["open_time"],utc=True)
    return F,[c for c in F.columns if c.startswith("fl_")]

def gen(pan, flowcols, flowsyms, use_flow_flag, outpath):
    """use_flow_flag: if True, syms with populated flow train on V0+flow; else V0-only for everyone."""
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fit_cut=c0-EMB
        tr=pan[(pan.exit_time<fit_cut)&pan["xs_z"].notna()]; te=pan[(pan.open_time>=c0)&(pan.open_time<c1)]; t_end=tr["open_time"].max()
        for sym,g in tr.groupby("symbol"):
            if len(g)<300: continue
            uf = use_flow_flag and (sym in flowsyms) and g[flowcols].notna().any().all()
            feats=V0+flowcols if uf else V0
            try:
                s,h=x6.fit_preproc(g,feats);X=x6.apply_preproc(g,feats,s,h)
                w=np.exp(-((t_end-g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,g["xs_z"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): Xv=x6.apply_preproc(gte,feats,s,h); rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,"pred_n":m.predict(Xv),"used_flow":uf}))
            except: pass
        print(f"    {c0.date()}→{c1.date()} [{time.time()-T0:.0f}s]",flush=True)
    rec=pd.concat(rec,ignore_index=True); rec["open_time"]=pd.to_datetime(rec["open_time"],utc=True)
    bl=pd.read_parquet(BASELINE); bl["open_time"]=pd.to_datetime(bl["open_time"],utc=True); bl["exit_time"]=pd.to_datetime(bl["exit_time"],utc=True)
    oos=bl[bl.open_time>=CUTS[0]].copy().merge(rec,on=["symbol","open_time"],how="inner")
    oos["pred"]=oos["pred_n"]; oos=oos.drop(columns=["pred_n"])
    oos.to_parquet(outpath)
    nflow=int(oos["used_flow"].sum()) if "used_flow" in oos else 0
    return len(oos),oos.symbol.nunique(),nflow

def main():
    global T0; T0=time.time(); print("=== LOOP2 iter-24: UNIFIED full-flow single book ===\n",flush=True)
    F,flowcols=build_flow(); flowsyms=set(F.symbol.unique()); print(f"  flow: {len(flowsyms)} syms, {len(flowcols)} feats [{time.time()-T0:.0f}s]",flush=True)
    pan=pd.read_parquet(PANEL,columns=["symbol","open_time","exit_time","return_pct"]+V0)
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True);pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)]
    pan=pan.merge(F,on=["symbol","open_time"],how="left")
    gc=pan.groupby("open_time");mu=gc["return_pct"].transform("mean");sd=gc["return_pct"].transform("std").replace(0,np.nan)
    pan["xs_z"]=((pan["return_pct"]-mu)/sd).clip(-10,10); pan=pan.sort_values(["symbol","open_time"]).reset_index(drop=True)
    print(f"  panel {len(pan):,} rows, {pan.symbol.nunique()} syms\n  --- UNIFIED V0+flow ---",flush=True)
    n,ns,nf=gen(pan,flowcols,flowsyms,True,OUT_FLOW); print(f"  unified-fullflow: {n:,} rows {ns} syms, flow-used {nf:,} ({100*nf/n:.0f}%) -> {OUT_FLOW.name} [{time.time()-T0:.0f}s]\n  --- CONTROL V0-only ---",flush=True)
    n2,ns2,_=gen(pan,flowcols,flowsyms,False,OUT_V0); print(f"  v0-full(control): {n2:,} rows {ns2} syms -> {OUT_V0.name} [{time.time()-T0:.0f}s]")
    print(f"\nDONE [{time.time()-T0:.0f}s]")
if __name__=="__main__": main()
