"""LONG-PRED iter-041 — generate a SYSTEM-COMPATIBLE preds file from the 60d
recency-weighted monthly-retrain model, for replay validation.

Produces a parquet with the same columns the bot's load_preds expects
(symbol, open_time, alpha_A, return_pct, exit_time, pred, fold) but with `pred`
replaced by the 60d recency-weighted monthly-retrain predictions over the OOS window.
Rows without a recency pred keep the baseline pred (universe unchanged).
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
BASELINE = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
OUT_PREDS = REPO/"live/state/convexity/recency60_preds.parquet"

spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
FEAT = x6.BASE + x6.COHORT_EXTRAS
EMB = pd.Timedelta(days=1); HALFLIFE = 60.0
CUTS=[pd.Timestamp(t,tz="UTC") for t in
      ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01",
       "2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]

def main():
    t0=time.time()
    print(f"=== iter-041: generate 60d recency-weighted preds file ===\n", flush=True)
    cols=["symbol","open_time","exit_time","return_pct","target_z"]+FEAT
    panel=pd.read_parquet(PANEL, columns=cols)
    panel["open_time"]=pd.to_datetime(panel["open_time"],utc=True)
    panel["exit_time"]=pd.to_datetime(panel["exit_time"],utc=True)
    panel=panel[(panel["open_time"].dt.hour%4==0)&(panel["open_time"].dt.minute==0)]
    panel=panel.sort_values(["symbol","open_time"]).reset_index(drop=True)

    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fit_cut=c0-EMB
        train=panel[(panel["exit_time"]<fit_cut)&panel["target_z"].notna()]
        test=panel[(panel["open_time"]>=c0)&(panel["open_time"]<c1)]
        t_end=train["open_time"].max()
        models,ss,hh={},{},{}
        for sym,gtr in train.groupby("symbol"):
            if len(gtr)<300: continue
            try:
                s,h=x6.fit_preproc(gtr,FEAT); Xtr=x6.apply_preproc(gtr,FEAT,s,h)
                db=(t_end-gtr["open_time"]).dt.total_seconds().to_numpy()/86400.0
                w=np.exp(-db/HALFLIFE)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xtr,gtr["target_z"].to_numpy(),sample_weight=w)
                models[sym]=m; ss[sym]=s; hh[sym]=h
            except: pass
        for sym,gv in test.groupby("symbol"):
            if sym not in models: continue
            Xv=x6.apply_preproc(gv,FEAT,ss[sym],hh[sym])
            rec.append(pd.DataFrame({"symbol":sym,"open_time":gv["open_time"].values,
                                     "pred_rec":models[sym].predict(Xv)}))
        print(f"  segment {c0.date()}→{c1.date()}: {len(models)} syms [{time.time()-t0:.0f}s]", flush=True)
    rec=pd.concat(rec,ignore_index=True)

    # build system-compatible file: baseline OOS rows, swap pred where recency available
    bl=pd.read_parquet(BASELINE)
    bl["open_time"]=pd.to_datetime(bl["open_time"],utc=True)
    bl["exit_time"]=pd.to_datetime(bl["exit_time"],utc=True)
    oos=bl[bl["open_time"]>=CUTS[0]].copy()
    rec["open_time"]=pd.to_datetime(rec["open_time"],utc=True)
    m=oos.merge(rec,on=["symbol","open_time"],how="left")
    n_swap=m["pred_rec"].notna().sum()
    m["pred"]=m["pred_rec"].where(m["pred_rec"].notna(), m["pred"])
    m=m.drop(columns=["pred_rec"])
    m.to_parquet(OUT_PREDS)
    print(f"\n  wrote {OUT_PREDS}: {len(m):,} rows, swapped pred on {n_swap:,} ({100*n_swap/len(m):.0f}%)")
    print(f"DONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
