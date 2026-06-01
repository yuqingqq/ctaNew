"""LONG-PRED iter-037 — Monthly-retrain test (does fresh model recover long alpha?).

iter-036 confirmed staleness: long alpha decays ~+19 (fresh, mo0-1) -> ~+8 (stale, mo5+),
7/8 folds, paired t=-2.75. Production uses ~7-month folds -> mostly stale.

TEST: retrain per-sym Ridge MONTHLY (expanding window) over the OOS window so every
prediction comes from a <=1-month-old model. Compare long selection alpha (top-K=5 vs
cycle median) in H1 / H2 against the baseline 7-month-fold preds on identical cycles.

Reuses x6 preproc + RidgeCV (same as production trainer), 17 V0 features, target_z, 1d embargo.
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
BASELINE_PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
OUT = REPO/"agents_system/research/outputs/iter037"; OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
FEAT = x6.BASE + x6.COHORT_EXTRAS
EMB = pd.Timedelta(days=1)

H1S=pd.Timestamp("2025-10-04",tz="UTC"); H2S=pd.Timestamp("2026-01-22",tz="UTC"); H2E=pd.Timestamp("2026-05-26",tz="UTC")
K=5
# monthly retrain cutoffs (model trained on data < cutoff-embargo, predicts [cutoff, next))
CUTS = [pd.Timestamp(t,tz="UTC") for t in
        ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01",
         "2026-03-01","2026-04-01","2026-05-01","2026-05-26"]]

def long_alpha(df, s, e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    cyc=sub.groupby("open_time").apply(
        lambda g:(g.nlargest(K,"pred")["return_pct"].mean()-g["return_pct"].median())*1e4 if len(g)>=2*K else np.nan).dropna()
    a=cyc.values; m=a.mean(); t=m/(a.std()/np.sqrt(len(a))) if a.std()>0 else np.nan
    return m,t,len(a)

def main():
    t0=time.time()
    print("=== iter-037: Monthly-retrain test ===\n", flush=True)
    cols=["symbol","open_time","exit_time","return_pct","target_z"]+FEAT
    panel=pd.read_parquet(PANEL, columns=cols)
    panel["open_time"]=pd.to_datetime(panel["open_time"],utc=True)
    panel["exit_time"]=pd.to_datetime(panel["exit_time"],utc=True)
    panel=panel[(panel["open_time"].dt.hour%4==0)&(panel["open_time"].dt.minute==0)]
    print(f"  panel {len(panel):,} rows; monthly retrain {len(CUTS)-1} segments\n", flush=True)

    all_preds=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]
        fit_cut=c0-EMB
        train=panel[(panel["exit_time"]<fit_cut)&panel["target_z"].notna()]
        test=panel[(panel["open_time"]>=c0)&(panel["open_time"]<c1)]
        models,ss,hh={},{},{}
        for sym,gtr in train.groupby("symbol"):
            if len(gtr)<300: continue
            try:
                s,h=x6.fit_preproc(gtr,FEAT)
                Xtr=x6.apply_preproc(gtr,FEAT,s,h)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xtr,gtr["target_z"].to_numpy())
                models[sym]=m; ss[sym]=s; hh[sym]=h
            except: pass
        for sym,gv in test.groupby("symbol"):
            if sym not in models: continue
            Xv=x6.apply_preproc(gv,FEAT,ss[sym],hh[sym])
            out=gv[["symbol","open_time","return_pct"]].copy(); out["pred"]=models[sym].predict(Xv)
            all_preds.append(out)
        print(f"  segment {c0.date()}→{c1.date()}: trained {len(models)} syms on {len(train):,} rows [{time.time()-t0:.0f}s]", flush=True)
    mp=pd.concat(all_preds,ignore_index=True).sort_values(["open_time","symbol"])
    mp.to_parquet(OUT/"monthly_preds.parquet")

    # baseline (7-month fold) preds on identical window
    bl=pd.read_parquet(BASELINE_PREDS, columns=["symbol","open_time","return_pct","pred"])
    bl["open_time"]=pd.to_datetime(bl["open_time"],utc=True)
    bl=bl[(bl.open_time>=H1S)&(bl.open_time<H2E)&(bl.open_time.dt.hour%4==0)&(bl.open_time.dt.minute==0)]

    print(f"\n=== LONG SELECTION ALPHA: monthly-retrain vs baseline 7-mo-fold ===\n")
    print(f"  {'period':<6}{'baseline bps':>14}{'(t)':>7}   {'monthly bps':>13}{'(t)':>7}{'  lift':>8}")
    print("  "+"-"*64)
    for pl,s,e in [("H1",H1S,H2S),("H2",H2S,H2E),("ALL",H1S,H2E)]:
        bm,bt,bn=long_alpha(bl,s,e); mm,mt,mn=long_alpha(mp,s,e)
        print(f"  {pl:<6}{bm:>+12.1f}{bt:>+7.1f}   {mm:>+11.1f}{mt:>+7.1f}{mm-bm:>+8.1f}")
    print(f"\n  (lift>0 = monthly retrain recovers long alpha vs stale fold model)")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
