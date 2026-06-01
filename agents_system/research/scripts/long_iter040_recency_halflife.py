"""LONG-PRED iter-040 — Recency-weighting (exponential half-life) sweep on the long leg.

Distinct from iter-037 (which was retrain CADENCE, equal-weight expanding window). Here we
keep monthly retrain but add EXPONENTIAL sample weighting: weight = exp(-days_back/halflife).
Tests whether downweighting old-regime data (TRAINING CONTAMINATION mechanism) lets the
per-sym Ridge learn the CURRENT regime's relationship — potentially flipping the fragile
signal's sign in H2.

Half-life sweep: {30, 60, 90, 180, inf(=equal)} days. Monthly retrain, expanding window,
same x6 preproc + RidgeCV + 17 V0 features + target_z + 1d embargo.

Metric: long selection alpha (top-K=5 vs cycle median) per H1 / H2 / ALL, per half-life.
Want: some hl lifts H2 WITHOUT killing H1, stable across the sweep (not a single-hl fluke).
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
BASELINE = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
OUT = REPO/"agents_system/research/outputs/iter040"; OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
FEAT = x6.BASE + x6.COHORT_EXTRAS
EMB = pd.Timedelta(days=1)

H1S=pd.Timestamp("2025-10-04",tz="UTC"); H2S=pd.Timestamp("2026-01-22",tz="UTC"); H2E=pd.Timestamp("2026-05-26",tz="UTC")
K=5
CUTS=[pd.Timestamp(t,tz="UTC") for t in
      ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01",
       "2026-03-01","2026-04-01","2026-05-01","2026-05-26"]]
HALFLIVES=[30,60,90,180,None]   # None = equal weight (= iter-037 baseline)

def gen_preds(panel, halflife):
    preds=[]
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
                y=gtr["target_z"].to_numpy()
                if halflife:
                    db=(t_end-gtr["open_time"]).dt.total_seconds().to_numpy()/86400.0
                    w=np.exp(-db/halflife)
                    m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xtr,y,sample_weight=w)
                else:
                    m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xtr,y)
                models[sym]=m; ss[sym]=s; hh[sym]=h
            except: pass
        for sym,gv in test.groupby("symbol"):
            if sym not in models: continue
            Xv=x6.apply_preproc(gv,FEAT,ss[sym],hh[sym])
            out=gv[["symbol","open_time","return_pct"]].copy(); out["pred"]=models[sym].predict(Xv)
            preds.append(out)
    return pd.concat(preds,ignore_index=True)

def long_alpha(df,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    cyc=sub.groupby("open_time").apply(
        lambda g:(g.nlargest(K,"pred")["return_pct"].mean()-g["return_pct"].median())*1e4 if len(g)>=2*K else np.nan).dropna()
    a=cyc.values; m=a.mean(); t=m/(a.std()/np.sqrt(len(a))) if a.std()>0 else np.nan
    return m,t

def main():
    t0=time.time()
    print("=== iter-040: Recency half-life sweep (long leg) ===\n", flush=True)
    cols=["symbol","open_time","exit_time","return_pct","target_z"]+FEAT
    panel=pd.read_parquet(PANEL, columns=cols)
    panel["open_time"]=pd.to_datetime(panel["open_time"],utc=True)
    panel["exit_time"]=pd.to_datetime(panel["exit_time"],utc=True)
    panel=panel[(panel["open_time"].dt.hour%4==0)&(panel["open_time"].dt.minute==0)]
    panel=panel.sort_values(["symbol","open_time"]).reset_index(drop=True)

    # production 7-mo-fold baseline (reference)
    bl=pd.read_parquet(BASELINE, columns=["symbol","open_time","return_pct","pred"])
    bl["open_time"]=pd.to_datetime(bl["open_time"],utc=True)
    bl=bl[(bl.open_time>=H1S)&(bl.open_time<H2E)&(bl.open_time.dt.hour%4==0)&(bl.open_time.dt.minute==0)]
    bh1=long_alpha(bl,H1S,H2S); bh2=long_alpha(bl,H2S,H2E); ba=long_alpha(bl,H1S,H2E)
    print(f"  REFERENCE production 7-mo-fold: H1 {bh1[0]:+.1f}  H2 {bh2[0]:+.1f}  ALL {ba[0]:+.1f}\n", flush=True)

    print(f"  {'half-life':<12}{'H1 bps':>9}{'(t)':>7}{'H2 bps':>9}{'(t)':>7}{'ALL bps':>9}", flush=True)
    print("  "+"-"*54)
    rows=[]
    for hl in HALFLIVES:
        mp=gen_preds(panel,hl)
        h1=long_alpha(mp,H1S,H2S); h2=long_alpha(mp,H2S,H2E); al=long_alpha(mp,H1S,H2E)
        lbl=f"{hl}d" if hl else "equal(∞)"
        print(f"  {lbl:<12}{h1[0]:>+8.1f}{h1[1]:>+7.1f}{h2[0]:>+8.1f}{h2[1]:>+7.1f}{al[0]:>+8.1f}", flush=True)
        rows.append({"hl":lbl,"H1":h1[0],"H2":h2[0],"ALL":al[0]})
    print(f"\n  (want: an hl that lifts H2 vs equal(∞) WITHOUT killing H1, stable across sweep)")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
