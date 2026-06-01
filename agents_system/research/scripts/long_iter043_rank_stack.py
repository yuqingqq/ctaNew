"""LONG-PRED iter-043 — Cross-sectional rank-stack second layer (proper info-recovery).

Combine the model's orthogonal-alpha signal with the defensive axes the target strips,
in cross-sectional RANK space (scale-free, retains both axes):

  score = w0*rank(pred) + w1*rank(corr_to_btc) - w2*rank(rvol) - w3*rank(atr) - w4*rank(vol_ratio_recent)

Learn w0..w4 WALK-FORWARD by linear regression of raw-forward-return cross-sectional rank
on the 5 rank features (pooled over the trailing training window, refit each month).
Apply OOS, rank by score, top-K longs.

ACCEPTANCE BARS (both must pass):
  (1) weight STABILITY across walk-forward folds (signs consistent; w0>0, defensive signs as expected)
  (2) aggregate long alpha beats BOTH model-only AND parameter-free defensive on 3-way split

If weights are unstable -> reject, use parameter-free regime switch instead (decisive either way).
Reports per-slice long alpha for: model-only, defensive-only, rank-stack; + learned weights per fold.
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
H1S=pd.Timestamp("2025-10-04",tz="UTC"); H2S=pd.Timestamp("2026-01-22",tz="UTC"); H2E=pd.Timestamp("2026-05-26",tz="UTC")
VAL_E=pd.Timestamp("2025-12-01",tz="UTC"); INT_E=pd.Timestamp("2026-01-22",tz="UTC")
K=5; CPD=6
# monthly refit cutoffs for learning the stack weights (expanding window, all data < cut)
CUTS=[pd.Timestamp(t,tz="UTC") for t in
      ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
RANKF=["pred","corr_to_btc_1d","rvol_7d","atr_pct","vol_ratio_recent"]   # raw cols to rank
SIGN ={"pred":+1,"corr_to_btc_1d":+1,"rvol_7d":-1,"atr_pct":-1,"vol_ratio_recent":-1}  # expected weight sign

def add_ranks(d):
    g=d.groupby("open_time")
    for f in RANKF:
        d["r_"+f]=g[f].rank(pct=True)-0.5
    d["r_fwd"]=g["return_pct"].rank(pct=True)-0.5   # target: raw fwd return rank
    return d

def long_alpha(df, score_col, s, e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    cyc=sub.groupby("open_time").apply(
        lambda g:(g.nlargest(K,score_col)["return_pct"].mean()-g["return_pct"].median())*1e4 if len(g)>=2*K else np.nan).dropna()
    a=cyc.values; m=a.mean(); t=m/(a.std()/np.sqrt(len(a))) if a.std()>0 else np.nan
    return m,t

def main():
    t0=time.time()
    print("=== iter-043: Cross-sectional rank-stack second layer ===\n", flush=True)
    p=pd.read_parquet(PREDS, columns=["symbol","open_time","return_pct","pred"])
    p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
    p=p[(p.open_time>=H1S)&(p.open_time<H2E)&(p.open_time.dt.hour%4==0)&(p.open_time.dt.minute==0)]
    pf=pd.read_parquet(PANEL, columns=["symbol","open_time","corr_to_btc_1d","rvol_7d","atr_pct","idio_vol_to_btc_1d"])
    pf["open_time"]=pd.to_datetime(pf["open_time"],utc=True)
    pf=pf.sort_values(["symbol","open_time"]).reset_index(drop=True)
    g=pf.groupby("symbol",group_keys=False)
    pf["vol_ratio_recent"]=pf["idio_vol_to_btc_1d"]/g["idio_vol_to_btc_1d"].transform(lambda x:x.rolling(180,min_periods=90).mean().shift(1)).replace(0,np.nan)
    d=p.merge(pf,on=["symbol","open_time"],how="left").dropna(subset=RANKF)
    d=add_ranks(d)
    rcols=["r_"+f for f in RANKF]
    print(f"  {len(d):,} rows\n", flush=True)

    # ---- walk-forward learn w0..w4 (refit monthly, expanding window) ----
    print("=== Learned stack weights per monthly refit (sign check) ===\n")
    print(f"  {'fit<':<12}{'w0_pred':>9}{'w1_corr':>9}{'w2_rvol':>9}{'w3_atr':>9}{'w4_vr':>9}{'n_train':>9}")
    d["stack_score"]=np.nan
    weights_hist=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]
        # learn on all data with open_time < c0 (expanding, strictly past)
        tr=d[d.open_time<c0]
        if len(tr)<2000:   # fold-0 has no past OOS data -> use a flat parameter-free fallback (equal signs)
            w=np.array([SIGN[f] for f in RANKF],dtype=float)
        else:
            X=tr[rcols].values; y=tr["r_fwd"].values
            # ridge-ish least squares (tiny L2 for stability), no intercept (ranks centered)
            XtX=X.T@X + 1e-3*np.eye(X.shape[1]); w=np.linalg.solve(XtX, X.T@y)
        weights_hist.append(w)
        te=d[(d.open_time>=c0)&(d.open_time<c1)]
        d.loc[te.index,"stack_score"]=te[rcols].values@w
        wl=", ".join(f"{x:>+.2f}" for x in w)
        print(f"  {c0.date()!s:<12}{w[0]:>+9.2f}{w[1]:>+9.2f}{w[2]:>+9.2f}{w[3]:>+9.2f}{w[4]:>+9.2f}{len(tr):>9}")

    # stability check
    Wm=np.array(weights_hist[1:])   # skip fold-0 fallback
    print(f"\n  weight stability (folds with real fit, n={len(Wm)}):")
    ok=True
    for j,f in enumerate(RANKF):
        signs=np.sign(Wm[:,j]); consistent=(signs==SIGN[f]).mean()
        flag="OK" if consistent>=0.7 else "UNSTABLE"
        if consistent<0.7: ok=False
        print(f"    {f:<20} expected sign {SIGN[f]:+d}: {100*consistent:>3.0f}% of folds match  [{flag}]")
    print(f"  => weights {'STABLE' if ok else 'UNSTABLE (favor regime-switch)'}")

    # ---- also build defensive-only score (parameter-free) for comparison ----
    d["def_score"]=d["r_corr_to_btc_1d"]-d["r_rvol_7d"]-d["r_atr_pct"]-d["r_vol_ratio_recent"]

    # ---- per-slice long alpha: model-only vs defensive-only vs rank-stack ----
    print(f"\n=== Long selection alpha (bps vs median), per slice ===\n")
    print(f"  {'slice':<10}{'model-only':>12}{'defensive':>11}{'rank-stack':>12}")
    print("  "+"-"*45)
    for sl,s,e in [("VAL/H1a",H1S,VAL_E),("INT/H1b",VAL_E,INT_E),("FIN/H2",H2S,H2E),("ALL",H1S,H2E)]:
        mm,_=long_alpha(d,"pred",s,e); dd,_=long_alpha(d,"def_score",s,e); rs,_=long_alpha(d,"stack_score",s,e)
        print(f"  {sl:<10}{mm:>+11.1f}{dd:>+10.1f}{rs:>+11.1f}")
    print(f"\n  ACCEPT if rank-stack beats BOTH model-only and defensive on INT/H1b + FIN/H2 (OOS) AND weights stable.")

    d[["symbol","open_time","stack_score","pred","return_pct"]].to_parquet(REPO/"agents_system/research/outputs/iter043_stack.parquet")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
