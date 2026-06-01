"""LONG-PRED iter-039 — Parameter-free defensive two-stage long selection.

Lead (iter-038): oracle winners share a STABLE cross-regime signature — LOWER vol
(rvol/atr/idio) + HIGHER corr_to_btc. Model has these features but underweights them
(trained on mean-reversion target -> picks crashy bounce-candidates instead).

DISCRETE, NO continuous knob to overfit (lesson: tuned λ/margins fail nested-OOS):
  Stage 1: top-N long candidates by model pred
  Stage 2: among them, pick K=5 most DEFENSIVE by a parameter-free composite rank
           defensive_score = rank(corr_to_btc) - rank(rvol_7d) - rank(atr_pct)  (within the N)

Compare to baseline (top-K by pred) long selection alpha (vs cycle median) on the
3-way split (H1a calibrate-window only used to confirm not-N-sensitive; H1b + H2 are
the OOS checks). Report an N-sweep to show it's not N-tuned. Defensive filter uses ONLY
PIT features already in the panel.

Also a sanity oracle-direction check: does the defensive composite ALONE (ignore pred)
rank winners better than random? (confirms the signature, not the model).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"

# 3-way split (same as iter-032/035)
VAL_S,VAL_E=pd.Timestamp("2025-10-04",tz="UTC"),pd.Timestamp("2025-12-01",tz="UTC")   # H1a calibrate
INT_S,INT_E=pd.Timestamp("2025-12-01",tz="UTC"),pd.Timestamp("2026-01-22",tz="UTC")   # H1b check1
FIN_S,FIN_E=pd.Timestamp("2026-01-22",tz="UTC"),pd.Timestamp("2026-05-26",tz="UTC")   # H2 check2
SLICES=[("VAL/H1a",VAL_S,VAL_E),("INT/H1b",INT_S,INT_E),("FIN/H2",FIN_S,FIN_E)]
K=5; CYCLES_PER_YEAR=6*365
DEF_FEATS=["corr_to_btc_1d","rvol_7d","atr_pct"]

def defensive_score(g):
    # within-group parameter-free composite: high corr, low vol
    return (g["corr_to_btc_1d"].rank(pct=True)
            - g["rvol_7d"].rank(pct=True)
            - g["atr_pct"].rank(pct=True))

def alpha_series(d, mode, N, s, e):
    sub=d[(d.open_time>=s)&(d.open_time<e)]
    vals=[]
    for ot,g in sub.groupby("open_time"):
        if len(g)<2*K: continue
        med=g["return_pct"].median()
        if mode=="baseline":
            sel=g.nlargest(K,"pred")
        elif mode=="defensive":
            cand=g.nlargest(N,"pred")              # stage 1: model's top-N
            if len(cand)<K: continue
            cand=cand.assign(_d=defensive_score(cand))
            sel=cand.nlargest(K,"_d")              # stage 2: most defensive among them
        elif mode=="defensive_only":
            sel=g.assign(_d=defensive_score(g)).nlargest(K,"_d")  # ignore pred entirely
        vals.append(sel["return_pct"].mean()-med)
    a=np.array(vals)*1e4
    if len(a)==0: return (np.nan,)*4
    m=a.mean(); t=m/(a.std()/np.sqrt(len(a))) if a.std()>0 else np.nan
    sh=m/a.std()*np.sqrt(CYCLES_PER_YEAR) if a.std()>0 else np.nan
    return m,t,sh,len(a)

def main():
    t0=time.time()
    print("=== iter-039: Parameter-free defensive two-stage long selection ===\n", flush=True)
    p=pd.read_parquet(PREDS, columns=["symbol","open_time","return_pct","pred"])
    p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
    p=p[(p.open_time>=VAL_S)&(p.open_time<FIN_E)&(p.open_time.dt.hour%4==0)&(p.open_time.dt.minute==0)]
    pf=pd.read_parquet(PANEL, columns=["symbol","open_time"]+DEF_FEATS)
    pf["open_time"]=pd.to_datetime(pf["open_time"],utc=True)
    d=p.merge(pf,on=["symbol","open_time"],how="left").dropna(subset=DEF_FEATS)
    print(f"  {len(d):,} rows\n", flush=True)

    # baseline reference
    print("=== Long selection alpha (bps vs median), per slice ===\n")
    print(f"  {'slice':<9}{'baseline':>10}{'(Sh)':>7}", end="")
    Ns=[8,10,12,15,20]
    for N in Ns: print(f"{'defN='+str(N):>11}", end="")
    print()
    print("  "+"-"*(26+11*len(Ns)))
    for slabel,s,e in SLICES:
        bm,bt,bsh,bn=alpha_series(d,"baseline",None,s,e)
        row=f"  {slabel:<9}{bm:>+9.1f}{bsh:>+7.2f}"
        for N in Ns:
            dm,dt,dsh,dn=alpha_series(d,"defensive",N,s,e)
            row+=f"{dm:>+11.1f}"
        print(row)
    print(f"\n  (each defN cell = defensive two-stage long alpha; compare to baseline col)")

    # lift table (defensive - baseline) per slice, to read generalization
    print(f"\n=== Lift vs baseline (bps); want POSITIVE & STABLE across INT/H1b + FIN/H2 ===\n")
    print(f"  {'slice':<9}", end="")
    for N in Ns: print(f"{'defN='+str(N):>11}", end="")
    print()
    for slabel,s,e in SLICES:
        bm,_,_,_=alpha_series(d,"baseline",None,s,e)
        row=f"  {slabel:<9}"
        for N in Ns:
            dm,_,_,_=alpha_series(d,"defensive",N,s,e)
            row+=f"{dm-bm:>+11.1f}"
        print(row)

    # defensive-only (ignore model) — confirms the signature stands alone
    print(f"\n=== Defensive-ONLY (ignore pred) long alpha — confirms signature ===\n")
    print(f"  {'slice':<9}{'def-only bps':>13}{'(Sh)':>7}{'  baseline':>11}")
    for slabel,s,e in SLICES:
        dm,dt,dsh,dn=alpha_series(d,"defensive_only",None,s,e)
        bm,_,_,_=alpha_series(d,"baseline",None,s,e)
        print(f"  {slabel:<9}{dm:>+12.1f}{dsh:>+7.2f}{bm:>+10.1f}")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
