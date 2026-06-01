"""LONG-PRED iter-045 — Regime classifier on pred_disp (model conviction).

iter-044: pred_disp is the best PIT switch signal (Q4-Q1 long-alpha spread +49.5 bps,
known AT decision time = zero lag). High pred_disp -> model has conviction -> model mode;
low pred_disp -> model flat -> defensive mode.

Classifier: per cycle, if pred_disp >= THR -> long = top-K by pred (model mode)
                       else            -> long = top-K by defensive composite (defensive mode)

THR calibrated PIT on H1a (e.g. a low percentile of H1a pred_disp so H1a stays in model mode),
then FROZEN and verified on H1b + H2. Also sweep fixed THR to show robustness.
Defensive composite (iter-039/042): rank(corr) - rank(rvol) - rank(atr) - rank(vol_ratio_recent).

Compare long alpha per slice: model-only, defensive-only, SWITCH. Accept if SWITCH beats
both on OOS (H1b + H2).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
H1S=pd.Timestamp("2025-10-04",tz="UTC"); VAL_E=pd.Timestamp("2025-12-01",tz="UTC")
INT_E=pd.Timestamp("2026-01-22",tz="UTC"); H2S=INT_E; H2E=pd.Timestamp("2026-05-26",tz="UTC")
K=5
DEF=["corr_to_btc_1d","rvol_7d","atr_pct","vol_ratio_recent"]

def main():
    t0=time.time()
    print("=== iter-045: Regime classifier on pred_disp ===\n", flush=True)
    p=pd.read_parquet(PREDS, columns=["symbol","open_time","return_pct","pred"])
    p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
    p=p[(p.open_time>=H1S)&(p.open_time<H2E)&(p.open_time.dt.hour%4==0)&(p.open_time.dt.minute==0)]
    pf=pd.read_parquet(PANEL, columns=["symbol","open_time","corr_to_btc_1d","rvol_7d","atr_pct","idio_vol_to_btc_1d"])
    pf["open_time"]=pd.to_datetime(pf["open_time"],utc=True)
    pf=pf.sort_values(["symbol","open_time"]).reset_index(drop=True)
    g=pf.groupby("symbol",group_keys=False)
    pf["vol_ratio_recent"]=pf["idio_vol_to_btc_1d"]/g["idio_vol_to_btc_1d"].transform(lambda x:x.rolling(180,min_periods=90).mean().shift(1)).replace(0,np.nan)
    # pred_disp on FULL prediction universe (matches iter-044; NOT the defensive-filtered subset)
    full_disp=p.groupby("open_time")["pred"].std()
    d=p.merge(pf,on=["symbol","open_time"],how="left").dropna(subset=DEF)
    d["pred_disp"]=d["open_time"].map(full_disp)
    gc=d.groupby("open_time")
    d["def_score"]=(gc["corr_to_btc_1d"].rank(pct=True)-gc["rvol_7d"].rank(pct=True)
                    -gc["atr_pct"].rank(pct=True)-gc["vol_ratio_recent"].rank(pct=True))

    # H1a pred_disp distribution to set THR honestly
    h1a_disp=d[d.open_time<VAL_E].groupby("open_time")["pred_disp"].first()
    print(f"  H1a pred_disp: p10={h1a_disp.quantile(.10):.2f} p25={h1a_disp.quantile(.25):.2f} median={h1a_disp.median():.2f}")
    thr_pit=float(h1a_disp.quantile(.20))   # PIT-honest: below this = 'flat' regime (rare in H1a)
    print(f"  PIT-calibrated THR (H1a p10) = {thr_pit:.2f}\n", flush=True)

    def long_alpha(df,mode,thr,s,e):
        sub=df[(df.open_time>=s)&(df.open_time<e)]; vals=[]
        for ot,g in sub.groupby("open_time"):
            if len(g)<2*K: continue
            med=g["return_pct"].median()
            if mode=="model": sel=g.nlargest(K,"pred")
            elif mode=="defensive": sel=g.nlargest(K,"def_score")
            elif mode=="switch":
                key="pred" if g["pred_disp"].iloc[0]>=thr else "def_score"
                sel=g.nlargest(K,key)
            vals.append(sel["return_pct"].mean()-med)
        a=np.array(vals)*1e4
        return a.mean() if len(a) else np.nan

    SL=[("VAL/H1a",H1S,VAL_E),("INT/H1b",VAL_E,INT_E),("FIN/H2",H2S,H2E),("ALL",H1S,H2E)]
    print(f"=== Long alpha (bps vs median) — model / defensive / SWITCH(THR={thr_pit:.2f}) ===\n")
    print(f"  {'slice':<10}{'model':>9}{'defensive':>11}{'SWITCH':>9}{'%model-mode':>13}")
    print("  "+"-"*52)
    for sl,s,e in SL:
        mm=long_alpha(d,"model",None,s,e); dd=long_alpha(d,"defensive",None,s,e); sw=long_alpha(d,"switch",thr_pit,s,e)
        sub=d[(d.open_time>=s)&(d.open_time<e)]
        pct_model=100*(sub.groupby("open_time")["pred_disp"].first()>=thr_pit).mean()
        print(f"  {sl:<10}{mm:>+8.1f}{dd:>+10.1f}{sw:>+8.1f}{pct_model:>12.0f}%")

    # robustness: sweep THR
    print(f"\n=== THR robustness sweep (SWITCH long alpha per slice) ===\n")
    print(f"  {'THR':<7}"+''.join(f"{sl[0]:>10}" for sl in SL))
    for thr in [0.6,0.8,1.0,1.2,1.5]:
        row=f"  {thr:<7}"
        for sl,s,e in SL: row+=f"{long_alpha(d,'switch',thr,s,e):>+10.1f}"
        print(row)
    print(f"\n  ACCEPT if SWITCH beats BOTH model & defensive on INT/H1b + FIN/H2, stable across THR.")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
