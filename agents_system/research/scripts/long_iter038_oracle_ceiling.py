"""LONG-PRED iter-038 — Oracle ceiling + characteristics of the optimal longs.

Part A: theoretical ceiling. Per cycle, oracle long = top-K=5 by REALIZED return
(perfect foresight) paired with the held beta-neutral basket hedge. Compare to the
model's top-K long. Report long selection alpha (vs cycle median) and capture ratio,
per H1 / H2 / ALL. The gap = alpha left on the table.

Part B: what do the optimal longs LOOK LIKE? Pool cycles; winner = top-K by realized
return, loser = bottom-K. For each entry-time feature, compare winner vs loser via
rank-AUC = P(feature_winner > feature_loser). AUC≈0.5 => winners indistinguishable from
losers in that feature (ceiling unreachable from it); AUC far from 0.5 => learnable signal.
Done separately for H1 (model worked) and H2 (model failed) — the key contrast.

Uses production per-sym Ridge preds (what the system runs) + panel features.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"

H1S=pd.Timestamp("2025-10-04",tz="UTC"); H2S=pd.Timestamp("2026-01-22",tz="UTC"); H2E=pd.Timestamp("2026-05-26",tz="UTC")
K=5; CYCLES_PER_YEAR=6*365
FEATS=["return_1d","ret_3d","funding_rate","funding_rate_z_7d","rvol_7d","btc_rvol_7d",
       "atr_pct","idio_vol_to_btc_1d","bars_since_high","corr_to_btc_1d",
       "beta_to_btc_change_5d","obv_z_1d","vwap_slope_96"]

def long_alpha(df, key, s, e):
    """top-K by `key` long alpha vs cycle median, + Sharpe of the alpha series."""
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    cyc=sub.groupby("open_time").apply(
        lambda g:(g.nlargest(K,key)["return_pct"].mean()-g["return_pct"].median()) if len(g)>=2*K else np.nan).dropna()
    a=cyc.values*1e4; m=a.mean(); t=m/(a.std()/np.sqrt(len(a))) if a.std()>0 else np.nan
    sh=m/a.std()*np.sqrt(CYCLES_PER_YEAR) if a.std()>0 else np.nan
    return m,t,sh,len(a)

def main():
    t0=time.time()
    print("=== iter-038: Oracle ceiling + optimal-long characteristics ===\n", flush=True)
    p=pd.read_parquet(PREDS, columns=["symbol","open_time","return_pct","pred"])
    p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
    p=p[(p.open_time>=H1S)&(p.open_time<H2E)&(p.open_time.dt.hour%4==0)&(p.open_time.dt.minute==0)]
    pf=pd.read_parquet(PANEL, columns=["symbol","open_time"]+FEATS)
    pf["open_time"]=pd.to_datetime(pf["open_time"],utc=True)
    d=p.merge(pf,on=["symbol","open_time"],how="left")
    print(f"  {len(d):,} rows\n", flush=True)

    # ---- Part A: oracle ceiling vs model ----
    print("=== A: long selection alpha (bps vs median) + alpha-Sharpe ===\n")
    print(f"  {'period':<6}{'MODEL bps':>11}{'(Sh)':>7}   {'ORACLE bps':>11}{'(Sh)':>7}{'  capture':>9}")
    print("  "+"-"*58)
    for pl,s,e in [("H1",H1S,H2S),("H2",H2S,H2E),("ALL",H1S,H2E)]:
        mm,mt,msh,mn=long_alpha(d,"pred",s,e)
        om,ot,osh,on=long_alpha(d,"return_pct",s,e)
        cap=100*mm/om if om>0 else np.nan
        print(f"  {pl:<6}{mm:>+10.1f}{msh:>+7.2f}   {om:>+10.1f}{osh:>+7.2f}{cap:>+8.0f}%")
    print(f"\n  (ORACLE = top-K by realized return, perfect foresight = ceiling; capture = model/oracle)")

    # ---- Part B: characteristics of winners vs losers ----
    print(f"\n=== B: do optimal longs share characteristics? (rank-AUC winner vs loser) ===\n")
    print("  AUC = P(feature higher for oracle-winner than oracle-loser). 0.50=indistinguishable.\n")
    def auc_table(s,e,label):
        sub=d[(d.open_time>=s)&(d.open_time<e)]
        win_rows,los_rows=[],[]
        for ot,g in sub.groupby("open_time"):
            if len(g)<2*K: continue
            win_rows.append(g.nlargest(K,"return_pct")); los_rows.append(g.nsmallest(K,"return_pct"))
        W=pd.concat(win_rows); L=pd.concat(los_rows)
        print(f"  --- {label} (n_win={len(W):,}) ---")
        print(f"    {'feature':<24}{'win mean':>11}{'los mean':>11}{'rank-AUC':>10}")
        rows=[]
        for f in FEATS:
            wv=W[f].dropna().values; lv=L[f].dropna().values
            if len(wv)<50 or len(lv)<50: continue
            # rank-AUC via Mann-Whitney U / (n1*n2)
            allv=np.concatenate([wv,lv]); r=pd.Series(allv).rank().values
            rw=r[:len(wv)].sum(); auc=(rw-len(wv)*(len(wv)+1)/2)/(len(wv)*len(lv))
            rows.append((f,wv.mean(),lv.mean(),auc))
        for f,wm,lm,auc in sorted(rows,key=lambda x:-abs(x[3]-0.5)):
            flag="<<" if abs(auc-0.5)>0.05 else ""
            print(f"    {f:<24}{wm:>+10.4f}{lm:>+10.4f}{auc:>9.3f} {flag}")
        # overall separability: max |AUC-0.5|
        msep=max(abs(a-0.5) for *_,a in rows)
        print(f"    => max |AUC-0.5| = {msep:.3f} ({'SEPARABLE (learnable)' if msep>0.05 else 'INDISTINGUISHABLE (ceiling unreachable)'})\n")
    auc_table(H1S,H2S,"H1 (model worked)")
    auc_table(H2S,H2E,"H2 (model failed)")

    print(f"DONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
