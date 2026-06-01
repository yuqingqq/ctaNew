"""X114 — Efficacy regime over the EXTENDED 2021-2026 panel: how many sign-flips?

Make-or-break for the leading predictor (task #109): if the per-cycle cross-sec IC of the
V0 pred flips sign across multiple 2021-2026 regimes (esp 2022 LUNA/FTX bear), there are
flip EVENTS to train on. If it's positive-everywhere-except-2026 again, no learnable signal.

Reports: quarterly smoothed eff90 (sign), per-fold IC, and # of sign-flip episodes.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x113_ext_v0_preds.parquet"


def main():
    t0=time.time()
    print("=== X114 extended-panel efficacy regime ===\n", flush=True)
    d=pd.read_parquet(PREDS)
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]
    print(f"{d['symbol'].nunique()} syms, {d['open_time'].min().date()}→{d['open_time'].max().date()}, {len(d):,} rows(4h)\n", flush=True)

    recs=[]
    for ot,g in d.groupby("open_time"):
        gv=g.dropna(subset=["pred","alpha_A"])
        if len(gv)<6: continue
        recs.append((ot, spearmanr(gv["pred"],gv["alpha_A"]).correlation, g["fold"].iloc[0] if "fold" in g else -1))
    m=pd.DataFrame(recs,columns=["open_time","ic","fold"]).set_index("open_time").sort_index()
    m["eff90"]=m["ic"].rolling(90,min_periods=45).mean()

    print("=== quarterly smoothed eff90 (efficacy sign) ===")
    prev=None; flips=0
    for ts,v in m["eff90"].resample("QE").mean().items():
        sg = "+" if v>0 else "-"
        if prev is not None and np.sign(v)!=prev and not np.isnan(v): flips+=1
        if not np.isnan(v): prev=np.sign(v)
        print(f"  {ts.date()}  eff90={v:+.4f}  [{sg}]", flush=True)
    print(f"\n  quarterly sign-flip episodes: {flips}")

    print("\n=== per-fold mean per-cycle IC (regime diversity) ===")
    foldreg={0:"21BULL",1:"21-22SIDE",2:"22BEAR",3:"22-23BULL",4:"23BULL",5:"23-24BULL",6:"24-25BULL",7:"25BULL",8:"25-26BEAR"}
    for f,g in m.groupby("fold"):
        print(f"  f{int(f)} {foldreg.get(int(f),'?'):<10} meanIC={g['ic'].mean():+.4f}  median_eff90={g['eff90'].median():+.4f}  n={len(g)}", flush=True)

    # monthly sign to count finer flips
    msign=np.sign(m["ic"].rolling(60,min_periods=30).mean().resample("ME").mean())
    nflip=int((msign.diff().abs()>0).sum())
    print(f"\n  monthly(60-smooth) sign-flip count: {nflip}")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__=="__main__":
    main()
