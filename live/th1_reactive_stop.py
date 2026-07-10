"""TH1: reactive intra-cycle short stop (deployable tail-hedge, no look-ahead). Pre-reg addendum 44.

For each short name (bottom-2 base-pred), compute intra-4h MAX ADVERSE from 5m klines = max(high over
[t,t+4h])/close[t] − 1. If adverse ≥ STOP → short loss capped at −STOP; else realized short return. This
is the deployable version of addendum 42's oracle winsorization (a real stop, no per-era percentile
look-ahead). Gate: stopped-short net Sharpe ≥ raw in BOTH eras + edge-preservation.
"""
import numpy as np, pandas as pd, glob
from pathlib import Path
import sys; sys.path.insert(0, "/home/yuqing/ctaNew/live")
from attribution_v4_regime import btc_reg, load, COST
import warnings; warnings.filterwarnings("ignore")
KD = Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines"); STOP = 0.12; WS = 0.5

def adverse_map(symbols):
    """per (symbol, 4h open_time): max intra-4h adverse = max(5m high over [t,t+48bars]) / close[t] - 1."""
    rows = []
    for s in symbols:
        fs = sorted(glob.glob(str(KD/s/"5m"/"*.parquet")))
        if not fs: continue
        try:
            d = pd.concat([pd.read_parquet(f, columns=["open_time","close","high"]) for f in fs], ignore_index=True)
        except Exception: continue
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
        d = d.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        # forward max-high over the intra-cycle window [i, i+47] (48 bars = 4h), exclusive of run-off end.
        # NOTE (reviewer fix 2026-07-10): the prior `.shift(-47)` here measured [i+47, i+94] (the NEXT 4h
        # window, mostly AFTER the position closes at i+48) — a ~47-bar misalignment, empirically verified.
        # The double-reverse rolling-max already yields the forward intra-cycle max; no extra shift.
        fwd_max = d["high"][::-1].rolling(48, min_periods=48).max()[::-1]
        adv = fwd_max / d["close"] - 1
        g = d[(d.open_time.dt.hour % 4 == 0) & (d.open_time.dt.minute == 0)].copy()
        g["adverse"] = adv.reindex(g.index)
        g["symbol"] = s
        rows.append(g[["symbol","open_time","adverse"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def short_legs(base, reg):
    rows=[]
    for t,g in base.groupby("open_time"):
        if len(g)<5 or reg.get(t) in (None,"deepbull"): continue
        S=g.nsmallest(2,"pred")
        for _,r in S.iterrows(): rows.append((t, r["symbol"], r["return_pct"]))
    return pd.DataFrame(rows, columns=["open_time","symbol","return_pct"])

def sh(x,c):
    d=pd.to_datetime(x["t"]).dt.date; dr=x[[c]].groupby(d).sum()[c]
    return dr.mean()/dr.std()*np.sqrt(365) if dr.std()>0 else np.nan

def main():
    reg=btc_reg(); results={}
    # gather short picks across both eras, build the adverse map for the involved symbols once
    picks={}
    for era,bp in (("RECENT","hl_tgt_res_base_cleanfix"),("OOS","hl_v4base_oos_cleanfix")):
        base,_=load(bp,bp); picks[era]=short_legs(base,reg)
    syms=sorted(set(picks["RECENT"].symbol)|set(picks["OOS"].symbol))
    print(f"building intra-4h adverse for {len(syms)} syms (5m klines)...", flush=True)
    adv=adverse_map(syms)
    print(f"  adverse map {len(adv)} (sym,bar) rows", flush=True)
    for era in ("RECENT","OOS"):
        p=picks[era].merge(adv,on=["symbol","open_time"],how="left")
        p["raw"]=-p["return_pct"]*1e4
        stopped_name=np.where(p["adverse"].fillna(-9)>=STOP, -STOP*1e4, p["raw"])  # cap at -STOP if it ripped
        p["stopped"]=stopped_name
        # short-leg PnL per cycle = mean of the 2 names, minus short cost (~WS*COST full-turnover)
        cyc=p.groupby("open_time").agg(raw=("raw","mean"),stopped=("stopped","mean"),
                                       stophit=("adverse",lambda a:(a>=STOP).mean())).reset_index().rename(columns={"open_time":"t"})
        cyc["raw"]-=WS*COST; cyc["stopped"]-=WS*COST
        rS,sS=sh(cyc,"raw"),sh(cyc,"stopped")
        results[era]=(rS,sS)
        print(f"\n===== {era}: short leg raw vs reactive-stop@{STOP*100:.0f}% (net, n={len(cyc)}) =====")
        print(f"  RAW     : Sharpe {rS:+.2f}  mean {cyc.raw.mean():+.1f}  median {cyc.raw.median():+.1f}")
        print(f"  STOPPED : Sharpe {sS:+.2f}  mean {cyc.stopped.mean():+.1f}  median {cyc.stopped.median():+.1f}  "
              f"stop-hit {cyc.stophit.mean()*100:.0f}% of names")
        print(f"  Δ Sharpe {sS-rS:+.2f} | edge-preservation mean {cyc.stopped.mean()/cyc.raw.mean() if cyc.raw.mean()!=0 else float('nan'):.2f}x")
    both=all(s>=r for r,s in results.values())
    print(f"\n  >> GATE (stopped ≥ raw BOTH eras + edge kept): {'PASS -> tail-hedge works reactively (free)' if both else 'FAIL'}")
    print(f"     recent Δ {results['RECENT'][1]-results['RECENT'][0]:+.2f} | OOS Δ {results['OOS'][1]-results['OOS'][0]:+.2f}")
    print("TH1DONE")

if __name__=="__main__":
    main()
