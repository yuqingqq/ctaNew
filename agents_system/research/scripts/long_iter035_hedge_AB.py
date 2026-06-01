"""LONG-PRED iter-035 — Does the passive hedge improve previous performance? A2 vs A4 with REAL BTC.

Fixes iter-034's flaw (A2 and A4 used the same median proxy). Here:
  A2 hedge return = equal-weight ALT-basket short  = cross-sym median return (full turnover)
  A4 hedge return = REAL BTC 4h forward return     (single instrument, low turnover)

All variants restricted to cycles where real BTC data exists (Nov 2025 -> May 1 2026),
so the comparison is apples-to-apples. Long book = model top-K=5 by raw pred.

Questions:
  Q1: does A2 / A4 improve on the PREVIOUS strategy (A0 = model longs + model shorts)?
  Q2: A2 vs A4 — which hedge is better (net Sharpe on the true holdout FINAL)?
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"agents_system/research/outputs/iter025/preds_L1_V0_pooled.parquet"
BTC1H = REPO/"data/ml/cache/hl_klines_BTCUSDT_1h.parquet"

VAL_S,VAL_E = pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2025-12-01",tz="UTC")
INT_S,INT_E = pd.Timestamp("2025-12-01",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC")
FIN_S,FIN_E = pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC")
SLICES=[("VAL",VAL_S,VAL_E),("INTERIM",INT_S,INT_E),("FINAL",FIN_S,FIN_E)]
K=5; COST_RT_BPS=9.0; CYCLES_PER_YEAR=6*365

def build_btc_fwd4h():
    b=pd.read_parquet(BTC1H).sort_index()
    c=b["close"]; c4=c[c.index.hour%4==0]
    fwd=(c4.shift(-1)/c4-1).dropna().rename("btc_ret")
    fwd.index=fwd.index.tz_convert("UTC"); fwd.index.name="open_time"
    return fwd

def run(sub, variant, btc_map):
    rows=[]; prev_long=set(); prev_short=set()
    for ot,gc in sub.groupby("open_time"):
        if len(gc)<2*K: continue
        mkt=gc["return_pct"].median()
        longs=gc.nlargest(K,"pred"); long_ret=longs["return_pct"].mean()
        ls=set(longs["symbol"]); to_long=len(ls-prev_long)/K; prev_long=ls
        to_short=0.0; hedge_notional=0.0
        if variant=="A0":
            shorts=gc.nsmallest(K,"pred"); sr=shorts["return_pct"].mean()
            ss=set(shorts["symbol"]); to_short=len(ss-prev_short)/K; prev_short=ss
            gross=long_ret-sr
        elif variant=="A2":
            gross=long_ret-mkt; hedge_notional=1.0     # alt-basket re-struck each cycle
        elif variant=="A4":
            btc=btc_map.get(ot,np.nan)
            if np.isnan(btc): continue
            gross=long_ret-btc; hedge_notional=0.15     # hold BTC short, low roll cost
        cost=COST_RT_BPS*(to_long+to_short+hedge_notional)/1e4
        rows.append({"ot":ot,"net":gross-cost,"gross":gross,"long_alpha":long_ret-mkt,"mkt":mkt})
    return pd.DataFrame(rows)

def stat(res):
    if len(res)==0: return None
    net=res["net"].values*1e4; n=len(res); mkt=res["mkt"].values*1e4
    beta=np.polyfit(mkt,net,1)[0] if n>2 and mkt.std()>0 else np.nan
    return dict(n=n,net=net.mean(),t=net.mean()/(net.std()/np.sqrt(n)) if net.std()>0 else 0,
                sharpe=net.mean()/net.std()*np.sqrt(CYCLES_PER_YEAR) if net.std()>0 else 0,
                gross=res["gross"].mean()*1e4,long_alpha=res["long_alpha"].mean()*1e4,beta=beta)

def main():
    t0=time.time()
    print("=== iter-035: Passive hedge vs previous; A2(alt-basket) vs A4(real BTC) ===\n",flush=True)
    preds=pd.read_parquet(PREDS); preds["open_time"]=pd.to_datetime(preds["open_time"],utc=True)
    btc=build_btc_fwd4h(); btc_map=btc.to_dict()
    btc_times=set(btc.index)
    # restrict ALL variants to cycles with BTC data (apples-to-apples)
    preds=preds[preds["open_time"].isin(btc_times)].copy()
    print(f"  restricted to {preds.open_time.nunique()} cycles with real BTC data")
    print(f"  (A2=alt-basket short, A4=real BTC short; both vs A0=model L/S)\n")

    labels={"A0":"A0 model longs+shorts (PREV)","A2":"A2 longs + alt-basket short","A4":"A4 longs + real BTC short"}
    for slabel,s,e in SLICES:
        sub=preds[(preds.open_time>=s)&(preds.open_time<e)].copy()
        nc=sub.open_time.nunique()
        print(f"=== {slabel} ({nc} cycles w/ BTC) ===")
        print(f"  {'variant':<30}{'net bps':>9}{'t':>6}{'Sharpe':>8}{'gross':>8}{'L_alpha':>9}{'beta':>7}")
        print("  "+"-"*76)
        base=None
        for v in ["A0","A2","A4"]:
            st=stat(run(sub,v,btc_map))
            if st is None: continue
            if v=="A0": base=st
            tag="★" if abs(st["t"])>1.96 else " "
            dvs = f"  Δnet={st['net']-base['net']:+.1f}" if (base and v!="A0") else ""
            print(f"  {labels[v]:<30}{st['net']:>+8.2f}{tag}{st['t']:>+5.1f}{st['sharpe']:>+7.2f}"
                  f"{st['gross']:>+7.1f}{st['long_alpha']:>+8.1f}{st['beta']:>+6.2f}{dvs}")
        print()
    print(f"DONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
