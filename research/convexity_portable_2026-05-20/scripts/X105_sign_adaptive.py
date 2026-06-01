"""X105 — Sign-adaptive / simple-reversal test: can we recover 2026 where frozen ML can't?

X103 finding: features still carry per-symbol |IC| recently but the SIGN inverted in 2026,
and the frozen ML model's learned signs are stale. Two ways to react to a sign-flip
(retraining on more history does NOT help — it dilutes, shrinking pred dispersion 0.26->0.09):
  (b) SIGN-ADAPTIVE: flip the signal's sign based on its own trailing realized cross-sec IC.
  (c) SIMPLE STATIC REVERSAL: rank by recent return, long losers / short winners.

Test on 44-sym 3yr panel + V0 preds. Per-cycle cross-sec IC + held-book (K=5, 24h, cost
4.5bps). Per YEAR (isolate 2026). PIT: sign-adaptive uses trailing IC over cycles <= t-HOLD.
Honest caveat: window W is a hyperparam; DDI says per-cycle IC ~unpredictable (R2=0.005) so
trailing-IC sign may lag. Report a couple W; flag if 2026 only rescued by hindsight.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
PANEL = REPO/"outputs/vBTC_features/panel_3yr_v5.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"
COST=4.5e-4; K=5; HOLD=6


def load_close(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def ann(x):
    x=pd.Series(x).dropna(); return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan


def main():
    t0=time.time()
    print("=== X105 sign-adaptive / reversal test ===\n", flush=True)
    p=pd.read_parquet(RC/"x70_v0_3yr_preds.parquet", columns=["symbol","open_time","pred","alpha_A","return_pct"])
    p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
    p=p[(p["open_time"].dt.hour%4==0)&(p["open_time"].dt.minute==0)]
    feats=pd.read_parquet(PANEL, columns=["symbol","open_time","ret_3d","return_1d"])
    feats["open_time"]=pd.to_datetime(feats["open_time"],utc=True)
    d=p.merge(feats,on=["symbol","open_time"],how="left")
    # BTC regime for flat-bear
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left")
    d["regime"]=np.where(d["btc_ret_30d"]>0.10,"bull",np.where(d["btc_ret_30d"]<-0.10,"bear","side"))
    print(f"{d['symbol'].nunique()} syms, {d['open_time'].min().date()}->{d['open_time'].max().date()}, {len(d):,} rows(4h)\n", flush=True)

    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}

    # precompute per-cycle realized cross-sec IC of pred (for sign-adaptive), PIT
    pred_ic=[]
    for ot in times:
        g=by_t[ot].dropna(subset=["pred","alpha_A"])
        pred_ic.append(spearmanr(g["pred"],g["alpha_A"]).correlation if len(g)>=8 else np.nan)
    pred_ic=np.array(pred_ic)

    def heldbook(weights):
        prev={}; pnl=[]
        for t in range(len(weights)):
            active=weights[max(0,t-HOLD+1):t+1]; net={}
            for w in active:
                for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
            alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
            rl=by_t[times[t]]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
            pnl.append(sum(net.get(s,0)*rmap.get(s,0.0) for s in net if np.isfinite(rmap.get(s,0.0)))-turn*0.5*COST); prev=net
        return pd.Series(pnl, index=pd.to_datetime(times))

    def build(score_col, long_high=True, flat_bear=True, sign_adapt=False, W=30):
        """long_high: long top-score/short bottom. sign_adapt: flip selection by trailing pred-IC sign."""
        ws=[]
        for ti,ot in enumerate(times):
            g=by_t[ot]
            if flat_bear and g["regime"].iloc[0]=="bear": ws.append({}); continue
            gg=g.dropna(subset=[score_col])
            if len(gg)<2*K: ws.append({}); continue
            sign=1.0
            if sign_adapt:
                hi=ti-HOLD
                if hi>=W:
                    v=pred_ic[:hi+1]; v=v[~np.isnan(v)]
                    if len(v)>=W: sign=1.0 if np.mean(v[-W:])>=0 else -1.0
            asc = (sign>0) == long_high  # determines which end is long
            gg=gg.sort_values(score_col)
            top=gg.tail(K)["symbol"].tolist(); bot=gg.head(K)["symbol"].tolist()
            L,S=(top,bot) if asc else (bot,top)
            w={}
            for s in L: w[s]=w.get(s,0)+1.0/K
            for s in S: w[s]=w.get(s,0)-1.0/K
            ws.append(w)
        return ws

    def report(name, ws):
        p=heldbook(ws); pb=p*1e4; eq=pb.cumsum(); dd=(eq-eq.cummax()).min()
        yr={y:ann(g/1e4) for y,g in pb.groupby(pb.index.year)}
        s=" ".join(f"{y}:{yr.get(y,float('nan')):+.2f}" for y in [2023,2024,2025,2026])
        print(f"  {name:<34}{ann(p):>+7.2f}{eq.iloc[-1]:>+9.0f}{dd:>+9.0f}   {s}", flush=True)

    print(f"  {'signal (held-book, flat-bear)':<34}{'Sharpe':>7}{'totPnL':>9}{'maxDD':>9}   per-year Sharpe")
    report("V0 pred (long top)", build("pred", long_high=True))
    report("ret_3d REVERSAL (long bottom)", build("ret_3d", long_high=False))
    report("return_1d REVERSAL (long bottom)", build("return_1d", long_high=False))
    print("  -- sign-adaptive V0 pred (flip by trailing pred-IC) --")
    for W in [30,60,90]:
        report(f"sign-adapt pred W={W}", build("pred", long_high=True, sign_adapt=True, W=W))

    print(f"\nGOAL: any signal with POSITIVE 2026 Sharpe = recoverable recent edge. Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
