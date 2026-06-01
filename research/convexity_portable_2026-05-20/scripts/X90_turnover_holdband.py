"""X90 — Turnover reduction via HOLD-BAND on the mean-rev (sideways) leg.

Root-cause finding: turnover is the binding constraint (V5 lost to V0 purely on +23%
turnover; sideways mean-rev is the workhorse but cost-marginal). Lever: a parameter-
light HOLD-BAND — keep a current basket member as long as it stays within the top-N
(N = band×K) ranked names; only replace when it falls out of the band. Reduces churn
without a tuned pred-margin.

Base config (corrected best): momentum bull / mean-rev sideways / FLAT bear.
Sweep band ∈ {1.0 (no band, baseline), 1.5, 2.0, 3.0} on the mean-rev leg.
Held-book (24h, continuous), net of cost. 3yr + 12mo. Report Sharpe + turnover.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RCACHE = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
COST=4.5e-4; K=3; HOLD=6


def load_close(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def ann(x):
    x=pd.Series(x).dropna()
    return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan


def select_band(ranked, k, band, cur):
    """ranked: symbols sorted best→worst for this side. Keep cur members still in
    top-(band*k); fill remaining slots from the top. Returns new member list."""
    N=int(round(band*k))
    topN=set(ranked[:N])
    keep=[s for s in cur if s in topN][:k]
    new=list(keep)
    for s in ranked:
        if len(new)>=k: break
        if s not in new: new.append(s)
    return new[:k]


def heldbook(cyc_w, cyc_times, ret_lookup):
    n=len(cyc_w); prev={}; rets=[]; turns=[]
    for t in range(n):
        active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
        turns.append(turn)
        rl=ret_lookup.get(cyc_times[t],{})
        pnl=sum(net.get(s,0)*rl.get(s,0.0) for s in net)
        rets.append(pnl-turn*0.5*COST); prev=net
    return pd.Series(rets,index=cyc_times), np.mean(turns)


def main():
    t0=time.time()
    print("=== X90 turnover hold-band (mean-rev leg) ===\n", flush=True)
    apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
    apd=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)]
    syms=sorted(apd["symbol"].unique())
    print("mom_30d...", flush=True)
    mr=[]
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]; mom=(c4/c4.shift(180)-1).shift(1)
        mr.append(pd.DataFrame({"symbol":sym,"open_time":mom.index,"mom30":mom.values}))
    mom=pd.concat(mr,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    apd=apd.merge(mom,on=["symbol","open_time"],how="left")
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index()
    btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    apd=apd.merge(btc30,on="open_time",how="left").dropna(subset=["btc_ret_30d"])
    apd["regime"]=np.where(apd["btc_ret_30d"]>0.10,"bull",np.where(apd["btc_ret_30d"]<-0.10,"bear","side"))
    ret_lookup={ot:dict(zip(g["symbol"],g["return_pct"])) for ot,g in apd.groupby("open_time")}

    def build(sub, band):
        """momentum bull / mean-rev sideways(with hold-band) / FLAT bear."""
        times=sorted(sub["open_time"].unique()); ws=[]
        cur_l, cur_s = [], []
        for ot in times:
            g=sub[sub["open_time"]==ot]; rg=g["regime"].iloc[0]
            if rg=="bear": ws.append({}); cur_l, cur_s = [], []; continue
            key = "mom30" if rg=="bull" else "pred"
            gg=g.dropna(subset=[key])
            if len(gg)<2*K: ws.append({}); continue
            long_rank=gg.sort_values(key,ascending=False)["symbol"].tolist()   # best→worst long
            short_rank=gg.sort_values(key,ascending=True)["symbol"].tolist()   # best→worst short
            if rg=="side" and band>1.0:
                L=select_band(long_rank, K, band, cur_l); S=select_band(short_rank, K, band, cur_s)
            else:
                L=long_rank[:K]; S=short_rank[:K]
            cur_l, cur_s = L, S
            w={}
            for s in L: w[s]=w.get(s,0)+1.0/K
            for s in S: w[s]=w.get(s,0)-1.0/K
            ws.append(w)
        return times, ws

    for tag, sub in [("3yr", apd), ("12mo", apd[apd["open_time"]>=pd.Timestamp("2025-05-01",tz="UTC")])]:
        print(f"--- {tag} (flat-bear base; hold-band on sideways mean-rev leg) ---")
        print(f"  {'band':<8}{'Sharpe':>9}{'PnL_bps':>10}{'turnover':>10}")
        for band in [1.0, 1.5, 2.0, 3.0]:
            times, ws = build(sub, band)
            r, turn = heldbook(ws, times, ret_lookup)
            print(f"  {band:<8.1f}{ann(r):>+9.2f}{r.sum()*1e4:>+10.0f}{turn:>10.3f}", flush=True)

    print(f"\nVERDICT: if higher band → higher Sharpe with lower turnover, the hold-band")
    print(f"harvests the cost-marginal sideways edge. Find the turnover/edge sweet spot.")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
