"""X93 — Block-bootstrap CI on the OPTIMIZED config (capstone validation).

Optimized = simple held-book HYBRID: momentum(mom_30d) bull / mean-rev(V0) sideways /
FLAT bear, all 44 syms, K=3, 24h hold (6 overlap), cost 2.25 bps/unit-delta.
This is a clean OOS number (walk-forward preds + PIT rules, no in-sample fit) = +1.89 (3yr).
Question: is +1.89 ROBUST (CI above 0, high P(>0)) or fragile?

Method: block-bootstrap by FOLD (resample the 9 walk-forward folds with replacement,
50 iters), recompute held-book hybrid net Sharpe each. Report mean, std, 95% CI, P(>0).
Folds are the natural independent blocks (cycles within a fold are 24h-overlap-dependent).
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


def heldbook(cyc_w, ret_seq):
    n=len(cyc_w); prev={}; rets=[]
    for t in range(n):
        active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
        rl=ret_seq[t]
        rets.append(sum(net.get(s,0)*rl.get(s,0.0) for s in net) - turn*0.5*COST); prev=net
    return np.array(rets)


def main():
    t0=time.time()
    print("=== X93 block-bootstrap CI on optimized held-book hybrid ===\n", flush=True)
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

    # build per-cycle weights (hybrid flat-bear) + per-cycle return dict + fold, ordered by time
    times=sorted(apd["open_time"].unique())
    by_t={ot:g for ot,g in apd.groupby("open_time")}
    cyc_w=[]; ret_seq=[]; cyc_fold=[]
    for ot in times:
        g=by_t[ot]; rg=g["regime"].iloc[0]
        ret_seq.append(dict(zip(g["symbol"],g["return_pct"])))
        cyc_fold.append(int(g["fold"].iloc[0]))
        if rg=="bear": cyc_w.append({}); continue
        key="mom30" if rg=="bull" else "pred"
        gg=g.dropna(subset=[key])
        if len(gg)<2*K: cyc_w.append({}); continue
        gg=gg.sort_values(key); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
        w={}
        for s in L: w[s]=w.get(s,0)+1.0/K
        for s in S: w[s]=w.get(s,0)-1.0/K
        cyc_w.append(w)

    base=heldbook(cyc_w, ret_seq)
    print(f"\nOptimized hybrid (flat-bear) point Sharpe: {ann(base):+.2f}  ({len(base)} cycles)")

    # block bootstrap by fold
    folds=sorted(set(cyc_fold)); cyc_fold=np.array(cyc_fold)
    idx_by_fold={f:np.where(cyc_fold==f)[0] for f in folds}
    np.random.seed(20260522); shs=[]
    for i in range(50):
        sample_folds=np.random.choice(folds,size=len(folds),replace=True)
        idxs=np.concatenate([idx_by_fold[f] for f in sample_folds])
        # rebuild contiguous cyc_w/ret_seq in sampled order
        bw=[cyc_w[j] for j in idxs]; br=[ret_seq[j] for j in idxs]
        r=heldbook(bw, br); shs.append(ann(r))
        if (i+1)%10==0: print(f"  iter {i+1}/50: mean={np.nanmean(shs):+.2f}", flush=True)
    a=np.array(shs)
    print(f"\n=== Block-bootstrap (n=50, by fold) ===")
    print(f"  mean ± std : {np.nanmean(a):+.2f} ± {np.nanstd(a):.2f}")
    print(f"  median     : {np.nanmedian(a):+.2f}")
    print(f"  95% CI     : [{np.nanpercentile(a,2.5):+.2f}, {np.nanpercentile(a,97.5):+.2f}]")
    print(f"  P(>0)      : {(a>0).mean()*100:.0f}%   P(>1): {(a>1).mean()*100:.0f}%")
    print(f"\nDone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
