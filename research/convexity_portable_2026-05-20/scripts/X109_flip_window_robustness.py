"""X109 — Is flip_t0's 'positive every year' robust to the ic-window choice?

User point: flip_t0 (flip sym if trailing-IC<0) is positive EVERY year while base blows
up in 2026 → as a robustness-first rule it may make most sense. But flip_t0 used W=90
(chosen). Test whether 'positive every year' + the 2026 rescue survive across windows
W in {45,60,90,120,180}. If robust across windows → committing to the flip is defensible
(structural, not a W-cherry-pick). If only W=90 works → fragile.

Full hybrid (mom-bull / mean-rev-side BN / flat-bear, K=5), tau=0 (flip whenever ic_W<0).
Report per-year Sharpe + full + worst-year + maxDD for base and each W.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
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
    print("=== X109 flip window robustness (tau=0) ===\n", flush=True)
    d=pd.read_parquet(RC/"x70_v0_3yr_preds.parquet", columns=["symbol","open_time","pred","alpha_A","return_pct"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy().sort_values(["symbol","open_time"])
    WINDOWS=[45,60,90,120,180]
    for W in WINDOWS:
        d[f"ic{W}"]=d.groupby("symbol",group_keys=False).apply(
            lambda g: g["pred"].rolling(W,min_periods=max(20,W//2)).corr(g["alpha_A"]).shift(HOLD))
    print("mom + beta...", flush=True)
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    syms=sorted(d["symbol"].unique()); mom_rows=[]; beta_map={}
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,"mom30":(c4/c4.shift(180)-1).shift(1).values}))
        r=np.log(c4/c4.shift(1)); ri,bi=r.align(br,join="inner")
        beta_map[sym]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    betas=pd.concat([s.rename(k) for k,s in beta_map.items()],axis=1)
    d=d.merge(mom,on=["symbol","open_time"],how="left")
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left").dropna(subset=["btc_ret_30d"])
    d["regime"]=np.where(d["btc_ret_30d"]>0.10,"bull",np.where(d["btc_ret_30d"]<-0.10,"bear","side"))
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}

    def build(iccol):
        ws=[]
        for ot in times:
            g=by_t[ot]; rg=g["regime"].iloc[0]
            if rg=="bear": ws.append({}); continue
            key="mom30" if rg=="bull" else "pred"
            gg=g.dropna(subset=[key]).copy()
            if rg=="side" and iccol is not None:
                gg["score"]=gg["pred"]*np.where(gg[iccol].fillna(0.0)<0,-1.0,1.0); sc="score"
            else: sc=key
            if len(gg)<2*K: ws.append({}); continue
            gg=gg.sort_values(sc); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
            a=b=1.0
            if rg=="side":
                brow=betas.loc[ot] if ot in betas.index else None
                if brow is not None:
                    mbL=np.nanmean([brow.get(s,np.nan) for s in L]); mbS=np.nanmean([brow.get(s,np.nan) for s in S])
                    if np.isfinite(mbL) and np.isfinite(mbS) and mbL>0 and mbS>0: a=2*mbS/(mbL+mbS); b=2*mbL/(mbL+mbS)
            w={}
            for s in L: w[s]=w.get(s,0)+a/K
            for s in S: w[s]=w.get(s,0)-b/K
            ws.append(w)
        prev={}; pnl=[]
        for t in range(len(ws)):
            active=ws[max(0,t-HOLD+1):t+1]; net={}
            for w in active:
                for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
            alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
            rl=by_t[times[t]]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
            pnl.append(sum(net.get(s,0)*rmap.get(s,0.0) for s in net if np.isfinite(rmap.get(s,0.0)))-turn*0.5*COST); prev=net
        return pd.Series(pnl,index=pd.to_datetime(times))

    def row(name,p):
        pb=p*1e4; eq=pb.cumsum(); dd=(eq-eq.cummax()).min()
        yr={y:ann(g/1e4) for y,g in pb.groupby(pb.index.year)}
        worst=min(yr.values())
        s=" ".join(f"{y}:{yr.get(y,float('nan')):+.2f}" for y in [2023,2024,2025,2026])
        print(f"  {name:<14}{ann(p):>+7.2f}{worst:>+8.2f}{dd:>+9.0f}   {s}", flush=True)

    print(f"  {'variant':<14}{'full':>7}{'worstYr':>8}{'maxDD':>9}   per-year Sharpe")
    row("base", build(None))
    for W in WINDOWS: row(f"flip ic{W}", build(f"ic{W}"))
    print(f"\nROBUST iff flip positive EVERY year across most W AND 2026 rescued across W. Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
