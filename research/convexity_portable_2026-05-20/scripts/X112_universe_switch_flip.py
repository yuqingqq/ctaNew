"""X112 — Does the 2026 sign-flip (and the ic120 flip-rescue) generalize across UNIVERSES?

User Q: switch the symbol set and check performance. The strategy is known to be
universe-overfit (memory: UNI stress test). Test base vs committed ic120 flip on different
symbol subsets of the 44: (a) 10 random 30-symbol draws, (b) disjoint halves A/B.
Report per-subset 2026 Sharpe for base vs flip — does base bleed in 2026 across universes,
and does the flip rescue it across universes? Robustness of the sign-flip finding.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
COST=4.5e-4; K=5; HOLD=6; W=120
Y26=pd.Timestamp("2026-01-01",tz="UTC")


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
    print("=== X112 universe-switch: base vs ic120 flip ===\n", flush=True)
    d=pd.read_parquet(RC/"x70_v0_3yr_preds.parquet", columns=["symbol","open_time","pred","alpha_A","return_pct"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy().sort_values(["symbol","open_time"])
    d["ic120"]=d.groupby("symbol",group_keys=False).apply(
        lambda g: g["pred"].rolling(W,min_periods=W//2).corr(g["alpha_A"]).shift(HOLD))
    print("mom+beta...", flush=True)
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    allsyms=sorted(d["symbol"].unique()); mom_rows=[]; beta_map={}
    for sym in allsyms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,"mom30":(c4/c4.shift(180)-1).shift(1).values}))
        r=np.log(c4/c4.shift(1)); ri,bi=r.align(br,join="inner")
        beta_map[sym]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    betas=pd.concat([s.rename(k) for k,s in beta_map.items()],axis=1)
    d=d.merge(mom,on=["symbol","open_time"],how="left")
    btc30=(b4/b4.shift(180)-1).to_frame("b30").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left").dropna(subset=["b30"])
    d["regime"]=np.where(d["b30"]>0.10,"bull",np.where(d["b30"]<-0.10,"bear","side"))
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}

    def run(subset, flip):
        sub=set(subset); ws=[]
        for ot in times:
            g=by_t[ot]; g=g[g["symbol"].isin(sub)]; rg=g["regime"].iloc[0] if len(g) else "side"
            if len(g)==0 or rg=="bear": ws.append({}); continue
            key="mom30" if rg=="bull" else "pred"
            gg=g.dropna(subset=[key]).copy()
            if rg=="side" and flip:
                gg["score"]=gg["pred"]*np.where(gg["ic120"].fillna(0.0)<0,-1.0,1.0); sc="score"
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
        p=pd.Series(pnl,index=pd.to_datetime(times)); p26=p[p.index>=Y26]
        return ann(p), ann(p26)

    rng=np.random.default_rng(7)
    print(f"  {'universe':<22}{'base full':>10}{'base 2026':>10}{'flip full':>10}{'flip 2026':>10}")
    bf,b26,ff,f26=run(allsyms,False)[0],run(allsyms,False)[1],run(allsyms,True)[0],run(allsyms,True)[1]
    print(f"  {'ALL-44':<22}{bf:>+10.2f}{b26:>+10.2f}{ff:>+10.2f}{f26:>+10.2f}", flush=True)
    # disjoint halves
    sh=list(allsyms); rng.shuffle(sh); HA, HB=sorted(sh[:22]), sorted(sh[22:])
    for nm,sub in [("halfA(22)",HA),("halfB(22)",HB)]:
        bfu,b2=run(sub,False); ffu,f2=run(sub,True)
        print(f"  {nm:<22}{bfu:>+10.2f}{b2:>+10.2f}{ffu:>+10.2f}{f2:>+10.2f}", flush=True)
    # 10 random 30-sym draws
    print("  -- 10 random 30-symbol subsets --")
    base26s=[]; flip26s=[]
    for i in range(10):
        sub=sorted(rng.choice(allsyms,30,replace=False))
        bfu,b2=run(sub,False); ffu,f2=run(sub,True)
        base26s.append(b2); flip26s.append(f2)
        print(f"  {'rand30_#'+str(i+1):<22}{bfu:>+10.2f}{b2:>+10.2f}{ffu:>+10.2f}{f2:>+10.2f}", flush=True)
    print(f"\n  random30 2026: base mean {np.mean(base26s):+.2f} (neg in {sum(x<0 for x in base26s)}/10) | "
          f"flip mean {np.mean(flip26s):+.2f} (pos in {sum(x>0 for x in flip26s)}/10)")
    print(f"  flip rescues 2026 (flip>base) in {sum(f>b for f,b in zip(flip26s,base26s))}/10 subsets. Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
