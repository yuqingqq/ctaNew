"""X117 — HL70 base: equity curve + drawdown (#1) and cost sensitivity (#3).

HL70 (x64 V5mv3 preds) with the regime hybrid (mom-bull / mean-rev-side BN / flat-bear, K=5)
earns +1.93 (X116). Now: (1) equity/DD curve + by-period stats + chart; (2) cost sweep
{0,1,2,3,4.5,6,9} bps/leg — show performance at HL maker (~1bp) / taker (~3bp) vs the
conservative 4.5bp baseline.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"research/convexity_portable_2026-05-20/results"
PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"
K=5; HOLD=6; Y26=pd.Timestamp("2026-01-01",tz="UTC")


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
    print("=== X117 HL70 PnL/DD + cost sensitivity ===\n", flush=True)
    d=pd.read_parquet(PREDS, columns=["symbol","open_time","pred","return_pct"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy()
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
    btc30=(b4/b4.shift(180)-1).to_frame("b30").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left").dropna(subset=["b30"])
    d["regime"]=np.where(d["b30"]>0.10,"bull",np.where(d["b30"]<-0.10,"bear","side"))
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}

    # build weights once (cost-independent), then apply cost in held-book
    cyc_w=[]
    for ot in times:
        g=by_t[ot]; rg=g["regime"].iloc[0]
        if rg=="bear": cyc_w.append({}); continue
        key="mom30" if rg=="bull" else "pred"; gg=g.dropna(subset=[key])
        if len(gg)<2*K: cyc_w.append({}); continue
        gg=gg.sort_values(key); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
        a=b=1.0
        if rg=="side":
            brow=betas.loc[ot] if ot in betas.index else None
            if brow is not None:
                mbL=np.nanmean([brow.get(s,np.nan) for s in L]); mbS=np.nanmean([brow.get(s,np.nan) for s in S])
                if np.isfinite(mbL) and np.isfinite(mbS) and mbL>0 and mbS>0: a=2*mbS/(mbL+mbS); b=2*mbL/(mbL+mbS)
        w={}
        for s in L: w[s]=w.get(s,0)+a/K
        for s in S: w[s]=w.get(s,0)-b/K
        cyc_w.append(w)

    def heldbook(cost):
        prev={}; pnl=[]
        for t in range(len(cyc_w)):
            active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
            for w in active:
                for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
            alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
            rl=by_t[times[t]]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
            pnl.append(sum(net.get(s,0)*rmap.get(s,0.0) for s in net if np.isfinite(rmap.get(s,0.0)))-turn*0.5*cost); prev=net
        return pd.Series(pnl,index=pd.to_datetime(times))

    # #1 equity/DD at 4.5bps
    p=heldbook(4.5e-4); pb=p*1e4; eq=pb.cumsum(); peak=eq.cummax(); dd=eq-peak
    print(f"=== #1 HL70 base PnL/DD (cost 4.5bps, {len(p)} cycles, {(p.index[-1]-p.index[0]).days}d) ===")
    print(f"  Sharpe {ann(p):+.2f}  totPnL {eq.iloc[-1]:+.0f}bps  maxDD {dd.min():+.0f}bps  Calmar {pb.mean()*6*365/abs(dd.min()):+.2f}")
    print(f"  %pos cycles {(pb>0).mean()*100:.1f}  worst {pb.min():+.0f}  best {pb.max():+.0f}")
    for yr,g in pb.groupby(pb.index.year):
        e=g.cumsum(); print(f"    {yr}: PnL {g.sum():+.0f}bps Sharpe {ann(g/1e4):+.2f} maxDD {(e-e.cummax()).min():+.0f}bps", flush=True)

    fig,(a1,a2)=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    a1.plot(eq.index,eq.values,color="navy",lw=1.3); a1.axhline(0,color="grey",lw=0.6); a1.grid(alpha=0.3)
    a1.set_title(f"HL70 base (regime hybrid, K=5, 4.5bps) — Sharpe {ann(p):+.2f}, maxDD {dd.min():+.0f}bps, total {eq.iloc[-1]:+.0f}bps")
    a1.set_ylabel("cum PnL (bps)")
    a2.fill_between(dd.index,dd.values,0,color="crimson",alpha=0.5); a2.grid(alpha=0.3); a2.set_ylabel("drawdown (bps)")
    fig.tight_layout(); png=OUT/"X117_hl70_pnl_dd.png"; fig.savefig(png,dpi=110)
    print(f"  chart → {png}")

    # #3 cost sweep
    print(f"\n=== #3 HL70 cost sensitivity (bps/leg) ===")
    print(f"  {'cost':>6}{'Sharpe':>8}{'totPnL':>9}{'maxDD':>9}   note")
    notes={0:"frictionless",1:"HL maker~1",3:"HL taker~3",4.5:"baseline(conserv)"}
    for c in [0,1,2,3,4.5,6,9]:
        pp=heldbook(c*1e-4); ppb=pp*1e4; e=ppb.cumsum()
        print(f"  {c:>5.1f}{ann(pp):>+8.2f}{e.iloc[-1]:>+9.0f}{(e-e.cummax()).min():>+9.0f}   {notes.get(c,'')}", flush=True)
    print(f"\nDone [{time.time()-t0:.0f}s]")


if __name__=="__main__":
    main()
