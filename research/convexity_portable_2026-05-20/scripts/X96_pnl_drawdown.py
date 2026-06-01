"""X96 — PnL equity curve + drawdown for the optimized config.

Optimized = held-book hybrid: momentum bull / mean-rev sideways (beta-neutral leg) /
FLAT bear, all 44 syms, K=3, 24h hold (6 overlap), cost 2.25bps/unit-delta.
Computes: per-cycle net PnL series, cumulative equity (bps), max drawdown, Calmar,
by-year stats. Saves an equity+drawdown chart.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/yuqing/ctaNew")
RCACHE = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"research/convexity_portable_2026-05-20/results"
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


def main():
    t0=time.time()
    print("=== X96 PnL + drawdown (optimized hybrid) ===\n", flush=True)
    apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
    apd=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)]
    syms=sorted(apd["symbol"].unique())
    print("mom + beta...", flush=True)
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    mom_rows=[]; beta_map={}
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,"mom30":(c4/c4.shift(180)-1).shift(1).values}))
        r=np.log(c4/c4.shift(1)); ri,bi=r.align(br,join="inner")
        beta_map[sym]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    betas=pd.concat([s.rename(k) for k,s in beta_map.items()],axis=1)
    apd=apd.merge(mom,on=["symbol","open_time"],how="left")
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index()
    btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    apd=apd.merge(btc30,on="open_time",how="left").dropna(subset=["btc_ret_30d"])
    apd["regime"]=np.where(apd["btc_ret_30d"]>0.10,"bull",np.where(apd["btc_ret_30d"]<-0.10,"bear","side"))

    times=sorted(apd["open_time"].unique()); by_t={ot:g for ot,g in apd.groupby("open_time")}
    cyc_w=[]; ret_seq=[]
    for ot in times:
        g=by_t[ot]; rg=g["regime"].iloc[0]; ret_seq.append(dict(zip(g["symbol"],g["return_pct"])))
        if rg=="bear": cyc_w.append({}); continue
        key="mom30" if rg=="bull" else "pred"
        gg=g.dropna(subset=[key])
        if len(gg)<2*K: cyc_w.append({}); continue
        gg=gg.sort_values(key); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
        # beta-neutral leg sizing in sideways only
        a=b=1.0
        if rg=="side":
            brow=betas.loc[ot] if ot in betas.index else None
            if brow is not None:
                mbL=np.nanmean([brow.get(s,np.nan) for s in L]); mbS=np.nanmean([brow.get(s,np.nan) for s in S])
                if np.isfinite(mbL) and np.isfinite(mbS) and mbL>0 and mbS>0:
                    a=2*mbS/(mbL+mbS); b=2*mbL/(mbL+mbS)
        w={}
        for s in L: w[s]=w.get(s,0)+a/K
        for s in S: w[s]=w.get(s,0)-b/K
        cyc_w.append(w)

    prev={}; pnl=[]
    for t in range(len(cyc_w)):
        active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
        rl=ret_seq[t]; pnl.append(sum(net.get(s,0)*rl.get(s,0.0) for s in net) - turn*0.5*COST); prev=net
    pnl=pd.Series(pnl, index=pd.to_datetime(times)); pnl_bps=pnl*1e4
    eq=pnl_bps.cumsum()  # cumulative bps (additive — bps on constant gross notional)
    # drawdown on cumulative bps
    peak=eq.cummax(); dd=eq-peak

    sharpe=ann(pnl)
    total=eq.iloc[-1]
    maxdd=dd.min()
    # annualized return in bps (mean per cycle × 6×365)
    ann_ret_bps=pnl_bps.mean()*6*365
    calmar=ann_ret_bps/abs(maxdd) if maxdd!=0 else np.nan
    days=(pnl.index[-1]-pnl.index[0]).days

    print(f"\n=== Optimized hybrid PnL/DD ({days} days, {len(pnl)} cycles) ===")
    print(f"  Sharpe (ann)        : {sharpe:+.2f}")
    print(f"  Total PnL           : {total:+.0f} bps  (= {total/100:+.1f}% on gross notional, additive)")
    print(f"  Ann return          : {ann_ret_bps:+.0f} bps/yr (~{ann_ret_bps/100:+.1f}%/yr)")
    print(f"  Max drawdown        : {maxdd:+.0f} bps (~{maxdd/100:+.1f}%)")
    print(f"  Calmar (annret/maxDD): {calmar:+.2f}")
    print(f"  worst cycle         : {pnl_bps.min():+.1f} bps   best cycle: {pnl_bps.max():+.1f} bps")
    print(f"  % positive cycles   : {(pnl_bps>0).mean()*100:.1f}%")
    # drawdown duration
    in_dd = dd<0;
    print(f"\n  By year:")
    for yr,g in pnl_bps.groupby(pnl_bps.index.year):
        e=g.cumsum(); p=e.cummax(); d=(e-p).min()
        print(f"    {yr}: PnL {g.sum():+.0f}bps  Sharpe {ann(g/1e4):+.2f}  maxDD(in-yr) {d:+.0f}bps", flush=True)

    # chart
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    ax1.plot(eq.index, eq.values, color="navy", lw=1.3)
    ax1.set_title(f"Optimized hybrid — cumulative PnL (bps)  |  Sharpe {sharpe:+.2f}, maxDD {maxdd:+.0f}bps, total {total:+.0f}bps")
    ax1.set_ylabel("cum PnL (bps)"); ax1.grid(alpha=0.3); ax1.axhline(0,color="grey",lw=0.6)
    ax2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.5)
    ax2.set_ylabel("drawdown (bps)"); ax2.grid(alpha=0.3); ax2.set_xlabel("date")
    fig.tight_layout()
    out_png=OUT/"X96_optimized_pnl_drawdown.png"; fig.savefig(out_png, dpi=110)
    print(f"\nSaved chart → {out_png}")
    pnl_bps.to_frame("pnl_bps").to_parquet(OUT/"X96_pnl_series.parquet")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
