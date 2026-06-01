"""X98 — Where does the drawdown actually live? Localize DD by YEAR and REGIME.

User hypothesis: the -56% maxDD (and weak 2026 YTD) is a MARKET-REGIME problem
concentrated in 2023 and 2026 — not a sizing problem (confirmed: invvol/vol-target
in X97 didn't help, so DD is broad/correlated, not single-name).

This recomputes the optimized hybrid (momentum bull / mean-rev sideways BN / flat bear,
held-book K=5 — the X97 DD winner) and reports:
  - cumulative equity + drawdown
  - per-YEAR: PnL, Sharpe, in-year maxDD, regime mix (bull/side/bear cycle share)
  - per-(YEAR×REGIME): PnL, Sharpe — which (year,regime) cells bleed
  - the worst peak-to-trough DD episodes with date range + dominant regime
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RCACHE = REPO/"research/convexity_portable_2026-05-20/results/_cache"
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
    x=pd.Series(x).dropna()
    return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan


def main():
    t0=time.time()
    print("=== X98 regime/year drawdown anatomy (optimized hybrid, K=5) ===\n", flush=True)
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
    cyc_w=[]; ret_seq=[]; regime_seq=[]
    for ot in times:
        g=by_t[ot]; rg=g["regime"].iloc[0]; regime_seq.append(rg)
        ret_seq.append(dict(zip(g["symbol"],g["return_pct"])))
        if rg=="bear": cyc_w.append({}); continue
        key="mom30" if rg=="bull" else "pred"
        gg=g.dropna(subset=[key])
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

    prev={}; pnl=[]
    for t in range(len(cyc_w)):
        active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
        rl=ret_seq[t]; pnl.append(sum(net.get(s,0)*rl.get(s,0.0) for s in net) - turn*0.5*COST); prev=net
    idx=pd.to_datetime(times)
    pnl=pd.Series(pnl, index=idx); pb=pnl*1e4
    reg=pd.Series(regime_seq, index=idx)
    eq=pb.cumsum(); peak=eq.cummax(); dd=eq-peak

    print(f"Whole sample: Sharpe {ann(pnl):+.2f}  totPnL {eq.iloc[-1]:+.0f}bps  maxDD {dd.min():+.0f}bps  {len(pnl)} cycles\n")

    # per-year
    print(f"=== Per YEAR ===")
    print(f"  {'yr':<6}{'cyc':>6}{'PnL':>9}{'Sharpe':>8}{'inYrDD':>9}{'bull%':>7}{'side%':>7}{'bear%':>7}")
    for yr,g in pb.groupby(pb.index.year):
        rg=reg.loc[g.index]; e=g.cumsum(); d=(e-e.cummax()).min()
        n=len(g); bl=(rg=="bull").mean()*100; sd=(rg=="side").mean()*100; be=(rg=="bear").mean()*100
        print(f"  {yr:<6}{n:>6}{g.sum():>+9.0f}{ann(g/1e4):>+8.2f}{d:>+9.0f}{bl:>7.1f}{sd:>7.1f}{be:>7.1f}", flush=True)

    # per year x regime PnL
    print(f"\n=== PnL by YEAR x REGIME (bps; bear=flat→0) ===")
    print(f"  {'yr':<6}{'bull':>10}{'side':>10}{'bear':>10}")
    for yr,g in pb.groupby(pb.index.year):
        rg=reg.loc[g.index]
        bl=g[rg=="bull"].sum(); sd=g[rg=="side"].sum(); be=g[rg=="bear"].sum()
        print(f"  {yr:<6}{bl:>+10.0f}{sd:>+10.0f}{be:>+10.0f}", flush=True)

    # per year x regime Sharpe
    print(f"\n=== Sharpe by YEAR x REGIME ===")
    print(f"  {'yr':<6}{'bull':>10}{'side':>10}")
    for yr,g in pb.groupby(pb.index.year):
        rg=reg.loc[g.index]
        sb=ann(g[rg=='bull']/1e4); ss=ann(g[rg=='side']/1e4)
        print(f"  {yr:<6}{sb:>+10.2f}{ss:>+10.2f}", flush=True)

    # worst peak-to-trough DD episodes
    print(f"\n=== Worst drawdown episodes (peak→trough) ===")
    # find local DD troughs: walk dd, group contiguous underwater stretches
    underwater = dd < -1e-9
    episodes=[]
    i=0; n=len(dd)
    while i<n:
        if underwater.iloc[i]:
            j=i
            while j<n and underwater.iloc[j]: j+=1
            seg=dd.iloc[i:j]
            trough_pos=seg.idxmin(); trough=seg.min()
            # peak date = last time eq made a high before i
            peak_date=eq.iloc[:i].idxmax() if i>0 else eq.index[0]
            seg_reg=reg.loc[i if isinstance(i,pd.Timestamp) else dd.index[i]:dd.index[j-1]]
            rmix=reg.loc[dd.index[i]:dd.index[j-1]].value_counts(normalize=True)
            dom=rmix.idxmax() if len(rmix) else "?"
            episodes.append((trough, peak_date, trough_pos, dd.index[j-1], (dd.index[j-1]-peak_date).days, dom, rmix.to_dict()))
            i=j
        else: i+=1
    episodes.sort(key=lambda e:e[0])
    print(f"  {'trough(bps)':>12}  {'peak_date':<12}{'trough_date':<12}{'recov_date':<12}{'days':>5}  dom_regime  mix")
    for tr,pk,trd,rec,dys,dom,mix in episodes[:8]:
        mixs=" ".join(f"{k}:{v*100:.0f}%" for k,v in sorted(mix.items()))
        print(f"  {tr:>+12.0f}  {str(pk.date()):<12}{str(trd.date()):<12}{str(rec.date()):<12}{dys:>5}  {dom:<11}{mixs}", flush=True)

    print(f"\nDone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
