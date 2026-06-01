"""iter-002 — HL70-specific drawdown anatomy.

Reconstructs the HL70 regime-hybrid held book EXACTLY as X116/X117 (K=5, HOLD=6,
6 overlapping sleeves, mom-bull / mean-rev-side-BN / flat-bear, regime = BTC 30d ret
+/-10%, cost 4.5bps). Then, instead of just the book net PnL, it emits PER-CYCLE:

  - net long-leg PnL, net short-leg PnL (leg attribution)
  - gross long, gross short, net beta exposure
  - regime label
  - candidate-mechanism context series (all PIT, trailing):
      * BTC 30d ret (regime driver) and BTC trailing realized vol
      * cross-sectional return dispersion (std of cycle alpha_A across syms)
      * market-wide pairwise correlation of alt 4h returns (trailing 7d)  <- "alts move together"
      * breadth (frac of universe with positive 4h return this cycle)
      * funding extremes (mean |funding_z| across universe) -- if available
  - turnover

Then: locate the deepest drawdown episode on the cumulative equity, and compare the
distribution of each context variable INSIDE vs OUTSIDE the drawdown window. Also a
simple regression / quantile-bin of next-cycle book PnL on each context variable to
see which actually PRECEDES the bad cycles.

This is analysis only (no strategy change). Does not modify X116/X117 or cached preds.
"""
from __future__ import annotations
import time, json
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"research/convexity_portable_2026-05-20/results"
PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"
K=5; HOLD=6; COST=4.5e-4
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
    print("=== iter-002 HL70 drawdown anatomy ===\n", flush=True)
    cols=["symbol","open_time","pred","alpha_A","return_pct"]
    # funding optional
    avail = pd.read_parquet(PREDS, columns=None).columns.tolist() if False else None
    d=pd.read_parquet(PREDS, columns=cols)
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy().sort_values(["symbol","open_time"])

    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    btc_rvol_7d=br.rolling(42,min_periods=20).std().shift(1)  # PIT trailing 7d (42 4h-cycles)

    syms=sorted(d["symbol"].unique()); mom_rows=[]; beta_map={}; ret4_map={}
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,"mom30":(c4/c4.shift(180)-1).shift(1).values}))
        r=np.log(c4/c4.shift(1)); ri,bi=r.align(br,join="inner")
        beta_map[sym]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
        ret4_map[sym]=r  # 4h log return for correlation/breadth
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    betas=pd.concat([s.rename(k) for k,s in beta_map.items()],axis=1)
    ret4=pd.concat([s.rename(k) for k,s in ret4_map.items()],axis=1).sort_index()

    d=d.merge(mom,on=["symbol","open_time"],how="left")
    btc30=(b4/b4.shift(180)-1).to_frame("b30").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left").dropna(subset=["b30"])
    d["regime"]=np.where(d["b30"]>0.10,"bull",np.where(d["b30"]<-0.10,"bear","side"))
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}

    # ---- build per-cycle weights (identical to X117) ----
    cyc_w=[]; cyc_regime=[]
    for ot in times:
        g=by_t[ot]; rg=g["regime"].iloc[0]; cyc_regime.append(rg)
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

    # ---- held-book with leg attribution ----
    prev={}; rows=[]
    for t in range(len(cyc_w)):
        ot=times[t]
        active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
        rl=by_t[ot]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
        long_pnl=sum(net[s]*rmap.get(s,0.0) for s in net if net[s]>0 and np.isfinite(rmap.get(s,0.0)))
        short_pnl=sum(net[s]*rmap.get(s,0.0) for s in net if net[s]<0 and np.isfinite(rmap.get(s,0.0)))
        gross_l=sum(v for v in net.values() if v>0); gross_s=-sum(v for v in net.values() if v<0)
        # net beta exposure of held book
        brow=betas.loc[ot] if ot in betas.index else None
        netbeta=np.nan
        if brow is not None:
            netbeta=sum(net[s]*brow.get(s,np.nan) for s in net if np.isfinite(brow.get(s,np.nan)))
        cost=turn*0.5*COST
        pnl=long_pnl+short_pnl-cost
        rows.append(dict(open_time=ot, regime=cyc_regime[t], pnl=pnl, long_pnl=long_pnl,
                         short_pnl=short_pnl, gross_l=gross_l, gross_s=gross_s, netbeta=netbeta,
                         turn=turn, n_pos=len(net)))
        prev=net
    bk=pd.DataFrame(rows).set_index("open_time")

    # ---- context series (PIT, trailing) aligned to cycle times ----
    idx=bk.index
    # cross-sectional alpha dispersion this cycle (contemporaneous descriptor of the cycle)
    disp = d.groupby("open_time")["alpha_A"].std().reindex(idx)
    # market-wide pairwise corr of 4h returns, trailing 7d (42 cycles), PIT shift(1)
    rt=ret4.reindex(idx)
    def avg_pair_corr(window=42):
        out=pd.Series(index=idx, dtype=float)
        arr=rt.values
        for i in range(len(idx)):
            lo=max(0,i-window); sub=rt.iloc[lo:i]  # strictly trailing (excludes current)
            if len(sub)<20: continue
            cmat=sub.corr(min_periods=10).values
            iu=np.triu_indices_from(cmat,k=1)
            vals=cmat[iu]; vals=vals[np.isfinite(vals)]
            if len(vals)>0: out.iloc[i]=np.nanmean(vals)
        return out
    corr7d=avg_pair_corr(42)
    # breadth: frac of universe with positive 4h return THIS cycle (descriptor)
    breadth=(rt>0).sum(axis=1)/rt.notna().sum(axis=1)
    # btc context
    b30s=btc30.set_index("open_time")["b30"].reindex(idx)
    brv=btc_rvol_7d.reindex(idx)

    ctx=pd.DataFrame({"pnl":bk["pnl"]*1e4, "long_pnl":bk["long_pnl"]*1e4, "short_pnl":bk["short_pnl"]*1e4,
                      "regime":bk["regime"], "netbeta":bk["netbeta"], "turn":bk["turn"],
                      "b30":b30s, "btc_rvol_7d":brv, "xs_disp":disp, "corr7d":corr7d, "breadth":breadth})

    # ---- equity + deepest DD episode ----
    eq=ctx["pnl"].cumsum(); peak=eq.cummax(); dd=eq-peak
    mdd=dd.min(); trough=dd.idxmin()
    peak_t=eq.loc[:trough].idxmax()
    # recovery: first time after trough eq regains peak value (may be never)
    after=eq.loc[trough:]; peakval=eq.loc[peak_t]
    rec=after[after>=peakval]
    rec_t=rec.index[0] if len(rec)>0 else None
    print(f"Full: Sharpe {ann(ctx['pnl']/1e4):+.2f}  totPnL {eq.iloc[-1]:+.0f}bps  maxDD {mdd:+.0f}bps  Calmar {ctx['pnl'].mean()*6*365/abs(mdd):+.2f}")
    print(f"Deepest DD episode: peak {peak_t.date()} (eq {peakval:+.0f}) -> trough {trough.date()} (dd {mdd:+.0f}) -> recover {rec_t.date() if rec_t is not None else 'NEVER'}")
    dd_mask=(idx>=peak_t)&(idx<=trough)
    print(f"  DD window length: {dd_mask.sum()} cycles ({(trough-peak_t).days}d)")

    # regime mix inside vs outside DD
    print("\n--- regime mix (cycle counts) ---")
    print("  in-DD :", ctx.loc[dd_mask,"regime"].value_counts().to_dict())
    print("  out   :", ctx.loc[~dd_mask,"regime"].value_counts().to_dict())

    # leg attribution: long vs short contribution inside DD
    print("\n--- leg attribution (sum bps) ---")
    print(f"  in-DD : long {ctx.loc[dd_mask,'long_pnl'].sum():+.0f}  short {ctx.loc[dd_mask,'short_pnl'].sum():+.0f}  net {ctx.loc[dd_mask,'pnl'].sum():+.0f}")
    print(f"  out   : long {ctx.loc[~dd_mask,'long_pnl'].sum():+.0f}  short {ctx.loc[~dd_mask,'short_pnl'].sum():+.0f}  net {ctx.loc[~dd_mask,'pnl'].sum():+.0f}")
    # mean per-cycle leg PnL
    print(f"  in-DD mean/cyc: long {ctx.loc[dd_mask,'long_pnl'].mean():+.1f}  short {ctx.loc[dd_mask,'short_pnl'].mean():+.1f}")
    print(f"  out   mean/cyc: long {ctx.loc[~dd_mask,'long_pnl'].mean():+.1f}  short {ctx.loc[~dd_mask,'short_pnl'].mean():+.1f}")

    # context distribution inside vs outside DD
    print("\n--- context: mean (in-DD vs out), and t-like separation ---")
    for col in ["b30","btc_rvol_7d","xs_disp","corr7d","breadth","netbeta","turn"]:
        a=ctx.loc[dd_mask,col].dropna(); b=ctx.loc[~dd_mask,col].dropna()
        sep=(a.mean()-b.mean())/np.sqrt(a.var()/max(len(a),1)+b.var()/max(len(b),1)) if len(a)>1 and len(b)>1 else np.nan
        print(f"  {col:<14} in-DD {a.mean():+.4f}  out {b.mean():+.4f}  Δ {a.mean()-b.mean():+.4f}  sep {sep:+.2f}")

    # ---- does context PRECEDE bad cycles? quantile bins of NEXT-cycle pnl on lagged context ----
    print("\n--- predictive: mean next-cycle book PnL (bps) by quintile of PIT context (context known at t, pnl at t) ---")
    # context is already PIT (trailing/shift). align context_t with pnl_t (the pnl realized over cycle t, decision made with info up to t)
    for col in ["btc_rvol_7d","corr7d","b30"]:
        c=ctx[[col,"pnl"]].dropna()
        if len(c)<50: continue
        c["q"]=pd.qcut(c[col],5,labels=False,duplicates="drop")
        gb=c.groupby("q")["pnl"].agg(["mean","count"])
        sh=c.groupby("q")["pnl"].apply(lambda x: x.mean()/x.std()*np.sqrt(6*365) if x.std()>0 else np.nan)
        print(f"  {col}:")
        for q in gb.index:
            print(f"     q{q}: mean {gb.loc[q,'mean']:+6.1f} bps  Sharpe {sh.loc[q]:+5.2f}  n={int(gb.loc[q,'count'])}")

    # 2D: btc_rvol_7d x corr7d
    print("\n--- 2D: mean book PnL (bps) by (btc_rvol_7d tercile, corr7d tercile) ---")
    c=ctx[["btc_rvol_7d","corr7d","pnl"]].dropna()
    c["rq"]=pd.qcut(c["btc_rvol_7d"],3,labels=["lo","mid","hi"],duplicates="drop")
    c["cq"]=pd.qcut(c["corr7d"],3,labels=["lo","mid","hi"],duplicates="drop")
    piv=c.pivot_table(index="rq",columns="cq",values="pnl",aggfunc="mean")
    cnt=c.pivot_table(index="rq",columns="cq",values="pnl",aggfunc="count")
    print("  mean PnL:"); print(piv.round(1).to_string())
    print("  counts:"); print(cnt.to_string())

    # corr7d high vs low: side-regime short leg behaviour (the L/S-breaks hypothesis)
    print("\n--- corr7d hi vs lo: side-regime leg PnL ---")
    cc=ctx[ctx["regime"]=="side"][["corr7d","long_pnl","short_pnl","pnl"]].dropna()
    hi=cc[cc["corr7d"]>=cc["corr7d"].quantile(0.7)]; lo=cc[cc["corr7d"]<=cc["corr7d"].quantile(0.3)]
    print(f"  hi-corr (n={len(hi)}): long {hi['long_pnl'].mean():+.1f} short {hi['short_pnl'].mean():+.1f} net {hi['pnl'].mean():+.1f} Sharpe {ann(hi['pnl']/1e4):+.2f}")
    print(f"  lo-corr (n={len(lo)}): long {lo['long_pnl'].mean():+.1f} short {lo['short_pnl'].mean():+.1f} net {lo['pnl'].mean():+.1f} Sharpe {ann(lo['pnl']/1e4):+.2f}")

    # save context for downstream / implementation
    ctx.to_parquet(OUT/"iter002_hl70_context.parquet")
    print(f"\nSaved context -> {OUT/'iter002_hl70_context.parquet'}")

    # chart: equity + DD + corr7d/btc_rvol overlay
    fig,(a1,a2)=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    a1.plot(eq.index,eq.values,color="navy",lw=1.2,label="equity")
    a1.axvspan(peak_t,trough,color="crimson",alpha=0.12,label="deepest DD")
    a1.legend(); a1.grid(alpha=0.3); a1.set_ylabel("cum PnL (bps)")
    a1.set_title(f"HL70 base — maxDD {mdd:+.0f} ({peak_t.date()}->{trough.date()})")
    a2b=a2.twinx()
    a2.plot(idx,ctx["corr7d"],color="darkorange",lw=1.0,label="avg pair-corr 7d")
    a2b.plot(idx,ctx["btc_rvol_7d"],color="green",lw=0.9,alpha=0.7,label="btc rvol 7d")
    a2.axvspan(peak_t,trough,color="crimson",alpha=0.12)
    a2.set_ylabel("corr7d",color="darkorange"); a2b.set_ylabel("btc_rvol_7d",color="green"); a2.grid(alpha=0.3)
    fig.tight_layout(); png=OUT/"iter002_hl70_dd_anatomy.png"; fig.savefig(png,dpi=110)
    print(f"chart -> {png}")
    print(f"\nDone [{time.time()-t0:.0f}s]")


if __name__=="__main__":
    main()
