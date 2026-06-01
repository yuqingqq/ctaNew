"""iter-006 — ROOT-CAUSE decomposition of the HL70 deep drawdown.

Reuses the iter-002 held-book engine EXACTLY (K=5, HOLD=6, mom-bull/mean-rev-side-BN/
flat-bear, trailing-180-bar shift(1) betas for the side beta-neutral sizing) but instruments
it to TEST four competing mechanisms for the −6,142 bps long-leg loss in the deep DD window:

  H1 dispersion collapse / correlation spike (cross-sec spread the mean-rev book harvests)
  H2 stale-beta hedge failure  -> decompose long-leg PnL into ALPHA vs BETA using REALIZED
     (forward-window) betas, and compare realized net-beta vs the trailing net-beta the hedge used
  H3 alpha sign/decay          -> per-fold cross-sec IC of pred vs forward alpha (side regime)
  H4 regime mislabel           -> BTC 30d vs equal-weight-alt-index 30d in the DD window

All PIT for the strategy itself; the realized-beta decomposition uses forward info ONLY for
ATTRIBUTION (explaining the loss), never for the trading decision.
"""
from __future__ import annotations
import time, json
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"research/convexity_portable_2026-05-20/results"
PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"
K=5; HOLD=6; COST=4.5e-4


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
    print("=== iter-006 HL70 ROOT-CAUSE decomposition ===\n", flush=True)
    d=pd.read_parquet(PREDS, columns=["symbol","open_time","pred","alpha_A","return_pct"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy().sort_values(["symbol","open_time"])

    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    # BTC 4h-FORWARD return aligned at open_time (matches return_pct horizon = open->exit +4h)
    btc_fwd=(b4.shift(-1)/b4-1.0)  # simple fwd return over the holding 4h

    syms=sorted(d["symbol"].unique()); mom_rows=[]; beta_tr={}; ret4_map={}
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,"mom30":(c4/c4.shift(180)-1).shift(1).values}))
        r=np.log(c4/c4.shift(1)); ri,bi=r.align(br,join="inner")
        # TRAILING beta used by the hedge (shift(1), 180-bar 30d window)
        beta_tr[sym]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
        ret4_map[sym]=r
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    betas=pd.concat([s.rename(k) for k,s in beta_tr.items()],axis=1)
    ret4=pd.concat([s.rename(k) for k,s in ret4_map.items()],axis=1).sort_index()

    # REALIZED forward beta over the HOLD window (attribution only, not used for trading):
    # cov(r_i, r_btc) / var(r_btc) over the FORWARD HOLD-bar window centered on the holding period.
    fwdwin=42  # use trailing-42 (7d) realized beta evaluated AT cycle t = contemporaneous regime beta
    real_beta={}
    for sym in syms:
        if sym not in ret4.columns: continue
        ri,bi=ret4[sym].align(br,join="inner")
        # realized beta over a SHORT contemporaneous 7d window (no shift) -> "what beta actually was"
        rb=(ri.rolling(fwdwin,min_periods=20).cov(bi)/br.rolling(fwdwin,min_periods=20).var())
        real_beta[sym]=rb
    rbetas=pd.concat([s.rename(k) for k,s in real_beta.items()],axis=1)

    d=d.merge(mom,on=["symbol","open_time"],how="left")
    btc30=(b4/b4.shift(180)-1).to_frame("b30").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left").dropna(subset=["b30"])
    d["regime"]=np.where(d["b30"]>0.10,"bull",np.where(d["b30"]<-0.10,"bear","side"))
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}

    # ---- weights identical to X117 ----
    cyc_w=[]; cyc_regime=[]; cyc_L=[]; cyc_S=[]; cyc_ab=[]
    for ot in times:
        g=by_t[ot]; rg=g["regime"].iloc[0]; cyc_regime.append(rg)
        if rg=="bear": cyc_w.append({}); cyc_L.append([]); cyc_S.append([]); cyc_ab.append((1.,1.)); continue
        key="mom30" if rg=="bull" else "pred"; gg=g.dropna(subset=[key])
        if len(gg)<2*K: cyc_w.append({}); cyc_L.append([]); cyc_S.append([]); cyc_ab.append((1.,1.)); continue
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
        cyc_w.append(w); cyc_L.append(L); cyc_S.append(S); cyc_ab.append((a,b))

    # ---- held book with FULL attribution incl trailing vs realized net beta ----
    prev={}; rows=[]
    for t in range(len(cyc_w)):
        ot=times[t]
        active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
        rl=by_t[ot]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
        btc_ret=btc_fwd.get(ot,np.nan)  # BTC 4h-fwd return over the holding cycle
        long_pnl=sum(net[s]*rmap.get(s,0.0) for s in net if net[s]>0 and np.isfinite(rmap.get(s,0.0)))
        short_pnl=sum(net[s]*rmap.get(s,0.0) for s in net if net[s]<0 and np.isfinite(rmap.get(s,0.0)))
        gl=sum(v for v in net.values() if v>0); gs=-sum(v for v in net.values() if v<0)
        # trailing net beta (what the hedge believed)
        bt=betas.loc[ot] if ot in betas.index else None
        rb=rbetas.loc[ot] if ot in rbetas.index else None
        nbeta_tr=nbeta_real=lbeta_tr=lbeta_real=np.nan
        if bt is not None:
            nbeta_tr=sum(net[s]*bt.get(s,np.nan) for s in net if np.isfinite(bt.get(s,np.nan)))
            lbeta_tr=sum(net[s]*bt.get(s,np.nan) for s in net if net[s]>0 and np.isfinite(bt.get(s,np.nan)))
        if rb is not None:
            nbeta_real=sum(net[s]*rb.get(s,np.nan) for s in net if np.isfinite(rb.get(s,np.nan)))
            lbeta_real=sum(net[s]*rb.get(s,np.nan) for s in net if net[s]>0 and np.isfinite(rb.get(s,np.nan)))
        cost=turn*0.5*COST; pnl=long_pnl+short_pnl-cost
        # alpha/beta split of LONG leg pnl using REALIZED beta: beta_pnl = lbeta_real * btc_ret ; alpha = resid
        long_beta_pnl = lbeta_real*btc_ret if (np.isfinite(lbeta_real) and np.isfinite(btc_ret)) else np.nan
        long_alpha_pnl = long_pnl - long_beta_pnl if np.isfinite(long_beta_pnl) else np.nan
        rows.append(dict(open_time=ot, regime=cyc_regime[t], pnl=pnl*1e4, long_pnl=long_pnl*1e4,
                         short_pnl=short_pnl*1e4, gl=gl, gs=gs, btc_ret=btc_ret,
                         nbeta_tr=nbeta_tr, nbeta_real=nbeta_real, lbeta_tr=lbeta_tr, lbeta_real=lbeta_real,
                         long_beta_pnl=(long_beta_pnl*1e4 if np.isfinite(long_beta_pnl) else np.nan),
                         long_alpha_pnl=(long_alpha_pnl*1e4 if np.isfinite(long_alpha_pnl) else np.nan),
                         turn=turn))
        prev=net
    bk=pd.DataFrame(rows).set_index("open_time")

    # ---- deepest DD window ----
    eq=bk["pnl"].cumsum(); peak=eq.cummax(); dd=eq-peak
    trough=dd.idxmin(); peak_t=eq.loc[:trough].idxmax(); mdd=dd.min()
    ddm=(bk.index>=peak_t)&(bk.index<=trough)
    print(f"Full Sharpe {ann(bk['pnl']/1e4):+.2f} maxDD {mdd:+.0f}  DD window {peak_t.date()}->{trough.date()} ({ddm.sum()} cyc)")

    sidem = (bk["regime"]=="side")
    dd_side = ddm & sidem
    print(f"\nDD-window leg sums: long {bk.loc[ddm,'long_pnl'].sum():+.0f}  short {bk.loc[ddm,'short_pnl'].sum():+.0f}  net {bk.loc[ddm,'pnl'].sum():+.0f}")
    print(f"  of which SIDE-regime: long {bk.loc[dd_side,'long_pnl'].sum():+.0f}  short {bk.loc[dd_side,'short_pnl'].sum():+.0f}  net {bk.loc[dd_side,'pnl'].sum():+.0f}  (n_side={dd_side.sum()})")

    # ============ H2: ALPHA vs BETA decomposition of the long-leg loss ============
    print("\n===== H2  long-leg ALPHA vs BETA (realized-beta attribution) =====")
    L_total = bk.loc[ddm,"long_pnl"].sum()
    L_beta  = bk.loc[ddm,"long_beta_pnl"].sum()
    L_alpha = bk.loc[ddm,"long_alpha_pnl"].sum()
    print(f"  in-DD long-leg total {L_total:+.0f} = beta {L_beta:+.0f} + alpha {L_alpha:+.0f}")
    print(f"     beta explains {100*L_beta/L_total:.0f}% of the long-leg loss ; alpha-residual {100*L_alpha/L_total:.0f}%")
    Ls=bk.loc[dd_side,"long_pnl"].sum(); Lsb=bk.loc[dd_side,"long_beta_pnl"].sum(); Lsa=bk.loc[dd_side,"long_alpha_pnl"].sum()
    print(f"  in-DD SIDE long-leg {Ls:+.0f} = beta {Lsb:+.0f} ({100*Lsb/Ls:.0f}%) + alpha {Lsa:+.0f} ({100*Lsa/Ls:.0f}%)")

    print("\n  net-beta the hedge BELIEVED (trailing) vs REALIZED (7d contemporaneous):")
    for lab,m in [("out-of-DD",~ddm),("in-DD",ddm),("in-DD side",dd_side)]:
        print(f"   {lab:<12} nbeta_tr {bk.loc[m,'nbeta_tr'].mean():+.3f}  nbeta_real {bk.loc[m,'nbeta_real'].mean():+.3f}  "
              f"long_beta_tr {bk.loc[m,'lbeta_tr'].mean():+.3f}  long_beta_real {bk.loc[m,'lbeta_real'].mean():+.3f}")
    # is the residual net-beta x btc-ret the loss?
    nb_btc = (bk.loc[ddm,"nbeta_real"]*bk.loc[ddm,"btc_ret"]*1e4)
    print(f"  sum(realized_net_beta * btc_ret) in-DD = {nb_btc.sum():+.0f} bps (the directional-beta P&L of the WHOLE book)")

    # ============ H1: dispersion collapse / correlation spike ============
    print("\n===== H1  dispersion / correlation (in-DD vs out) =====")
    disp = d.groupby("open_time")["alpha_A"].std().reindex(bk.index)   # cross-sec dispersion of model target
    rdisp= d.groupby("open_time")["return_pct"].std().reindex(bk.index) # cross-sec dispersion of realized ret
    rt=ret4.reindex(bk.index)
    def avg_pair_corr(window=42):
        out=pd.Series(index=bk.index,dtype=float)
        for i in range(len(bk.index)):
            sub=rt.iloc[max(0,i-window):i]
            if len(sub)<20: continue
            cm=sub.corr(min_periods=10).values; iu=np.triu_indices_from(cm,k=1)
            v=cm[iu]; v=v[np.isfinite(v)]
            if len(v)>0: out.iloc[i]=np.nanmean(v)
        return out
    corr7d=avg_pair_corr(42)
    for lab,a in [("xs_disp(target)",disp),("xs_disp(realized ret)",rdisp),("corr7d",corr7d)]:
        i=a[ddm].dropna(); o=a[~ddm].dropna()
        print(f"  {lab:<22} in-DD {i.mean():+.5f}  out {o.mean():+.5f}  ratio {i.mean()/o.mean():.2f}")
    # does dispersion correlate with side long-leg pnl?
    sd=pd.DataFrame({"disp":rdisp,"corr7d":corr7d,"long_pnl":bk["long_pnl"]})[sidem].dropna()
    print(f"  side-regime corr(realized-ret dispersion, long_pnl) = {sd['disp'].corr(sd['long_pnl']):+.3f}")
    print(f"  side-regime corr(corr7d, long_pnl)                  = {sd['corr7d'].corr(sd['long_pnl']):+.3f}")

    # ============ H3: alpha sign / decay (cross-sec IC of pred vs fwd alpha, by fold) ============
    print("\n===== H3  cross-sectional IC of pred vs alpha_A, side regime, per period =====")
    dd2=d.copy(); dd2["dd"]=dd2["open_time"].isin(bk.index[ddm])
    side_d=dd2[dd2["regime"]=="side"]
    def xsic(df):
        ics=[]
        for ot,g in df.groupby("open_time"):
            gg=g.dropna(subset=["pred","alpha_A"])
            if len(gg)>=8: ics.append(gg["pred"].corr(gg["alpha_A"],method="spearman"))
        ics=[x for x in ics if np.isfinite(x)]
        return np.mean(ics),len(ics)
    for lab,sub in [("side ALL",side_d),("side in-DD",side_d[side_d["dd"]]),("side out-DD",side_d[~side_d["dd"]])]:
        m,n=xsic(sub); print(f"  {lab:<14} mean XS-IC {m:+.4f}  (n_cyc={n})")

    # ============ H4: regime mislabel (BTC30 vs equal-weight alt index 30d) ============
    print("\n===== H4  regime label: BTC-30d vs equal-weight ALT-index 30d =====")
    altidx=ret4.drop(columns=[c for c in ["BTCUSDT","ETHUSDT"] if c in ret4.columns]).mean(axis=1)  # eq-wt alt 4h ret
    alt_cum=altidx.cumsum()
    alt30=(alt_cum-alt_cum.shift(180)).reindex(bk.index)   # ~30d cumulative alt log-ret
    b30s=btc30.set_index("open_time")["b30"].reindex(bk.index)
    print(f"  in-DD window: BTC-30d mean {b30s[ddm].mean():+.3f}  |  alt-index 30d cum-ret mean {alt30[ddm].mean():+.3f}")
    print(f"  out         : BTC-30d mean {b30s[~ddm].mean():+.3f}  |  alt-index 30d cum-ret mean {alt30[~ddm].mean():+.3f}")
    # how many in-DD side cycles had alts in DD (alt30<-10%) while BTC said "sideways"?
    mislabel = ddm & sidem & (alt30 < -0.10)
    print(f"  in-DD SIDE cycles where BTC=sideways BUT alt-index 30d < -10% (hidden alt-bear): {mislabel.sum()} / {dd_side.sum()}")
    print(f"     long-leg pnl on those mislabeled cycles: {bk.loc[mislabel,'long_pnl'].sum():+.0f} bps")

    bk.to_parquet(OUT/"iter006_rootcause_HL70.parquet")
    print(f"\nSaved -> {OUT/'iter006_rootcause_HL70.parquet'}  [{time.time()-t0:.0f}s]")


if __name__=="__main__":
    main()
