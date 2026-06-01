"""X97 — Drawdown-reduction levers on the optimized hybrid.

Problem: maxDD -56%, 44.7% positive cycles, ±19% single-cycle swings — driven by
EQUAL-weighting K=3 (volatile alts get same weight as majors → single-name blowups
move the book). Test risk-based fixes:
  1. INVERSE-VOL leg weighting (w_i ∝ 1/rvol_i) — shrink volatile-alt weights
  2. K sweep (3,5,7) — more diversification
  3. VOL-TARGET overlay — scale book to constant trailing realized vol (de-risk turbulence)
Measure Sharpe, maxDD, Calmar, %pos. GOAL: cut maxDD while preserving Sharpe (raise Calmar).
Base config: momentum bull / mean-rev sideways (BN-sideways) / flat bear, held-book.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RCACHE = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
COST=4.5e-4; HOLD=6


def load_close(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def stats(pnl):
    pnl=pd.Series(pnl); pb=pnl*1e4; eq=pb.cumsum(); dd=(eq-eq.cummax())
    sh=pnl.mean()/pnl.std()*np.sqrt(6*365) if pnl.std()>0 else np.nan
    mdd=dd.min(); annr=pb.mean()*6*365
    return sh, mdd, (annr/abs(mdd) if mdd<0 else np.nan), (pb>0).mean()*100, eq.iloc[-1]


def main():
    t0=time.time()
    print("=== X97 drawdown-reduction levers ===\n", flush=True)
    apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
    apd=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)]
    syms=sorted(apd["symbol"].unique())
    print("mom + rvol + beta...", flush=True)
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    mom_rows=[]; rvol_map={}; beta_map={}
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]; r=np.log(c4/c4.shift(1))
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,"mom30":(c4/c4.shift(180)-1).shift(1).values}))
        rvol_map[sym]=r.rolling(42,min_periods=12).std().shift(1)  # 7d trailing vol, PIT
        ri,bi=r.align(br,join="inner")
        beta_map[sym]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    rvols=pd.concat([s.rename(k) for k,s in rvol_map.items()],axis=1)
    betas=pd.concat([s.rename(k) for k,s in beta_map.items()],axis=1)
    apd=apd.merge(mom,on=["symbol","open_time"],how="left")
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index()
    btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    apd=apd.merge(btc30,on="open_time",how="left").dropna(subset=["btc_ret_30d"])
    apd["regime"]=np.where(apd["btc_ret_30d"]>0.10,"bull",np.where(apd["btc_ret_30d"]<-0.10,"bear","side"))
    times=sorted(apd["open_time"].unique()); by_t={ot:g for ot,g in apd.groupby("open_time")}

    def build_weights(K, weighting):
        ws=[]; rs=[]
        for ot in times:
            g=by_t[ot]; rg=g["regime"].iloc[0]; rs.append(dict(zip(g["symbol"],g["return_pct"])))
            if rg=="bear": ws.append({}); continue
            key="mom30" if rg=="bull" else "pred"
            gg=g.dropna(subset=[key])
            if len(gg)<2*K: ws.append({}); continue
            gg=gg.sort_values(key); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
            rvrow=rvols.loc[ot] if ot in rvols.index else None
            def legw(names):
                if weighting=="invvol" and rvrow is not None:
                    iv=np.array([1.0/max(rvrow.get(s,np.nan),1e-6) if np.isfinite(rvrow.get(s,np.nan)) else np.nan for s in names])
                    if np.isfinite(iv).sum()<len(names): return np.ones(len(names))/len(names)
                    return iv/iv.sum()
                return np.ones(len(names))/len(names)
            # beta-neutral scale in sideways (keep prior refinement)
            a=bsc=1.0
            if rg=="side":
                brow=betas.loc[ot] if ot in betas.index else None
                if brow is not None:
                    mbL=np.nanmean([brow.get(s,np.nan) for s in L]); mbS=np.nanmean([brow.get(s,np.nan) for s in S])
                    if np.isfinite(mbL) and np.isfinite(mbS) and mbL>0 and mbS>0: a=2*mbS/(mbL+mbS); bsc=2*mbL/(mbL+mbS)
            wL=legw(L)*a; wS=legw(S)*bsc; w={}
            for s,wt in zip(L,wL): w[s]=w.get(s,0)+wt
            for s,wt in zip(S,wS): w[s]=w.get(s,0)-wt
            ws.append(w)
        return ws, rs

    def heldbook(ws, rs, voltarget=None):
        prev={}; pnl=[]
        for t in range(len(ws)):
            active=ws[max(0,t-HOLD+1):t+1]; net={}
            for w in active:
                for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
            # vol-target overlay: scale net by target / trailing realized pnl-vol
            scale=1.0
            if voltarget is not None and len(pnl)>=42:
                rv=np.std(pnl[-42:])
                if rv>0: scale=min(2.0, voltarget/rv)
            net={s:v*scale for s,v in net.items()}
            alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
            rl=rs[t]; pnl.append(sum(net.get(s,0)*rl.get(s,0.0) for s in net) - turn*0.5*COST); prev=net
        return np.array(pnl)

    print(f"  {'config':<34}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'%pos':>7}{'totPnL':>9}")
    # baseline + levers
    configs=[
        ("equal K=3 (baseline)", 3, "equal", None),
        ("invvol K=3", 3, "invvol", None),
        ("equal K=5", 5, "equal", None),
        ("invvol K=5", 5, "invvol", None),
        ("invvol K=7", 7, "invvol", None),
    ]
    # determine a vol-target = median trailing vol of baseline for the overlay tests
    wsb, rsb = build_weights(3,"equal"); base=heldbook(wsb, rsb)
    vt=np.median([np.std(base[max(0,i-42):i]) for i in range(42,len(base))])
    for name,K,wt,_ in configs:
        ws,rs=build_weights(K,wt); p=heldbook(ws,rs)
        sh,mdd,cal,pos,tot=stats(p)
        print(f"  {name:<34}{sh:>+8.2f}{mdd*1e4:>+9.0f}{cal:>+8.2f}{pos:>7.1f}{tot*1e4:>+9.0f}", flush=True)
    # vol-target overlays on invvol K=5
    for name,K,wt in [("invvol K=5 + vol-target", 5,"invvol"), ("invvol K=7 + vol-target",7,"invvol")]:
        ws,rs=build_weights(K,wt); p=heldbook(ws,rs,voltarget=vt)
        sh,mdd,cal,pos,tot=stats(p)
        print(f"  {name:<34}{sh:>+8.2f}{mdd*1e4:>+9.0f}{cal:>+8.2f}{pos:>7.1f}{tot*1e4:>+9.0f}", flush=True)

    print(f"\nGOAL: lower |maxDD| + higher Calmar while keeping Sharpe. invvol shrinks volatile-alt")
    print(f"weights (cuts ±19% tails); K↑ diversifies; vol-target de-risks turbulence.")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
