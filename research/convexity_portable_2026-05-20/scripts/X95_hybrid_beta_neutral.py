"""X95 — Beta-neutral leg sizing on the HYBRID: does it help (sideways drag) or
hurt (removes bull long-beta tailwind)? And a regime-conditional version.

X94 showed: sideways net-beta -0.071 (~0 PnL), bull net-beta +0.286 (+18% PnL tailwind),
overall +0.046 (near-neutral). So blanket beta-neutral removes the bull tailwind. Test:
  EQUAL   : equal-weight legs (current)                          [baseline]
  BN_ALL  : beta-neutral both regimes (scale legs to net-beta=0) [purer alpha, no dir risk]
  BN_SIDE : beta-neutral sideways only, keep bull long-beta      [neutralize the drag, keep tailwind]
Held-book (24h overlap, flat-bear), net of cost, 3yr + 12mo.
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


def leg_weights(L, S, bL, bS, mode, regime):
    """mode: equal / bn_all / bn_side. Returns weight dict."""
    do_bn = (mode=="bn_all") or (mode=="bn_side" and regime=="side")
    a=b=1.0
    if do_bn and len(bL)>0 and len(bS)>0:
        mbL, mbS = np.nanmean(bL), np.nanmean(bS)
        if np.isfinite(mbL) and np.isfinite(mbS) and (mbL+mbS)!=0 and mbL>0 and mbS>0:
            a=2*mbS/(mbL+mbS); b=2*mbL/(mbL+mbS)
    w={}
    for s in L: w[s]=w.get(s,0)+a/K
    for s in S: w[s]=w.get(s,0)-b/K
    return w


def main():
    t0=time.time()
    print("=== X95 hybrid beta-neutral leg sizing ===\n", flush=True)
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
    def build(times_sub, mode):
        ws=[]; rs=[]
        for ot in times_sub:
            g=by_t[ot]; rg=g["regime"].iloc[0]
            rs.append(dict(zip(g["symbol"],g["return_pct"])))
            if rg=="bear": ws.append({}); continue
            key="mom30" if rg=="bull" else "pred"
            gg=g.dropna(subset=[key])
            if len(gg)<2*K: ws.append({}); continue
            gg=gg.sort_values(key); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
            brow=betas.loc[ot] if ot in betas.index else None
            bL=[brow.get(s,np.nan) for s in L] if brow is not None else []
            bS=[brow.get(s,np.nan) for s in S] if brow is not None else []
            ws.append(leg_weights(L,S,bL,bS,mode,rg))
        return ws, rs

    for tag, tsub in [("3yr", times), ("12mo", [t for t in times if t>=pd.Timestamp("2025-05-01",tz="UTC")])]:
        print(f"--- {tag} (held-book, flat-bear, net) ---")
        for mode in ["equal","bn_all","bn_side"]:
            ws, rs = build(tsub, mode)
            r=heldbook(ws, rs)
            lbl={"equal":"EQUAL","bn_all":"BETA-NEUTRAL (all)","bn_side":"BETA-NEUTRAL (sideways only)"}[mode]
            print(f"  {lbl:<28} Sharpe={ann(r):+.2f}  PnL={r.sum()*1e4:+.0f}bps", flush=True)

    print(f"\nVERDICT: BN_all (purer alpha, kills bull tailwind) vs BN_side (keep bull beta) vs EQUAL.")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
