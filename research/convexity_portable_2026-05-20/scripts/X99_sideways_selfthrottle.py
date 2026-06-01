"""X99 — Self-throttling gate on the SIDEWAYS mean-rev sleeve.

X98 finding: drawdowns are the sideways V0 mean-rev sleeve INVERTING in 2023-H2
(Sharpe -0.72) and 2026 (-2.70). Bull momentum sleeve is fine. The clean held-book
dropped V3.1's conv_gate / rolling-IC quality filters → no protection when the
mean-rev edge is weak.

Test a PIT self-throttle on the SIDEWAYS sleeve only (bull/bear unchanged):
flatten (or half-size) the sideways basket whenever the sleeve's OWN trailing
realized edge has gone negative. Two throttle signals, both strictly PIT
(use only cycles < t):
  (A) trailing mean SIDE basket-spread (gross 1-cycle L-S return) over window W
  (B) trailing mean cross-sectional IC (corr(pred, alpha_A) across syms) over W
Variants: W in {30,60,90}; action = flat (w=0) or half (w*0.5) when signal<0.

GOAL: kill the 2023-H2 & 2026 sideways bleed WITHOUT hindsight → rescue maxDD &
2026 Sharpe while preserving the 2024-25 engine. Base = K=5 held-book (X97 DD winner).
Report whole-sample + per-year Sharpe/PnL + maxDD for each variant.
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
    print("=== X99 sideways self-throttle (PIT) ===\n", flush=True)
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
    # Pre-build per-cycle basket picks + realized 1-cycle spread + cross-sec IC (for sideways throttle signals)
    base_w=[]; ret_seq=[]; regime_seq=[]
    side_spread=[]   # realized gross L-S spread for the cycle's picks (NaN if not sideways/insufficient)
    side_ic=[]       # cross-sectional corr(pred, alpha_A) at this cycle (sideways only)
    for ot in times:
        g=by_t[ot]; rg=g["regime"].iloc[0]; regime_seq.append(rg)
        ret_seq.append(dict(zip(g["symbol"],g["return_pct"])))
        w={}; spr=np.nan; ic=np.nan
        if rg!="bear":
            key="mom30" if rg=="bull" else "pred"
            gg=g.dropna(subset=[key])
            if len(gg)>=2*K:
                gg=gg.sort_values(key); Lr=gg.tail(K); Sr=gg.head(K)
                L=Lr["symbol"].tolist(); S=Sr["symbol"].tolist()
                a=b=1.0
                if rg=="side":
                    brow=betas.loc[ot] if ot in betas.index else None
                    if brow is not None:
                        mbL=np.nanmean([brow.get(s,np.nan) for s in L]); mbS=np.nanmean([brow.get(s,np.nan) for s in S])
                        if np.isfinite(mbL) and np.isfinite(mbS) and mbL>0 and mbS>0: a=2*mbS/(mbL+mbS); b=2*mbL/(mbL+mbS)
                    # realized 1-cycle spread + IC (PIT: known at end of this cycle)
                    spr=Lr["return_pct"].mean()-Sr["return_pct"].mean()
                    gv=g.dropna(subset=["pred","alpha_A"])
                    if len(gv)>=8: ic=np.corrcoef(gv["pred"].rank(),gv["alpha_A"].rank())[0,1]
                for s in L: w[s]=w.get(s,0)+a/K
                for s in S: w[s]=w.get(s,0)-b/K
        base_w.append(w); side_spread.append(spr); side_ic.append(ic)
    idx=pd.to_datetime(times)
    reg=pd.Series(regime_seq, index=idx)
    spr_s=pd.Series(side_spread, index=idx); ic_s=pd.Series(side_ic, index=idx)

    def heldbook(weights):
        prev={}; pnl=[]
        for t in range(len(weights)):
            active=weights[max(0,t-HOLD+1):t+1]; net={}
            for w in active:
                for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
            alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
            rl=ret_seq[t]; pnl.append(sum(net.get(s,0)*rl.get(s,0.0) for s in net) - turn*0.5*COST); prev=net
        return pd.Series(pnl, index=idx)

    def throttled_weights(signal, W, action):
        # signal: per-cycle realized series (NaN off-sideways). Throttle SIDE basket at t using
        # trailing mean of signal over the last W non-NaN values strictly BEFORE t (lag HOLD to be
        # safe: a basket's realized spread isn't fully known until +HOLD cycles; use values whose
        # cycle index <= t-HOLD).
        sig=signal.values; out=[]
        # rolling buffer of (cycle_idx, value) for realized sideways signals
        for t in range(len(base_w)):
            w=base_w[t]
            if regime_seq[t]=="side" and w:
                # gather realized signal from cycles j with j<=t-HOLD and not NaN
                hi=t-HOLD
                if hi>=W:  # need at least W history
                    vals=sig[:hi+1]; vals=vals[~np.isnan(vals)]
                    if len(vals)>=W:
                        m=np.mean(vals[-W:])
                        if m<0:
                            if action=="flat": out.append({}); continue
                            else: out.append({s:v*0.5 for s,v in w.items()}); continue
            out.append(w)
        return out

    def report(name, pnl):
        pb=pnl*1e4; eq=pb.cumsum(); dd=(eq-eq.cummax()).min()
        yr={y:(g.sum(),ann(g/1e4)) for y,g in pb.groupby(pb.index.year)}
        s2326=" ".join(f"{y}:{yr[y][1]:+.2f}" for y in [2023,2024,2025,2026] if y in yr)
        print(f"  {name:<26}{ann(pnl):>+7.2f}{eq.iloc[-1]:>+9.0f}{dd:>+9.0f}   {s2326}", flush=True)

    print(f"  {'variant':<26}{'Sharpe':>7}{'totPnL':>9}{'maxDD':>9}   per-year Sharpe (2023..2026)")
    base=heldbook(base_w); report("base (no throttle) K=5", base)
    print("  -- throttle on trailing BASKET-SPREAD --")
    for W in [30,60,90]:
        for act in ["flat","half"]:
            report(f"spread W={W} {act}", heldbook(throttled_weights(spr_s, W, act)))
    print("  -- throttle on trailing cross-sec IC --")
    for W in [30,60,90]:
        for act in ["flat","half"]:
            report(f"IC W={W} {act}", heldbook(throttled_weights(ic_s, W, act)))

    print(f"\nGOAL: rescue 2023(-0.72)/2026(-2.70) sideways bleed without hurting 2024/25. PIT (lag {HOLD}).")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
