"""X100 — Nested-OOS validation of the X99 sideways self-throttle.

X99 winner (spread W=30 flat) was picked from a full-sample 12-variant sweep →
selection bias. De-bias: walk forward in 6-month BLOCKS; for each block pick the
throttle config (incl. no-throttle) that had the best Sharpe over ALL PRIOR blocks,
apply it to the current block, concatenate → nested-OOS pnl series. Block 1 uses
base (no history). Honest question: does past-block performance pick a config that
keeps winning forward, or was W=30/flat fitted to 2023/2026?

Candidates: base + {spread,IC} x W{30,60,90} x {flat,half} = 13.
Report: nested-OOS Sharpe/maxDD/per-year vs base vs the in-sample-best (oracle).
Also print which config each block selected (to see stability).
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
    print("=== X100 nested-OOS throttle validation ===\n", flush=True)
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
    base_w=[]; ret_seq=[]; regime_seq=[]; side_spread=[]; side_ic=[]
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
                    spr=Lr["return_pct"].mean()-Sr["return_pct"].mean()
                    gv=g.dropna(subset=["pred","alpha_A"])
                    if len(gv)>=8: ic=np.corrcoef(gv["pred"].rank(),gv["alpha_A"].rank())[0,1]
                for s in L: w[s]=w.get(s,0)+a/K
                for s in S: w[s]=w.get(s,0)-b/K
        base_w.append(w); side_spread.append(spr); side_ic.append(ic)
    idx=pd.to_datetime(times)
    spr_arr=np.array(side_spread); ic_arr=np.array(side_ic)

    def heldbook(weights):
        prev={}; pnl=[]
        for t in range(len(weights)):
            active=weights[max(0,t-HOLD+1):t+1]; net={}
            for w in active:
                for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
            alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
            rl=ret_seq[t]; pnl.append(sum(net.get(s,0)*rl.get(s,0.0) for s in net) - turn*0.5*COST); prev=net
        return pd.Series(pnl, index=idx)

    def throttled(sig, W, action):
        out=[]
        for t in range(len(base_w)):
            w=base_w[t]
            if regime_seq[t]=="side" and w:
                hi=t-HOLD
                if hi>=W:
                    vals=sig[:hi+1]; vals=vals[~np.isnan(vals)]
                    if len(vals)>=W and np.mean(vals[-W:])<0:
                        if action=="flat": out.append({}); continue
                        else: out.append({s:v*0.5 for s,v in w.items()}); continue
            out.append(w)
        return out

    # build all candidate pnl series
    cands={"base": heldbook(base_w)}
    for sig_name,sig in [("spread",spr_arr),("IC",ic_arr)]:
        for W in [30,60,90]:
            for act in ["flat","half"]:
                cands[f"{sig_name}_W{W}_{act}"]=heldbook(throttled(sig,W,act))
    names=list(cands.keys())
    M=pd.DataFrame({k:v for k,v in cands.items()})  # cycles x candidates

    # 6-month blocks
    block=(M.index.year-2023)*2 + (M.index.month>6).astype(int)
    block=pd.Series(block, index=M.index)
    blocks=sorted(block.unique())

    nested=pd.Series(0.0, index=M.index); picks=[]
    for bi_ in blocks:
        mask=block==bi_
        prior=block<bi_
        if prior.sum()<200:
            sel="base"
        else:
            sh={c:ann(M.loc[prior,c]) for c in names}
            sel=max(sh, key=lambda c: (sh[c] if np.isfinite(sh[c]) else -9))
        nested.loc[mask]=M.loc[mask,sel].values
        picks.append((bi_, sel))

    def line(name, pnl):
        pb=pnl*1e4; eq=pb.cumsum(); dd=(eq-eq.cummax()).min()
        yr={y:ann(g/1e4) for y,g in pb.groupby(pb.index.year)}
        s=" ".join(f"{y}:{yr[y]:+.2f}" for y in [2023,2024,2025,2026] if y in yr)
        print(f"  {name:<24}{ann(pnl):>+7.2f}{eq.iloc[-1]:>+9.0f}{dd:>+9.0f}   {s}", flush=True)

    print(f"  {'series':<24}{'Sharpe':>7}{'totPnL':>9}{'maxDD':>9}   per-year Sharpe")
    line("base (no throttle)", cands["base"])
    line("oracle spread_W30_flat", cands["spread_W30_flat"])
    line("NESTED-OOS", nested)
    print("\n  block selections (6-mo blocks; b = (yr-2023)*2 + H2):")
    for bi_,sel in picks:
        yr=2023+bi_//2; half="H1" if bi_%2==0 else "H2"
        print(f"    {yr}-{half}: {sel}", flush=True)

    print(f"\nVERDICT: nested-OOS Sharpe vs base {ann(cands['base']):+.2f} / oracle {ann(cands['spread_W30_flat']):+.2f}.")
    print(f"If nested >= base and rescues 2026, throttle generalizes. Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
