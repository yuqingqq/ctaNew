"""Validate convexity_paper_bot replay PnL against the X116-style research engine
on x132 preds, same window, same eligibility filter. Pass = cycle-by-cycle PnL
match to within numerical tolerance + Sharpe within 0.1.
"""
from __future__ import annotations
import sys, pickle, time
from pathlib import Path
from collections import deque
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"
COST=4.5e-4; K=5; HOLD=6

def load_close_4h(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return pd.Series(dtype="float64")
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    if not dfs: return pd.Series(dtype="float64")
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    c=df.set_index("open_time")["close"].astype(np.float64)
    return c[(c.index.hour%4==0)&(c.index.minute==0)]

def main():
    bot=pd.read_csv(REPO/"live/state/convexity/cycles.csv")
    bot["open_time"]=pd.to_datetime(bot["open_time"],utc=True)
    bot=bot.sort_values("open_time").reset_index(drop=True)
    t_start=bot["open_time"].min(); t_end=bot["open_time"].max()
    print(f"Bot window: {t_start} -> {t_end}, {len(bot)} cycles")

    # Load preds in same window
    d=pd.read_parquet(PREDS)
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy()
    d=d[(d["open_time"]>=t_start)&(d["open_time"]<=t_end)]
    syms=sorted(d["symbol"].unique())
    print(f"Research preds: {len(d):,} rows × {len(syms)} syms")

    # mom30 + beta + btc30 (same as bot)
    btc=load_close_4h("BTCUSDT")
    br=np.log(btc/btc.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    mom_rows={}; betas={}
    for s in syms:
        c=load_close_4h(s)
        if c.empty: continue
        mom_rows[s]=(c/c.shift(180)-1).shift(1)
        r=np.log(c/c.shift(1)); ri,bi=r.align(br,join="inner")
        betas[s]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
    d["mom30"]=d.apply(lambda r: mom_rows.get(r["symbol"],pd.Series()).get(r["open_time"],np.nan),axis=1)
    btc30=(btc/btc.shift(180)-1).reindex(sorted(d["open_time"].unique())).rename("b30").reset_index()
    btc30.columns=["open_time","b30"]
    d=d.merge(btc30,on="open_time",how="left")
    d["regime"]=np.where(d["b30"]>0.10,"bull",np.where(d["b30"]<-0.10,"bear","side"))

    # Apply same eligibility filter as bot: maturity>=180d from FULL preds
    full=pd.read_parquet(PREDS, columns=["symbol","open_time"])
    full["open_time"]=pd.to_datetime(full["open_time"],utc=True)
    full=full[(full["open_time"].dt.hour%4==0)&(full["open_time"].dt.minute==0)]
    earliest={s:g["open_time"].min() for s,g in full.groupby("symbol")}
    HYGIENE={"USDCUSDT","BUSDUSDT","TUSDUSDT","DAIUSDT","FDUSDUSDT","PAXGUSDT","XAUTUSDT",
             "WBTCUSDT","WBETHUSDT","WSTETHUSDT","WETHUSDT","STETHUSDT"}

    by_t={ot:g for ot,g in d.groupby("open_time")}
    times=sorted(by_t.keys())
    ws=[]; cycle_pnls=[]; cycle_pos=[]
    prev={}
    INITIAL=10000.0; equity=INITIAL
    for ot in times:
        g=by_t[ot]
        # eligibility
        elig=[s for s in g["symbol"].unique()
              if s not in HYGIENE and (ot-earliest.get(s,ot)).days>=180]
        g=g[g["symbol"].isin(elig)]
        rg=g["regime"].iloc[0] if len(g) else "unknown"
        if rg=="bear":
            ws.append({})
        else:
            key="mom30" if rg=="bull" else "pred"
            gg=g.dropna(subset=[key])
            if len(gg)<2*K:
                ws.append({})
            else:
                gg=gg.sort_values(key)
                L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
                a=b=1.0
                if rg=="side":
                    blist={s: betas.get(s,pd.Series()).get(ot,np.nan) for s in syms}
                    mbL=np.nanmean([blist.get(s,np.nan) for s in L])
                    mbS=np.nanmean([blist.get(s,np.nan) for s in S])
                    if np.isfinite(mbL) and np.isfinite(mbS) and mbL>0 and mbS>0:
                        a=2*mbS/(mbL+mbS); b=2*mbL/(mbL+mbS)
                w={}
                for s in L: w[s]=w.get(s,0)+a/K
                for s in S: w[s]=w.get(s,0)-b/K
                ws.append(w)
        # heldbook aggregate
        active=ws[max(0,len(ws)-HOLD):]
        net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        rmap=dict(zip(g["symbol"],g["return_pct"]))
        gross=sum(net.get(s,0)*rmap.get(s,0.0) for s in net if np.isfinite(rmap.get(s,0.0)))
        allk=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in allk)
        pnl=gross-turn*0.5*COST
        equity*= (1.0+pnl)
        cycle_pnls.append(pnl*1e4)   # bps
        cycle_pos.append({"open_time":ot,"research_pnl_bps":pnl*1e4,
                          "research_gross":sum(abs(v) for v in net.values()),
                          "research_top":",".join(sorted(s for s,w in net.items() if w>0)),
                          "research_bot":",".join(sorted(s for s,w in net.items() if w<0))})
        prev=net

    res=pd.DataFrame(cycle_pos)
    cmp=bot.merge(res,on="open_time",how="inner")
    cmp["pnl_diff_bps"]=cmp["pnl_bps"]-cmp["research_pnl_bps"]
    print(f"\n=== Cycle-by-cycle comparison ({len(cmp)} cycles) ===")
    print(f"Bot      Sharpe(ann) {(bot['pnl_bps']/1e4).mean()/(bot['pnl_bps']/1e4).std()*np.sqrt(6*365):+.3f}  totPnL {bot['pnl_bps'].sum():+.0f} bps")
    rp=pd.Series(cycle_pnls)/1e4
    print(f"Research Sharpe(ann) {rp.mean()/rp.std()*np.sqrt(6*365):+.3f}  totPnL {rp.sum()*1e4:+.0f} bps")
    print(f"\nDiff stats (bot - research) bps/cycle:")
    print(f"  mean   {cmp['pnl_diff_bps'].mean():+.3f}")
    print(f"  median {cmp['pnl_diff_bps'].median():+.3f}")
    print(f"  std    {cmp['pnl_diff_bps'].std():.3f}")
    print(f"  max abs {cmp['pnl_diff_bps'].abs().max():.3f}")
    print(f"  cycles where |diff| > 1bps: {(cmp['pnl_diff_bps'].abs()>1).sum()}/{len(cmp)}")
    cmp.to_csv(REPO/"live/state/convexity/replay_vs_research.csv",index=False)
    print(f"\nSaved {REPO/'live/state/convexity/replay_vs_research.csv'}")

    # show first 5 cycles for eyeball check
    print(f"\nFirst 5 cycles:")
    cols=["open_time","regime","top_k_long","bot_k_short","gross_target","pnl_bps","research_gross","research_pnl_bps","pnl_diff_bps"]
    print(cmp[cols].head(5).to_string(index=False))

if __name__=="__main__": main()
