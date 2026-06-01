"""X89 — CLEAN flat-in-bear test (continuous timeline held-book, no row-dropping).

X88's H3f dropped bear ROWS → broke the sleeve's continuity (gap artifact). A proper
"flat in bear" keeps the timeline continuous and simply holds NO position during bear
(the 24h book winds down naturally). This isolates the real effect.

Held-book (24h hold, 6 overlapping baskets), continuous timeline, net of cost. Variants:
  H2     : bull→momentum, side+bear→mean-rev (bear traded)         [the 2-regime hybrid]
  FLATB  : bull→momentum, side→mean-rev, bear→EMPTY basket (flat)  [clean flat-in-bear]
  REDB   : bear→half-size mean-rev (partial)                       [optional middle ground]
Compares net Sharpe + PnL, 3yr + 12mo. Also reports bear-cycle PnL contribution under H2
(does the held book actually lose in bear, or does mean-rev there contribute?).
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


def heldbook(cyc_w, cyc_times, ret_lookup):
    """cyc_w: list of dict sym->target weight per cycle (empty dict = flat).
    Net position = avg of last HOLD baskets. PnL = sum(net·ret) - turnover cost."""
    n=len(cyc_w); prev={}; rets=[]; bear_pnls=[]
    for t in range(n):
        active=cyc_w[max(0,t-HOLD+1):t+1]
        net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
        cost=turn*0.5*COST
        rl=ret_lookup.get(cyc_times[t],{})
        pnl=sum(net.get(s,0)*rl.get(s,0.0) for s in net)
        rets.append(pnl-cost); prev=net
    return pd.Series(rets,index=cyc_times)


def main():
    t0=time.time()
    print("=== X89 clean flat-in-bear (continuous held-book) ===\n", flush=True)
    apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
    apd=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)]
    syms=sorted(apd["symbol"].unique())

    print("mom_30d...", flush=True)
    mr=[]
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom=(c4/c4.shift(180)-1).shift(1)
        mr.append(pd.DataFrame({"symbol":sym,"open_time":mom.index,"mom30":mom.values}))
    mom=pd.concat(mr,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    apd=apd.merge(mom,on=["symbol","open_time"],how="left")

    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index()
    btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    apd=apd.merge(btc30,on="open_time",how="left").dropna(subset=["btc_ret_30d"])
    apd["regime"]=np.where(apd["btc_ret_30d"]>0.10,"bull",np.where(apd["btc_ret_30d"]<-0.10,"bear","side"))

    ret_lookup={ot:dict(zip(g["symbol"],g["return_pct"])) for ot,g in apd.groupby("open_time")}

    def build(sub, bear_mode):
        """bear_mode in {'meanrev','flat','half'}. Returns cyc_times, cyc_w, regimes."""
        times=sorted(sub["open_time"].unique()); ws=[]; regs=[]
        for ot in times:
            g=sub[sub["open_time"]==ot]; rg=g["regime"].iloc[0]; regs.append(rg)
            if rg=="bull":
                key="mom30"; gg=g.dropna(subset=[key]).sort_values(key); scale=1.0
            elif rg=="bear":
                if bear_mode=="flat": ws.append({}); continue
                key="pred"; gg=g.dropna(subset=[key]).sort_values(key); scale=0.5 if bear_mode=="half" else 1.0
            else:
                key="pred"; gg=g.dropna(subset=[key]).sort_values(key); scale=1.0
            if len(gg)<2*K: ws.append({}); continue
            L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
            w={}
            for s in L: w[s]=w.get(s,0)+scale/K
            for s in S: w[s]=w.get(s,0)-scale/K
            ws.append(w)
        return times, ws, regs

    for tag, sub in [("3yr", apd), ("12mo", apd[apd["open_time"]>=pd.Timestamp("2025-05-01",tz="UTC")])]:
        print(f"--- {tag} (continuous held-book, net) ---")
        res={}
        for mode in ["meanrev","flat","half"]:
            times, ws, regs = build(sub, mode)
            r=heldbook(ws, times, ret_lookup)
            res[mode]=(ann(r), r.sum()*1e4)
            label={"meanrev":"H2 bear=mean-rev","flat":"FLAT bear","half":"HALF bear"}[mode]
            print(f"  {label:<20} Sharpe={res[mode][0]:+.2f}  PnL={res[mode][1]:+.0f}bps", flush=True)
        # bear-cycle PnL contribution under H2
        times, ws, regs = build(sub, "meanrev")
        r=heldbook(ws, times, ret_lookup); rr=pd.Series(r.values, index=regs)
        for rg in ["bull","side","bear"]:
            seg=rr[rr.index==rg]
            if len(seg)>5: print(f"      [{rg}] cycles={len(seg)} sumPnL={seg.sum()*1e4:+.0f}bps meanSh={ann(seg):+.2f}", flush=True)

    print(f"\nVERDICT: if FLAT-bear Sharpe > H2 → going flat in bear helps (X88 H3f was a gap artifact).")
    print(f"If H2 > FLAT → bear mean-rev genuinely contributes in the held book.")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
