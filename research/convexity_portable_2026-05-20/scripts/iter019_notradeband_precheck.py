"""iter-019 PRE-CHECK #2 — transaction-cost-aware no-trade band on the HL70 held-book.

SOTA: cost-aware portfolio construction (Baldi-Lanfranchi 2024 FoFI; arXiv:2412.11575).
Mechanism: a no-trade band suppresses small rebalances of names that churn near the
K-cutoff across cycles (marginal-name turnover that pays cost with ~no signal — iter-016
showed sleeves 2-6 hold stale, mildly-anti-signal positions). This is NOT a prediction
(no G4-placeholder wall) and NOT the rejected pred-unit cost-margin swap (Phase K) — it is
a structural turnover reducer on the realized NET weight changes.

This pre-check (cheap, no nested-OOS yet):
  1. Decompose held-book turnover: how much is "core" (large persistent weights) vs
     "churn" (small in/out flips at the rank boundary)?
  2. Apply a candidate no-trade band: only execute a per-symbol weight change if
     |net_t - executed_{t-1}| >= band; else hold previous weight. Measure realized
     turnover reduction and PnL/Sharpe/maxDD impact at 4.5bps over a band sweep.
  3. PRE-CHECK pass: a band exists that cuts turnover materially while leaving gross
     PnL ~flat (cost-only saving) -> Calmar up. If every band that cuts cost also drops
     gross PnL proportionally (band drifts the book off-signal), it's a wash -> drop.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
K=5; HOLD=6

def load_close(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return None
    df=pd.concat([pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))],
                 ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)

def ann(x):
    x=pd.Series(x).dropna(); return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan

def main():
    t0=time.time()
    print("=== iter-019 no-trade-band PRE-CHECK (HL70) ===\n",flush=True)
    d=pd.read_parquet(PREDS,columns=["symbol","open_time","pred","return_pct"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy()
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    syms=sorted(d["symbol"].unique()); mom_rows=[]; beta_map={}
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,"mom30":(c4/c4.shift(180)-1).shift(1).values}))
        r=np.log(c4/c4.shift(1)); ri,bi=r.align(br,join="inner")
        beta_map[sym]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    betas=pd.concat([s.rename(k) for k,s in beta_map.items()],axis=1)
    d=d.merge(mom,on=["symbol","open_time"],how="left")
    btc30=(b4/b4.shift(180)-1).to_frame("b30").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left").dropna(subset=["b30"])
    d["regime"]=np.where(d["b30"]>0.10,"bull",np.where(d["b30"]<-0.10,"bear","side"))
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}

    cyc_w=[]
    for ot in times:
        g=by_t[ot]; rg=g["regime"].iloc[0]
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

    # net target weights each cycle (the 6-sleeve average) -- this is what the book WANTS to hold
    nets=[]
    for t in range(len(cyc_w)):
        active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        nets.append(net)

    # 1) turnover decomposition (baseline, band=0): per-trade size distribution
    base_trades=[]
    prev={}
    for net in nets:
        alls=set(net)|set(prev)
        for s in alls:
            ch=abs(net.get(s,0)-prev.get(s,0))
            if ch>1e-9: base_trades.append(ch)
        prev=net
    bt=np.array(base_trades)
    full_leg=1.0/K  # 0.20 = one full sleeve-leg
    print(f"baseline trades: {len(bt)}  total turnover {bt.sum():.1f}")
    for q in [0.25,0.5,0.75,0.9]:
        print(f"  trade-size p{int(q*100)}: {np.quantile(bt,q):.4f}  ({np.quantile(bt,q)/full_leg*100:.0f}% of a full leg)")
    small=bt[bt < 0.5*(full_leg/HOLD*2)]  # heuristic 'churn' threshold
    print(f"  small (<~1/2 sleeve-step) trades: {len(small)} = {len(small)/len(bt)*100:.0f}% of trades, {small.sum()/bt.sum()*100:.0f}% of turnover\n",flush=True)

    # 2) held-book with a per-symbol NO-TRADE BAND on net weight changes
    def heldbook_band(cost, band):
        prev={}       # last EXECUTED weights
        pnl=[]
        for t in range(len(nets)):
            target=nets[t]; alls=set(target)|set(prev); exe=dict(prev)
            turn=0.0
            for s in alls:
                tg=target.get(s,0.0); pv=prev.get(s,0.0)
                if abs(tg-pv) >= band:
                    exe[s]=tg; turn+=abs(tg-pv)
                else:
                    exe[s]=pv  # hold; no trade
            exe={s:w for s,w in exe.items() if abs(w)>1e-9}
            rl=by_t[times[t]]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
            gross=sum(exe.get(s,0)*rmap.get(s,0.0) for s in exe if np.isfinite(rmap.get(s,0.0)))
            pnl.append(gross - turn*0.5*cost); prev=exe
        return pd.Series(pnl,index=pd.to_datetime(times))

    def gross_turn(band):  # diagnostics at band
        prev={}; tot=0.0; g=0.0
        for t in range(len(nets)):
            target=nets[t]; alls=set(target)|set(prev); exe=dict(prev)
            for s in alls:
                tg=target.get(s,0.0); pv=prev.get(s,0.0)
                if abs(tg-pv)>=band: exe[s]=tg; tot+=abs(tg-pv)
                else: exe[s]=pv
            exe={s:w for s,w in exe.items() if abs(w)>1e-9}
            rl=by_t[times[t]]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
            g+=sum(exe.get(s,0)*rmap.get(s,0.0) for s in exe if np.isfinite(rmap.get(s,0.0)))
            prev=exe
        return tot, g*1e4

    print("=== no-trade band sweep @4.5bps (band in weight units; full leg = 0.200) ===")
    print(f"  {'band':>7}{'Sharpe':>8}{'totPnL':>9}{'maxDD':>9}{'turnover':>10}{'grossPnL':>10}")
    base_turn,base_gross=gross_turn(0.0)
    for band in [0.0,0.005,0.01,0.02,0.03,0.05,0.08]:
        p=heldbook_band(4.5e-4,band); pb=p*1e4; e=pb.cumsum()
        tot,g=gross_turn(band)
        print(f"  {band:>7.3f}{ann(p):>+8.2f}{e.iloc[-1]:>+9.0f}{(e-e.cummax()).min():>+9.0f}{tot:>10.1f}{g:>+10.0f}",flush=True)

    print(f"\nNOTE: baseline turnover {base_turn:.1f}, gross {base_gross:+.0f}bps.")
    print("PASS if a band cuts turnover (=>totPnL up) with gross ~flat (cost-only saving).")
    print(f"\nDone [{time.time()-t0:.0f}s]")

if __name__=="__main__":
    main()
