"""X104 — Basket model on the HL70 (70-symbol) panel.

Uses cached x64 HL70 preds (V5_mv3 + 7 crossX + filter t0.3), OOS 2025-03-30 → 2026-05-10.
CAVEAT: this window is almost entirely the POST-PEAK/decayed period (the 44-sym alpha
died ~mid-2025), so HL70 here is mostly a decayed-regime test, not a fresh-era test.

Reports:
  (1) cross-sec IC(pred, alpha_A) + top5-bot5 realized spread, by period + monthly.
  (2) held-book basket on preds: long top-K pred / short bottom-K, 24h hold (6 overlap),
      K=5, cost 4.5 bps/leg. Two variants: all-cycles, and flat-bear (BTC 30d<-10%).
Compare to the 44-sym numbers (2026 sideways IC ~0, decayed).
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
COST=4.5e-4; K=5; HOLD=6


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
    print("=== X104 basket model on HL70 ===\n", flush=True)
    d=pd.read_parquet(PREDS)
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]
    print(f"HL70 preds: {d['symbol'].nunique()} syms, {d['open_time'].min().date()}->{d['open_time'].max().date()}, {len(d):,} rows(4h)\n", flush=True)

    # BTC regime
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index()
    btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left")
    d["regime"]=np.where(d["btc_ret_30d"]>0.10,"bull",np.where(d["btc_ret_30d"]<-0.10,"bear","side"))
    d["period"]=np.where(d["open_time"]<pd.Timestamp("2026-01-01",tz="UTC"),"2025","2026")
    d["ym"]=d["open_time"].dt.strftime("%Y-%m")

    # (1) IC + spread
    recs=[]
    for ot,g in d.groupby("open_time"):
        gv=g.dropna(subset=["pred","alpha_A"])
        if len(gv)<8: continue
        ic=spearmanr(gv["pred"],gv["alpha_A"]).correlation
        spr=np.nan
        if len(gv)>=2*K:
            gg=gv.sort_values("pred"); spr=gg.tail(K)["alpha_A"].mean()-gg.head(K)["alpha_A"].mean()
        recs.append((ot,g["regime"].iloc[0],ic,spr))
    m=pd.DataFrame(recs,columns=["open_time","regime","ic","spr"])
    m["period"]=np.where(m["open_time"]<pd.Timestamp("2026-01-01",tz="UTC"),"2025","2026")
    m["ym"]=m["open_time"].dt.strftime("%Y-%m")

    print("=== cross-sec IC + top5-bot5 spread (ALL regimes / sideways-only) ===")
    print(f"  {'period':<8}{'cyc':>6}{'meanIC':>8}{'IC>0%':>7}{'sprBps':>9}{'sprSh':>7} | sideways: {'IC':>8}{'sprBps':>9}{'sprSh':>7}")
    for p in ["2025","2026"]:
        a=m[m["period"]==p]; s=a[a["regime"]=="side"]
        if len(a)<5: continue
        print(f"  {p:<8}{len(a):>6}{a['ic'].mean():>+8.3f}{(a['ic']>0).mean()*100:>6.0f}%{a['spr'].mean()*1e4:>+9.1f}{ann(a['spr']):>+7.2f} | "
              f"{s['ic'].mean():>+8.3f}{s['spr'].mean()*1e4:>+9.1f}{ann(s['spr']):>+7.2f}", flush=True)

    print("\n=== monthly cross-sec IC + spreadSharpe (ALL regimes) ===")
    print(f"  {'month':<9}{'cyc':>5}{'meanIC':>8}{'sprBps':>9}{'sprSh':>7}")
    for ym,g in m.groupby("ym"):
        print(f"  {ym:<9}{len(g):>5}{g['ic'].mean():>+8.3f}{g['spr'].mean()*1e4:>+9.1f}{ann(g['spr']):>+7.2f}", flush=True)

    # (2) held-book basket on preds
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}
    def build(flat_bear):
        ws=[]; rs=[]
        for ot in times:
            g=by_t[ot]; rg=g["regime"].iloc[0]; rs.append(dict(zip(g["symbol"],g["return_pct"])))
            if flat_bear and rg=="bear": ws.append({}); continue
            gg=g.dropna(subset=["pred"])
            if len(gg)<2*K: ws.append({}); continue
            gg=gg.sort_values("pred"); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
            w={}
            for s in L: w[s]=w.get(s,0)+1.0/K
            for s in S: w[s]=w.get(s,0)-1.0/K
            ws.append(w)
        return ws, rs
    def heldbook(ws,rs):
        prev={}; pnl=[]
        for t in range(len(ws)):
            active=ws[max(0,t-HOLD+1):t+1]; net={}
            for w in active:
                for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
            alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
            rl=rs[t]; pnl.append(sum(net.get(s,0)*rl.get(s,0.0) for s in net)-turn*0.5*COST); prev=net
        return pd.Series(pnl, index=pd.to_datetime(times))

    print("\n=== held-book basket on HL70 preds (K=5, 24h hold, mean-rev long-top/short-bot) ===")
    print(f"  {'variant':<20}{'Sharpe':>8}{'totPnL':>9}{'maxDD':>9}  per-year")
    for tag,fb in [("all-cycles",False),("flat-bear",True)]:
        ws,rs=build(fb); p=heldbook(ws,rs); pb=p*1e4; eq=pb.cumsum(); dd=(eq-eq.cummax()).min()
        yr=" ".join(f"{y}:{ann(g/1e4):+.2f}" for y,g in pb.groupby(pb.index.year))
        print(f"  {tag:<20}{ann(p):>+8.2f}{eq.iloc[-1]:>+9.0f}{dd:>+9.0f}  {yr}", flush=True)

    print(f"\nNOTE: window starts 2025-03 = mostly decayed regime. Compare 44-sym 2026 sideways IC~0. Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
