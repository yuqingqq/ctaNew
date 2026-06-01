"""X87 — Attribute the HYBRID strategy's PnL by regime (bull/sideways/bear).

Question: is the hybrid's positive PnL carried by the BULL-momentum overlay, or
spread across regimes? If ~all from bull (which rests on ~3-4 episodes), the edge
is concentrated/fragile; if mean-rev in sideways/bear also contributes, more robust.

Method (cohort-level per-cycle attribution, PIT):
  - Hybrid pred per cycle: bull → mom_30d (cross-sec z), else → V0 pred (z).
  - K=3 long/short by hybrid pred. Per-cycle PnL = mean(long return_pct) - mean(short).
  - Tag each cycle bull/sideways/bear (BTC trailing-30d: >+0.10 bull, <-0.10 bear, else side).
  - Aggregate: total PnL, Sharpe, and % of total PnL per regime. 3yr + 12mo.
  - Also report the mean-rev-only and momentum-only contributions separately.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"research/convexity_portable_2026-05-20/results"; RCACHE = OUT/"_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
K = 3


def load_close(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def ann_sharpe(x):
    x=pd.Series(x).dropna()
    return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan


def main():
    t0=time.time()
    print("=== X87 hybrid PnL regime attribution ===\n", flush=True)
    apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
    apd=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)]
    syms=sorted(apd["symbol"].unique())

    # mom_30d per sym
    print("building mom_30d...", flush=True)
    mom_rows=[]
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom=(c4/c4.shift(180)-1).shift(1)  # 30d trailing at 4h bars, PIT
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":mom.index,"mom30":mom.values}))
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    apd=apd.merge(mom,on=["symbol","open_time"],how="left")

    # regime
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index()
    btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    apd=apd.merge(btc30,on="open_time",how="left").dropna(subset=["btc_ret_30d"])
    apd["regime"]=np.where(apd["btc_ret_30d"]>0.10,"bull",np.where(apd["btc_ret_30d"]<-0.10,"bear","side"))

    # per-cycle z of pred and mom
    apd["pred_z"]=apd.groupby("open_time")["pred"].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
    apd["mom_z"]=apd.groupby("open_time")["mom30"].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
    apd["hybrid"]=np.where(apd["regime"]=="bull", apd["mom_z"], apd["pred_z"])

    def cycle_pnl(sub, signal):
        rows=[]
        for ot,g in sub.groupby("open_time"):
            g=g.dropna(subset=[signal,"return_pct"])
            if len(g)<2*K: continue
            gg=g.sort_values(signal)
            pnl=gg.tail(K)["return_pct"].mean()-gg.head(K)["return_pct"].mean()
            rows.append((ot, pnl, g["regime"].iloc[0]))
        return pd.DataFrame(rows,columns=["open_time","pnl","regime"])

    for tag, sub in [("3yr", apd), ("12mo", apd[apd["open_time"]>=pd.Timestamp("2025-05-01",tz="UTC")])]:
        d=cycle_pnl(sub, "hybrid")
        tot=d["pnl"].sum()*1e4
        print(f"\n=== {tag} HYBRID per-cycle PnL attribution ===")
        print(f"  total PnL: {tot:+.0f} bps over {len(d)} cycles, overall Sharpe={ann_sharpe(d['pnl']):+.2f}")
        print(f"  {'regime':<6} {'cycles':>7} {'PnL_bps':>10} {'%of_total':>10} {'Sharpe':>8} {'mean_bps':>9}")
        for rg in ["bull","side","bear"]:
            s=d[d["regime"]==rg]
            if len(s)==0: continue
            pnl=s["pnl"].sum()*1e4
            print(f"  {rg:<6} {len(s):>7} {pnl:>+10.0f} {pnl/tot*100 if tot!=0 else 0:>9.1f}% {ann_sharpe(s['pnl']):>+8.2f} {s['pnl'].mean()*1e4:>+9.2f}", flush=True)

    print(f"\nVERDICT: %of_total from BULL tells if the edge is bull-momentum-concentrated")
    print(f"(fragile, rests on few episodes) or spread across regimes (robust).")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
