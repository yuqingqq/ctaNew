"""X107 — Can ANY realizable estimator track the per-symbol sign? Hit-rate vs oracle.

X106: ORACLE per-sym sign rescues 2026 (+4.73) but trailing-IC PIT flip fails (-2.45).
Question: is the trailing estimator just one bad choice, or is per-sym sign fundamentally
un-trackable in real time? Measure, for 2026, each estimator's HIT-RATE vs the oracle
per-symbol sign (one sign/symbol over 2026), and the resulting 2026 held-book Sharpe.

Estimators (all PIT, lag HOLD):
  ic_W{30,60,90}   : sign(trailing-W per-sym Spearman(pred,alpha))
  ret_W{30,60}     : sign(-trailing mean per-sym alpha)  [is the sym's recent alpha +/-?]
  shrink_W60       : sign of (per-sym ic60 shrunk toward cross-sym mean ic60)
Hit-rate ~50% for all → sign un-trackable (estimation wall). >60% for some → worth building.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
COST=4.5e-4; K=5; HOLD=6
Y26=pd.Timestamp("2026-01-01",tz="UTC")


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
    print("=== X107 sign-estimator hit-rate vs oracle ===\n", flush=True)
    d=pd.read_parquet(RC/"x70_v0_3yr_preds.parquet", columns=["symbol","open_time","pred","alpha_A","return_pct"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy()
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left")
    d["regime"]=np.where(d["btc_ret_30d"]>0.10,"bull",np.where(d["btc_ret_30d"]<-0.10,"bear","side"))
    d=d.sort_values(["symbol","open_time"])

    # estimators (PIT lag HOLD)
    def roll_ic(g,W): return g["pred"].rolling(W,min_periods=max(20,W//2)).corr(g["alpha_A"]).shift(HOLD)
    for W in [30,60,90]:
        d[f"ic{W}"]=d.groupby("symbol",group_keys=False).apply(lambda g: roll_ic(g,W))
    for W in [30,60]:
        d[f"ret{W}"]=d.groupby("symbol",group_keys=False).apply(lambda g:(-g["alpha_A"].rolling(W,min_periods=max(20,W//2)).mean()).shift(HOLD))
    # shrinkage: ic60 shrunk toward cross-sym mean ic60 at each time
    cs_mean_ic60=d.groupby("open_time")["ic60"].transform("mean")
    d["shrink60"]=0.5*d["ic60"]+0.5*cs_mean_ic60

    # oracle per-sym 2026 sign
    osign={}
    for sym,g in d[d["open_time"]>=Y26].groupby("symbol"):
        gv=g.dropna(subset=["pred","alpha_A"])
        c=spearmanr(gv["pred"],gv["alpha_A"]).correlation if len(gv)>=30 else np.nan
        osign[sym]=np.sign(c) if np.isfinite(c) and c!=0 else 1.0
    d["osign"]=d["symbol"].map(osign).fillna(1.0)

    ests=["ic30","ic60","ic90","ret30","ret60","shrink60"]
    # hit-rate vs oracle on 2026 rows where estimator defined
    print(f"  {'estimator':<12}{'hit% vs oracle':>16}{'%flip-from+1':>14}   (2026)")
    d26=d[d["open_time"]>=Y26]
    for e in ests:
        sub=d26.dropna(subset=[e])
        hit=(np.sign(sub[e])==sub["osign"]).mean()*100
        flip=(np.sign(sub[e])<0).mean()*100
        print(f"  {e:<12}{hit:>15.1f}%{flip:>13.1f}%", flush=True)

    # 2026 held-book Sharpe per estimator
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}
    def hb(scorecol):
        ws=[]
        for ot in times:
            g=by_t[ot]
            if g["regime"].iloc[0]=="bear": ws.append({}); continue
            gg=g.dropna(subset=[scorecol]).copy()
            if len(gg)<2*K: ws.append({}); continue
            gg=gg.sort_values(scorecol); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
            w={}
            for s in L: w[s]=w.get(s,0)+1.0/K
            for s in S: w[s]=w.get(s,0)-1.0/K
            ws.append(w)
        prev={}; pnl=[]
        for t in range(len(ws)):
            active=ws[max(0,t-HOLD+1):t+1]; net={}
            for w in active:
                for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
            alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
            rl=by_t[times[t]]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
            pnl.append(sum(net.get(s,0)*rmap.get(s,0.0) for s in net if np.isfinite(rmap.get(s,0.0)))-turn*0.5*COST); prev=net
        return pd.Series(pnl,index=pd.to_datetime(times))

    for e in ests: d[f"adj_{e}"]=d["pred"]*np.sign(d[e]).fillna(1.0)
    d["adj_base"]=d["pred"]; d["adj_oracle"]=d["pred"]*d["osign"]
    by_t={ot:g for ot,g in d.groupby("open_time")}
    print(f"\n  {'variant':<14}{'2026 Sharpe':>12}{'full Sharpe':>12}")
    for tag in ["base"]+ests+["oracle"]:
        p=hb(f"adj_{tag}"); p26=p[p.index>=Y26]
        print(f"  {tag:<14}{ann(p26):>+12.2f}{ann(p):>+12.2f}", flush=True)

    print(f"\nREAD: hit-rate ~50% = sign un-trackable (coin flip). Even best est 2026 Sharpe vs oracle +4.73. Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
