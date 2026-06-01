"""X106 — Per-symbol sign-flip layer + ORACLE ceiling: is 2026 a sign problem or dead?

User intuition: |IC| is intact but sign flipped → flip the sign to recover. X105 flipped
on CROSS-SECTIONAL IC (~0 in 2026 → nothing to flip). X103's flip is PER-SYMBOL. So test
a proper per-symbol sign-flip layer: pred_adj[i] = pred[i] * sign(trailing per-symbol IC_i).

DECISIVE diagnostic = ORACLE: use each symbol's TRUE 2026 sign (look-ahead).
  - If oracle rescues 2026 but realized(PIT) doesn't → estimation/lag problem (signal exists).
  - If oracle ALSO fails → no exploitable cross-sec magnitude even with perfect signs → dead.

Variants (held-book K=5, 24h, flat-bear, 44-sym):
  base               : pred (long top)
  persym W=30/60     : pred * sign(rolling-W per-sym IC, PIT lag HOLD)
  ORACLE persym      : pred * sign(per-sym full-2026 IC)            [look-ahead ceiling]
  ORACLE global/cyc  : pred * sign(this-cycle cross-sec IC)          [look-ahead ceiling]
Report per-year Sharpe + 2026 cross-sec IC of the adjusted signal.
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
    print("=== X106 per-symbol sign-flip + oracle ===\n", flush=True)
    d=pd.read_parquet(RC/"x70_v0_3yr_preds.parquet", columns=["symbol","open_time","pred","alpha_A","return_pct"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy()
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left")
    d["regime"]=np.where(d["btc_ret_30d"]>0.10,"bull",np.where(d["btc_ret_30d"]<-0.10,"bear","side"))
    d=d.sort_values(["symbol","open_time"])
    print(f"{d['symbol'].nunique()} syms, {d['open_time'].min().date()}->{d['open_time'].max().date()}, {len(d):,} rows\n", flush=True)

    is2026=d["open_time"]>=pd.Timestamp("2026-01-01",tz="UTC")
    # per-symbol rolling IC (Pearson on values as fast sign proxy), PIT lag HOLD
    def add_rolling_ic(g, W):
        ic=g["pred"].rolling(W, min_periods=max(20,W//2)).corr(g["alpha_A"])
        return ic.shift(HOLD)
    for W in [30,60]:
        d[f"ic{W}"]=d.groupby("symbol",group_keys=False).apply(lambda g: add_rolling_ic(g,W))
    # oracle per-symbol sign over 2026
    osign={}
    for sym,g in d[is2026].groupby("symbol"):
        gv=g.dropna(subset=["pred","alpha_A"])
        c=spearmanr(gv["pred"],gv["alpha_A"]).correlation if len(gv)>=30 else np.nan
        osign[sym]=np.sign(c) if np.isfinite(c) and c!=0 else 1.0
    d["osign_persym"]=d["symbol"].map(osign).fillna(1.0)

    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}
    # oracle global per-cycle sign = sign of THIS cycle cross-sec IC (look-ahead)
    gsign={}
    for ot,g in by_t.items():
        gv=g.dropna(subset=["pred","alpha_A"])
        c=spearmanr(gv["pred"],gv["alpha_A"]).correlation if len(gv)>=8 else np.nan
        gsign[ot]=np.sign(c) if np.isfinite(c) and c!=0 else 1.0

    def heldbook(adj_col, use_gsign=False):
        # build weights then run held-book
        ws=[]
        for ot in times:
            g=by_t[ot]
            if g["regime"].iloc[0]=="bear": ws.append({}); continue
            gg=g.dropna(subset=[adj_col]) if adj_col!="pred_g" else g.dropna(subset=["pred"])
            if adj_col=="pred_g":
                gg=g.dropna(subset=["pred"]).copy(); gg["score"]=gg["pred"]*gsign[ot]
            else:
                gg=g.dropna(subset=[adj_col]).copy(); gg["score"]=gg[adj_col]
            if len(gg)<2*K: ws.append({}); continue
            gg=gg.sort_values("score"); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
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
        return pd.Series(pnl, index=pd.to_datetime(times))

    # build adjusted score columns
    d["pred_w30"]=d["pred"]*np.sign(d["ic30"]).fillna(1.0)
    d["pred_w60"]=d["pred"]*np.sign(d["ic60"]).fillna(1.0)
    d["pred_oraclesym"]=d["pred"]*d["osign_persym"]
    by_t={ot:g for ot,g in d.groupby("open_time")}  # refresh with new cols

    def cs_ic_2026(col):
        ics=[]
        for ot,g in by_t.items():
            if ot<pd.Timestamp("2026-01-01",tz="UTC"): continue
            gv=g.dropna(subset=[col,"alpha_A"])
            if len(gv)>=8:
                c=spearmanr(gv[col],gv["alpha_A"]).correlation
                if np.isfinite(c): ics.append(c)
        return np.mean(ics) if ics else np.nan

    def report(name, pnl, ic2026):
        pb=pnl*1e4; eq=pb.cumsum(); dd=(eq-eq.cummax()).min()
        yr={y:ann(g/1e4) for y,g in pb.groupby(pb.index.year)}
        s=" ".join(f"{y}:{yr.get(y,float('nan')):+.2f}" for y in [2023,2024,2025,2026])
        print(f"  {name:<26}{ann(pnl):>+7.2f}{dd:>+9.0f}{ic2026:>+9.3f}   {s}", flush=True)

    print(f"  {'variant':<26}{'Sharpe':>7}{'maxDD':>9}{'IC2026':>9}   per-year Sharpe")
    report("base pred", heldbook("pred"), cs_ic_2026("pred"))
    report("persym flip W=30 (PIT)", heldbook("pred_w30"), cs_ic_2026("pred_w30"))
    report("persym flip W=60 (PIT)", heldbook("pred_w60"), cs_ic_2026("pred_w60"))
    report("ORACLE persym (cheat)", heldbook("pred_oraclesym"), cs_ic_2026("pred_oraclesym"))
    report("ORACLE global/cyc (cheat)", heldbook("pred_g", use_gsign=True), np.nan)

    print(f"\nREAD: if ORACLE rescues 2026 but PIT doesn't → estimation/lag problem (signal exists).")
    print(f"If ORACLE also fails → no exploitable cross-sec magnitude → genuinely dead. Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
