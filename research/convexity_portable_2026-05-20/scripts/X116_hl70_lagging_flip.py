"""X116 — Lagging ic120 sign-flip on the HL70 panel (70 syms, 2025-03..2026-05).

Mirrors X112/X109 (regime hybrid mom-bull / mean-rev-side BN / flat-bear, K=5) but on the
cached x64 HL70 preds. Does the committed lagging flip help/hurt on the 70-sym universe?
NOTE: HL70 OOS window starts 2025-03 = mostly the decayed regime; per-period 2025(Apr+)/2026.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"
COST=4.5e-4; K=5; HOLD=6; W=120
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
    print("=== X116 HL70 lagging ic120 flip ===\n", flush=True)
    d=pd.read_parquet(PREDS, columns=["symbol","open_time","pred","alpha_A","return_pct"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy().sort_values(["symbol","open_time"])
    d["ic120"]=d.groupby("symbol",group_keys=False).apply(
        lambda g: g["pred"].rolling(W,min_periods=W//2).corr(g["alpha_A"]).shift(HOLD))
    print(f"{d['symbol'].nunique()} syms, {d['open_time'].min().date()}->{d['open_time'].max().date()}\nmom+beta...", flush=True)
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

    def hb(flip):
        ws=[]
        for ot in times:
            g=by_t[ot]; rg=g["regime"].iloc[0]
            if rg=="bear": ws.append({}); continue
            key="mom30" if rg=="bull" else "pred"; gg=g.dropna(subset=[key]).copy()
            if rg=="side" and flip:
                gg["score"]=gg["pred"]*np.where(gg["ic120"].fillna(0.0)<0,-1.0,1.0); sc="score"
            else: sc=key
            if len(gg)<2*K: ws.append({}); continue
            gg=gg.sort_values(sc); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
            a=b=1.0
            if rg=="side":
                brow=betas.loc[ot] if ot in betas.index else None
                if brow is not None:
                    mbL=np.nanmean([brow.get(s,np.nan) for s in L]); mbS=np.nanmean([brow.get(s,np.nan) for s in S])
                    if np.isfinite(mbL) and np.isfinite(mbS) and mbL>0 and mbS>0: a=2*mbS/(mbL+mbS); b=2*mbL/(mbL+mbS)
            w={}
            for s in L: w[s]=w.get(s,0)+a/K
            for s in S: w[s]=w.get(s,0)-b/K
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

    def row(name,p):
        pb=p*1e4; dd=(pb.cumsum()-pb.cumsum().cummax()).min()
        full=ann(p); s25=ann(p[p.index<Y26]/1.0); s26=ann(p[p.index>=Y26])
        print(f"  {name:<22}{full:>+7.2f}{dd:>+9.0f}   2025(Apr+):{ann(p[p.index<Y26]):+.2f}  2026:{s26:+.2f}", flush=True)
    print(f"  {'variant':<22}{'full Sh':>7}{'maxDD':>9}   per-period")
    row("base (no flip)", hb(False)); row("lagging ic120 flip", hb(True))
    print(f"\nCompare to 44-sym (base 2026 -1.19, flip +1.15) & 23-sym (base +0.71, flip +0.03). Done [{time.time()-t0:.0f}s]")


if __name__=="__main__":
    main()
