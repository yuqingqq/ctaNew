"""X94 — Alpha vs Beta decomposition of the OPTIMIZED HYBRID, per regime.

Identity: raw_spread = (long raw - short raw) = (long alpha - short alpha) + net_beta×BTC_ret
                     = ALPHA_component         + BETA_component
The hybrid switches to MOMENTUM in bull (long high-momentum = often high-beta winners),
which may flip net-beta POSITIVE in bull → bull gain could be riding the market (beta),
not alpha. Quantify alpha-gain vs beta-gain per regime (3yr, per-cycle cohort, gross).

Hybrid: bull→mom_30d, sideways→V0 mean-rev, bear→flat.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RCACHE = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
K=3


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
    print("=== X94 hybrid alpha-vs-beta decomposition by regime ===\n", flush=True)
    apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
    apd=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)]
    syms=sorted(apd["symbol"].unique())

    print("mom_30d + per-sym beta...", flush=True)
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

    rows=[]
    for ot,g in apd.groupby("open_time"):
        rg=g["regime"].iloc[0]
        if rg=="bear": continue  # flat in bear → no PnL
        key="mom30" if rg=="bull" else "pred"
        g=g.dropna(subset=[key,"alpha_A","return_pct"])
        if len(g)<2*K: continue
        gg=g.sort_values(key); L=gg.tail(K); S=gg.head(K)
        raw=L["return_pct"].mean()-S["return_pct"].mean()
        alpha=L["alpha_A"].mean()-S["alpha_A"].mean()
        beta_comp=raw-alpha
        brow=betas.loc[ot] if ot in betas.index else None
        nb=(np.nanmean([brow.get(s,np.nan) for s in L["symbol"]]) -
            np.nanmean([brow.get(s,np.nan) for s in S["symbol"]])) if brow is not None else np.nan
        rows.append((ot,rg,raw,alpha,beta_comp,nb))
    d=pd.DataFrame(rows,columns=["open_time","regime","raw","alpha","beta_comp","net_beta"])

    print(f"\n=== Per-regime decomposition (gross per-cycle, bear=flat excluded) ===")
    print(f"  {'regime':<6}{'cyc':>6}{'net_beta':>10}{'rawPnL':>9}{'ALPHA':>9}{'BETA':>9}{'alpha%':>8}{'beta%':>8}{'aSh':>7}{'bSh':>7}")
    for rg in ["bull","side","ALL"]:
        s=d if rg=="ALL" else d[d["regime"]==rg]
        if len(s)<5: continue
        rawsum=s["raw"].sum()*1e4; asum=s["alpha"].sum()*1e4; bsum=s["beta_comp"].sum()*1e4
        ap=asum/rawsum*100 if rawsum else 0; bp=bsum/rawsum*100 if rawsum else 0
        print(f"  {rg:<6}{len(s):>6}{s['net_beta'].mean():>+10.3f}{rawsum:>+9.0f}{asum:>+9.0f}{bsum:>+9.0f}{ap:>7.0f}%{bp:>7.0f}%{ann(s['alpha']):>+7.2f}{ann(s['beta_comp']):>+7.2f}", flush=True)

    print(f"\nKEY: net_beta sign by regime (bull momentum may be net-LONG-beta = riding market).")
    print(f"alpha% vs beta% = how much of raw PnL is genuine alpha vs carried beta, per regime.")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
