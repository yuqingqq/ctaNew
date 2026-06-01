"""X84 — Decompose basket PnL into ALPHA vs BETA, and test position-level
beta-neutral re-weighting (clean: ranking unchanged, only leg weights adjusted).

Answers: do we make money from cross-sectional alpha, or from carrying net beta?

Key identity: a basket's realized raw return decomposes as
    raw_spread = (long raw - short raw)
               = (long alpha - short alpha)  +  net_beta × BTC_return
               = ALPHA_component             +  BETA_component
Both pieces are directly observable: return_pct (raw fwd) and alpha_A (residual)
are already in the prediction parquets; BTC_return per cycle from klines.

Tests on V0_single and V0_KMeans_routed_K5 (the +1.07 winner), K=3:
  1. Decompose per-cycle PnL into alpha vs beta; Sharpe of each, overall + by regime.
  2. Position-level beta-neutral: scale long/short leg $ so net_beta=0 (ranking
     UNCHANGED), recompute raw PnL → if profit survives, it's alpha; if it vanishes,
     it's beta.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"research/convexity_portable_2026-05-20/results"; RCACHE = OUT/"_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
COST = 4.5e-4


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


def build_betas(syms):
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    btc_fwd=(b4.shift(-1)/b4-1)  # 4h fwd raw BTC return (for beta component)
    out=[]
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]; r=np.log(c4/c4.shift(1))
        ri,bi=r.align(br,join="inner")
        beta=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
        out.append(pd.DataFrame({"symbol":sym,"open_time":beta.index,"beta":beta.values}))
    betas=pd.concat(out,ignore_index=True); betas["open_time"]=pd.to_datetime(betas["open_time"],utc=True)
    btcf=btc_fwd.to_frame("btc_fwd").reset_index(); btcf["open_time"]=pd.to_datetime(btcf["open_time"],utc=True)
    return betas, btcf


def main():
    print("=== X84 alpha-vs-beta decomposition + position-level neutralization ===\n", flush=True)
    # regime
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    r30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index(); r30["open_time"]=pd.to_datetime(r30["open_time"],utc=True)

    cases={"V0_single":RCACHE/"x70_v0_3yr_preds.parquet",
           "V0_KMeans_routed_K5":RCACHE/"x75_routed_K5_preds.parquet"}
    # build betas once (union of syms)
    any_apd=pd.read_parquet(cases["V0_single"]); syms=sorted(any_apd["symbol"].unique())
    print("building per-sym betas + btc_fwd...", flush=True)
    betas, btcf = build_betas(syms)

    K=3
    for name,pth in cases.items():
        if not Path(pth).exists(): print(f"{name}: missing"); continue
        apd=pd.read_parquet(pth); apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
        apd=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)]
        m=apd.merge(betas,on=["symbol","open_time"],how="left").merge(btcf,on="open_time",how="left")
        rows=[]
        for ot,g in m.groupby("open_time"):
            g=g.dropna(subset=["pred","alpha_A","return_pct","beta"])
            if len(g)<2*K: continue
            gg=g.sort_values("pred"); L=gg.tail(K); S=gg.head(K)
            raw=L["return_pct"].mean()-S["return_pct"].mean()
            alpha=L["alpha_A"].mean()-S["alpha_A"].mean()
            beta_comp=raw-alpha
            net_beta=L["beta"].mean()-S["beta"].mean()
            # position-level beta-neutral: weight legs so a*meanβL = b*meanβS, a+b=2
            bL,bS=L["beta"].mean(),S["beta"].mean()
            if abs(bL)+abs(bS)>1e-6 and (bL>0 and bS>0):
                a=2*bS/(bL+bS); b=2*bL/(bL+bS)  # a*bL=b*bS
            else:
                a=b=1.0
            raw_neutral=a*L["return_pct"].mean()-b*S["return_pct"].mean()
            rows.append((ot,raw,alpha,beta_comp,net_beta,raw_neutral))
        d=pd.DataFrame(rows,columns=["open_time","raw","alpha","beta_comp","net_beta","raw_neutral"])
        d=d.merge(r30,on="open_time",how="left")
        d["regime"]=np.where(d["btc_ret_30d"]>0.10,"bull",np.where(d["btc_ret_30d"]<-0.10,"bear","side"))

        print(f"\n=== {name} (K={K}) ===")
        print(f"  component    mean_bps   Sharpe")
        print(f"  raw         {d['raw'].mean()*1e4:>+9.2f} {ann_sharpe(d['raw']):>+8.2f}")
        print(f"  ALPHA       {d['alpha'].mean()*1e4:>+9.2f} {ann_sharpe(d['alpha']):>+8.2f}")
        print(f"  BETA_comp   {d['beta_comp'].mean()*1e4:>+9.2f} {ann_sharpe(d['beta_comp']):>+8.2f}")
        print(f"  raw_neutral {d['raw_neutral'].mean()*1e4:>+9.2f} {ann_sharpe(d['raw_neutral']):>+8.2f}  (position-level beta=0)")
        print(f"  corr(raw, alpha)={d['raw'].corr(d['alpha']):.3f}  corr(raw, beta_comp)={d['raw'].corr(d['beta_comp']):.3f}")
        print(f"  by regime (raw / alpha / beta_comp Sharpe):")
        for rg in ["bull","side","bear"]:
            s=d[d["regime"]==rg]
            if len(s)<30: continue
            print(f"    {rg:<5} n={len(s):>5}  raw={ann_sharpe(s['raw']):>+6.2f}  alpha={ann_sharpe(s['alpha']):>+6.2f}  beta={ann_sharpe(s['beta_comp']):>+6.2f}")
        d.to_parquet(OUT/f"X84_{name}_decomp.parquet",index=False)

    print(f"\nINTERPRETATION:")
    print(f"  If ALPHA Sharpe >> BETA_comp Sharpe AND raw_neutral ≈ raw → profit is ALPHA.")
    print(f"  If BETA_comp carries it AND raw_neutral << raw → we're carrying beta.")


if __name__ == "__main__":
    main()
