"""X101 — Is the V0 alpha actually GONE in 2026, or just the portfolio?

Measure raw SIGNAL quality independent of held-book construction:
  (1) per-cycle cross-sectional IC = rank-corr(pred, alpha_A) across symbols, by YEAR
      and by MONTH (2025-2026) — for ALL cycles and SIDEWAYS-only cycles.
  (2) tradeable spread = topK realized alpha_A − bottomK realized alpha_A (the actual
      long-short edge the model would capture), by year, K=5.
  (3) % cycles with positive IC by year (is it a sign-flip or pure noise?).
  (4) data sanity: rows/cycle, # symbols with non-NaN pred, alpha_A dispersion by year
      (rule out "alpha_A degenerate / data missing in 2026").

If 2026 IC ≈ 0 → alpha genuinely decayed. If IC>0 but spread<0 → sign-flip / tail issue.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
RCACHE = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
K=5


def load_close(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def main():
    t0=time.time()
    print("=== X101 alpha decay check (raw signal quality) ===\n", flush=True)
    apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
    apd=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)].copy()
    # regime tag
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index()
    btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    apd=apd.merge(btc30,on="open_time",how="left").dropna(subset=["btc_ret_30d"])
    apd["regime"]=np.where(apd["btc_ret_30d"]>0.10,"bull",np.where(apd["btc_ret_30d"]<-0.10,"bear","side"))
    apd["year"]=apd["open_time"].dt.year
    apd["ym"]=apd["open_time"].dt.to_period("M").astype(str)

    def cyc_metrics(g):
        gv=g.dropna(subset=["pred","alpha_A"])
        if len(gv)<8: return np.nan, np.nan
        ic=spearmanr(gv["pred"], gv["alpha_A"]).correlation
        if len(gv)>=2*K:
            gg=gv.sort_values("pred"); spr=gg.tail(K)["alpha_A"].mean()-gg.head(K)["alpha_A"].mean()
        else: spr=np.nan
        return ic, spr

    # per-cycle metrics
    recs=[]
    for ot,g in apd.groupby("open_time"):
        ic,spr=cyc_metrics(g)
        recs.append((ot, g["regime"].iloc[0], ic, spr))
    d=pd.DataFrame(recs, columns=["open_time","regime","ic","spr"])
    d["year"]=d["open_time"].dt.year; d["ym"]=d["open_time"].dt.to_period("M").astype(str)

    def ann_sh(x):
        x=pd.Series(x).dropna(); return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan

    print("=== Cross-sectional IC(pred, alpha_A) + topK-botK spread, by YEAR ===")
    print(f"  {'yr':<6}{'cyc':>6}{'meanIC':>8}{'IC>0%':>7}{'spr(bps)':>10}{'sprSh':>7} | sideways-only: {'meanIC':>8}{'sprBps':>9}{'sprSh':>7}")
    for yr,g in d.groupby("year"):
        s=g[g["regime"]=="side"]
        print(f"  {yr:<6}{len(g):>6}{g['ic'].mean():>+8.3f}{(g['ic']>0).mean()*100:>6.0f}%{g['spr'].mean()*1e4:>+10.1f}{ann_sh(g['spr']):>+7.2f} | "
              f"{s['ic'].mean():>+8.3f}{s['spr'].mean()*1e4:>+9.1f}{ann_sh(s['spr']):>+7.2f}", flush=True)

    print("\n=== Monthly sideways IC + spread (2025-09 onward) ===")
    print(f"  {'month':<9}{'cyc':>5}{'meanIC':>8}{'sprBps':>9}{'sprSh':>7}")
    dm=d[(d["regime"]=="side")&(d["open_time"]>=pd.Timestamp("2025-09-01",tz="UTC"))]
    for ym,g in dm.groupby("ym"):
        print(f"  {ym:<9}{len(g):>5}{g['ic'].mean():>+8.3f}{g['spr'].mean()*1e4:>+9.1f}{ann_sh(g['spr']):>+7.2f}", flush=True)

    print("\n=== Data sanity by year (rule out degeneracy/missing data) ===")
    print(f"  {'yr':<6}{'avg#syms/cyc':>14}{'avg#pred/cyc':>14}{'alpha_A std(bps)':>18}{'pred std':>10}")
    for yr,g in apd.groupby("year"):
        per=g.groupby("open_time")
        nsym=per.size().mean(); npred=per["pred"].apply(lambda s:s.notna().sum()).mean()
        astd=g["alpha_A"].std()*1e4; pstd=g["pred"].std()
        print(f"  {yr:<6}{nsym:>14.1f}{npred:>14.1f}{astd:>18.1f}{pstd:>10.3f}", flush=True)

    print(f"\nVERDICT: 2026 sideways meanIC near 0 or negative → alpha decayed. Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
