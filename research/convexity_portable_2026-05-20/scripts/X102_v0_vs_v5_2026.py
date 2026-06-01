"""X102 — Does V5 (richer features) have alpha in 2026 where V0 decayed?

X101: V0 sideways IC ≈ 0 / spread negative in 2026. Question: does the richer V5
feature set (BASE+cohort+aggT+crossX) recover an edge in 2026? Apples-to-apples on
the cached _tmp_x35a_{V0,V5}_full_preds (same 2025-05→2026-05 window, same target).

Cross-sec IC(pred, alpha_A) + top5-bot5 realized alpha_A spread, by half-period and by
month, SIDEWAYS-only and ALL, for BOTH models. If V5 2026 IC>0 & spread>0 while V0≈0,
V5 holds an edge V0 lost → worth a 3yr V5 rebuild. If both ≈0, alpha is gone regardless.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
K=5


def load_close(sym):
    sd=KLINES/sym/"5m"
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def ann(x):
    x=pd.Series(x).dropna(); return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan


def prep(fname):
    d=pd.read_parquet(RC/fname)
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]
    return d


def regime_map():
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    r=(b4/b4.shift(180)-1)
    reg=np.where(r>0.10,"bull",np.where(r<-0.10,"bear","side"))
    return pd.Series(reg, index=b4.index)


def metrics(d, reg):
    d=d.merge(reg.rename("regime").reset_index().rename(columns={"index":"open_time"}),
              on="open_time", how="left").dropna(subset=["regime"])
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
    m["period"]=np.where(m["open_time"]<pd.Timestamp("2026-01-01",tz="UTC"),"2025H2","2026")
    m["ym"]=m["open_time"].dt.strftime("%Y-%m")
    return m


def main():
    t0=time.time()
    print("=== X102 V0 vs V5 alpha in 2026 (same window) ===\n", flush=True)
    reg=regime_map()
    v0=metrics(prep("_tmp_x35a_V0_full_preds.parquet"), reg)
    v5=metrics(prep("_tmp_x35a_V5_full_preds.parquet"), reg)
    print(f"window: {v0['open_time'].min().date()} -> {v0['open_time'].max().date()}\n")

    for scope in ["side","ALL"]:
        print(f"=== scope={scope} : IC / spread(bps) / spreadSharpe by period ===")
        print(f"  {'period':<8}{'mdl':>4}{'cyc':>6}{'meanIC':>8}{'IC>0%':>7}{'sprBps':>9}{'sprSh':>7}")
        for period in ["2025H2","2026"]:
            for tag,m in [("V0",v0),("V5",v5)]:
                s=m[(m["period"]==period)]
                if scope=="side": s=s[s["regime"]=="side"]
                if len(s)<5: continue
                print(f"  {period:<8}{tag:>4}{len(s):>6}{s['ic'].mean():>+8.3f}{(s['ic']>0).mean()*100:>6.0f}%"
                      f"{s['spr'].mean()*1e4:>+9.1f}{ann(s['spr']):>+7.2f}", flush=True)
        print()

    print("=== monthly sideways spreadSharpe: V0 vs V5 ===")
    print(f"  {'month':<9}{'V0_IC':>8}{'V0_sprSh':>10}{'V5_IC':>8}{'V5_sprSh':>10}")
    months=sorted(set(v0[v0['regime']=='side']['ym']))
    for ym in months:
        a=v0[(v0['regime']=='side')&(v0['ym']==ym)]; b=v5[(v5['regime']=='side')&(v5['ym']==ym)]
        print(f"  {ym:<9}{a['ic'].mean():>+8.3f}{ann(a['spr']):>+10.2f}{b['ic'].mean():>+8.3f}{ann(b['spr']):>+10.2f}", flush=True)

    print(f"\nVERDICT: if V5 2026 IC>0 & sprSh>0 while V0≈0 → V5 recovers edge. Else alpha gone for both.")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
