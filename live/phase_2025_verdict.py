"""Decisive verdict for the opt2025 loop: rank K-breadth configs by ROBUSTNESS, not single-metric peak.

For each config: daily dense Sharpe, 2025 Sharpe, maxDD, folds_beat/54, AND tail metrics on per-cycle pnl
(kurtosis, CVaR5 = mean worst-5% cycles, worst single cycle) — because the small-K leaders (ks2_kl2 etc.)
buy Sharpe with concentration, and we must see whether the tail got worse. Then nested-OOS over the full
K grid (pick by past-fold cumulative pnl -> apply forward) — the honest test a discrete K choice should pass.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO=Path("/home/yuqing/ctaNew"); OUT=REPO/"live/state/v3loop/opt2025"
ANN=np.sqrt(365)
def load(tag):
    f=OUT/tag/"cyc_fold.csv"
    if not f.exists(): return None
    c=pd.read_csv(f); c["open_time"]=pd.to_datetime(c["open_time"],utc=True)
    return c.sort_values("open_time").set_index("open_time")
def dsh(s):
    d=(s.fillna(0)/1e4).resample("1D").sum(); return float(d.mean()/d.std()*ANN) if d.std()>0 else np.nan
def maxdd(s):
    eq=s.fillna(0).cumsum(); return float((eq-eq.cummax()).min())
def tail(s):  # per-cycle pnl tail
    x=s.dropna().to_numpy()
    if len(x)<50: return (np.nan,np.nan,np.nan)
    k=float(pd.Series(x).kurtosis()); cv=float(np.mean(np.sort(x)[:max(1,len(x)//20)])); return (k,cv,float(x.min()))
def fp(c): return c.groupby("fold")["pnl_bps"].sum()

led=pd.read_csv(OUT/"ledger.csv")
KTAGS=[t for t in led[led.phase.isin(["P3","P3b"])]["tag"]]   # all K-breadth configs
base=load("baseline"); bf=fp(base)
b25=base.loc['2025-01-01':'2025-12-31','pnl_bps']; bdense=base.loc['2025-01-01':'2026-06-04','pnl_bps']
print(f"{'tag':10s} {'dense':>7s} {'2025':>7s} {'maxDD':>7s} {'folds':>6s} {'kurt':>6s} {'CVaR5':>7s} {'worst':>7s}")
bk,bcv,bw=tail(base['pnl_bps'])
print(f"{'baseline':10s} {dsh(bdense):+7.3f} {dsh(b25):+7.3f} {maxdd(bdense):+7.0f} {'--':>6s} {bk:6.1f} {bcv:+7.0f} {bw:+7.0f}")
rows=[]
for t in KTAGS:
    c=load(t)
    if c is None: continue
    cf=fp(c); common=bf.index.intersection(cf.index); beat=int((cf[common]>bf[common]).sum())
    d25=c.loc['2025-01-01':'2025-12-31','pnl_bps']; dd=c.loc['2025-01-01':'2026-06-04','pnl_bps']
    k,cv,w=tail(c['pnl_bps'])
    rows.append((t,dsh(dd),dsh(d25),maxdd(dd),f"{beat}/{len(common)}",k,cv,w))
for t,d,s25,md,fb,k,cv,w in sorted(rows,key=lambda r:-r[1]):
    print(f"{t:10s} {d:+7.3f} {s25:+7.3f} {md:+7.0f} {fb:>6s} {k:6.1f} {cv:+7.0f} {w:+7.0f}")

# ---- nested-OOS over the full K grid ----
def nested(cands):
    series={t:load(t) for t in ["baseline"]+cands}; series={k:v for k,v in series.items() if v is not None}
    folds=sorted(series["baseline"]["fold"].dropna().unique()); fps={t:fp(c) for t,c in series.items()}
    parts=[]; picks=[]
    for i,f in enumerate(folds):
        if i==0: pk="baseline"
        else:
            past=folds[:i]; cum={t:fps[t].reindex(past).fillna(0).sum() for t in series}; pk=max(cum,key=cum.get)
        picks.append(pk); parts.append(series[pk][series[pk]["fold"]==f]["pnl_bps"])
    nz=pd.concat(parts).sort_index()
    from collections import Counter
    print(f"\nNESTED-OOS (full K grid, {len(cands)} cands): 2025 {dsh(nz.loc['2025-01-01':'2025-12-31']):+.3f} (base {dsh(b25):+.3f})  "
          f"dense {dsh(nz.loc['2025-01-01':'2026-06-04']):+.3f} (base {dsh(bdense):+.3f})  maxDD {maxdd(nz.loc['2025-01-01':'2026-06-04']):+.0f}")
    print("  picks:",dict(Counter(picks)))
nested(KTAGS)
# market-neutral-only subset (exclude directional kl0 short-only)
mn=[t for t in KTAGS if not t.endswith("kl0")]
print("\n[market-neutral only — exclude short-only kl0]"); nested(mn)
